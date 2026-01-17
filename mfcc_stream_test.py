import numpy as np
import onnxruntime as ort
import torchaudio
import torch
import os
import glob
import argparse
from tqdm import tqdm

# --- Configuration (Must match mfcc_stream.py) ---
SAMPLE_RATE = 16000
N_MFCC = 20      # The input size for the model
N_MELS = 40      # Internal filterbank for MFCC calc
N_FFT = 480      # 30ms
HOP_LENGTH = 480 # 30ms (Low Compute)
WAKEWORD_IDX = 2 # 0=Silence, 1=Unknown, 2=Wakeword

# Threshold to consider it a detection for the summary report
DETECTION_THRESHOLD = 0.85 

# Padding to ensure the end of the file is processed
PADDING_DURATION_SEC = 0.5 

class OnnxStreamingTester:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        # print(f"Loading ONNX model: {model_path}")
        self.sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # --- Audio Preprocessing (MFCC) ---
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            melkwargs={
                'n_fft': N_FFT,
                'hop_length': HOP_LENGTH,
                'n_mels': N_MELS,
                'center': False, 
                'power': 2.0
            }
        )
        
        # Auto-detect input names from the ONNX file
        self.input_name = self.sess.get_inputs()[0].name
        self.state_names = [n.name for n in self.sess.get_inputs() if 'state' in n.name]
        self.output_names = [n.name for n in self.sess.get_outputs()]
        
        # Sort state names to ensure correct order (state_in_0, state_in_1...)
        self.state_names.sort(key=lambda x: int(x.split('_')[-1]))
        
        self.reset_states()

    def reset_states(self):
        """Creates zero-filled state buffers matching the model's expected shape."""
        self.states = []
        for state_name in self.state_names:
            # Get shape from ONNX metadata
            for input_meta in self.sess.get_inputs():
                if input_meta.name == state_name:
                    # Shape is usually [Batch, Channels, Height, Width]
                    shape = input_meta.shape
                    # Handle dynamic batch size (usually 'batch' or string in dim 0)
                    shape[0] = 1 
                    self.states.append(np.zeros(shape, dtype=np.float32))
                    break

    def process_file(self, filepath):
        """Runs streaming inference on a single audio file."""
        self.reset_states()
        
        # Load and pad audio
        waveform, sr = torchaudio.load(filepath)
        if sr != SAMPLE_RATE:
            transform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = transform(waveform)
            
        # Add padding to catch the tail end of words
        padding = torch.zeros(1, int(SAMPLE_RATE * PADDING_DURATION_SEC))
        waveform = torch.cat([waveform, padding], dim=1)

        # --- Generate Features (MFCC) ---
        features = self.mfcc_transform(waveform)
        
        # Features shape: [1, 20, TimeSteps]
        num_frames = features.shape[2]
        
        max_score = 0.0
        detected_class = 0
        
        # --- Streaming Loop ---
        for t in range(num_frames):
            # FIX: Add the Channel dimension!
            # Old: [1, 20, 1] -> Rank 3 (Error)
            # New: [1, 1, 20, 1] -> Rank 4 (Correct)
            frame = features[:, :, t:t+1].unsqueeze(1).numpy().astype(np.float32)
            
            # Prepare Inputs
            inputs = {self.input_name: frame}
            for i, name in enumerate(self.state_names):
                inputs[name] = self.states[i]
            
            # Run Inference
            outputs = self.sess.run(self.output_names, inputs)
            
            # Outputs: [logits, state_out_0, state_out_1, ...]
            logits = outputs[0]
            self.states = outputs[1:] # Update states for next frame
            
            # Softmax
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            score = probs[0][WAKEWORD_IDX]
            pred_class = np.argmax(probs)
            
            if score > max_score:
                max_score = score
                detected_class = pred_class
                
        return max_score, detected_class

def main():
    parser = argparse.ArgumentParser(description="Test Streaming MFCC Model (20 Bins)")
    parser.add_argument('--model', type=str, default='bcresnet_mfcc_stream.onnx', help='Path to ONNX model')
    parser.add_argument('--dataset', type=str, default='./dataset/testing', help='Path to testing dataset')
    args = parser.parse_args()

    tester = OnnxStreamingTester(args.model)
    
    # Stats tracking
    stats = {
        'silence': {'total': 0, 'false_positives': 0},
        'unknown': {'total': 0, 'false_positives': 0},
        'wakeword': {'total': 0, 'detected': 0}
    }
    
    print(f"\n🔍 Scanning dataset at {args.dataset}...")
    
    wav_files = glob.glob(os.path.join(args.dataset, '**', '*.wav'), recursive=True)
    if not wav_files:
        print("❌ No wav files found!")
        return

    # Create detailed log file
    with open("mfcc_streaming_results.txt", "w") as log:
        log.write("Filepath,Max_Wakeword_Score,Predicted_Class_At_Max\n")
        
        for wav in tqdm(wav_files):
            try:
                score, pred_class = tester.process_file(wav)
                is_detected = (score > DETECTION_THRESHOLD)
                
                # Relative path for cleaner logging
                rel_path = os.path.relpath(wav, args.dataset)
                log.write(f"{rel_path},{score:.4f},{pred_class}\n")
                
                # Update Stats
                if 'wakeword' in rel_path.lower():
                    stats['wakeword']['total'] += 1
                    if is_detected: stats['wakeword']['detected'] += 1
                elif 'unknown' in rel_path.lower():
                    stats['unknown']['total'] += 1
                    if is_detected: stats['unknown']['false_positives'] += 1
                elif 'silence' in rel_path.lower():
                    stats['silence']['total'] += 1
                    if is_detected: stats['silence']['false_positives'] += 1
                    
            except Exception as e:
                print(f"\n⚠️ Error: {wav} - {e}")

    # --- Summary Report ---
    print("\n" + "="*40)
    print("📊 MFCC STREAMING TEST SUMMARY")
    print("="*40)
    
    if stats['wakeword']['total'] > 0:
        ww_rate = 100 * stats['wakeword']['detected'] / stats['wakeword']['total']
        print(f"✅ True Positives (Wakeword): {ww_rate:.1f}% ({stats['wakeword']['detected']}/{stats['wakeword']['total']})")
    
    if stats['unknown']['total'] > 0:
        fa_rate = 100 * stats['unknown']['false_positives'] / stats['unknown']['total']
        print(f"❌ False Positives (Unknown):  {fa_rate:.1f}% ({stats['unknown']['false_positives']}/{stats['unknown']['total']})")
        
    if stats['silence']['total'] > 0:
        fa_sil = 100 * stats['silence']['false_positives'] / stats['silence']['total']
        print(f"❌ False Positives (Silence):  {fa_sil:.1f}% ({stats['silence']['false_positives']}/{stats['silence']['total']})")
        
    print("\n📄 Detailed results saved to: mfcc_streaming_results.txt")

if __name__ == "__main__":
    main()
