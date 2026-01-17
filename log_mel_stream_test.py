import numpy as np
import onnxruntime as ort
import torchaudio
import torch
import os
import glob
import argparse
from tqdm import tqdm

# --- Configuration (Must match train.py) ---
SAMPLE_RATE = 16000
N_MELS = 40
N_FFT = 480      # 30ms
HOP_LENGTH = 480 # 30ms (Low Compute)
WAKEWORD_IDX = 2 # Assumed: 0=Silence, 1=Unknown, 2=Wakeword

# Threshold to consider it a detection for the summary report
DETECTION_THRESHOLD = 0.85 

# Padding to ensure the end of the file is processed
PADDING_DURATION_SEC = 0.5 

class OnnxStreamingTester:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        self.sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Audio Preprocessing
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            center=False, 
            power=2.0
        )
        
        self.input_name = self.sess.get_inputs()[0].name
        self.state_names = [n.name for n in self.sess.get_inputs() if 'state' in n.name]
        self.state_shapes = [n.shape for n in self.sess.get_inputs() if 'state' in n.name]
        
        # Fix dynamic shapes in ONNX runtime (replace strings with 1)
        for i, shape in enumerate(self.state_shapes):
            new_shape = list(shape)
            if isinstance(new_shape[0], str): new_shape[0] = 1
            self.state_shapes[i] = tuple(new_shape)

    def reset_states(self):
        states = {}
        for name, shape in zip(self.state_names, self.state_shapes):
            states[name] = np.zeros(shape, dtype=np.float32)
        return states

    def preprocess(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
            
        # Append Padding (Silence) to push final frames through the buffer
        pad_samples = int(SAMPLE_RATE * PADDING_DURATION_SEC)
        waveform = torch.nn.functional.pad(waveform, (0, pad_samples))
        
        mel_spec = self.mel_transform(waveform)
        log_mel = torch.log(mel_spec + 1e-6)
        
        return log_mel.numpy()

    def run_file(self, file_path):
        features = self.preprocess(file_path) # [1, 40, T]
        total_frames = features.shape[2]
        
        states = self.reset_states()
        
        max_wakeword_score = 0.0
        final_prediction = 0
        
        for t in range(total_frames):
            # [1, 1, 40, 1]
            frame = features[:, :, t:t+1]
            frame = np.expand_dims(frame, axis=1) 
            
            ort_inputs = {self.input_name: frame}
            ort_inputs.update(states)
            
            outputs = self.sess.run(None, ort_inputs)
            
            logits = outputs[0]
            new_states = outputs[1:]
            
            for i, name in enumerate(self.state_names):
                states[name] = new_states[i]
                
            # Softmax
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            wakeword_score = probs[0][WAKEWORD_IDX]
            prediction = np.argmax(probs[0])
            
            if wakeword_score > max_wakeword_score:
                max_wakeword_score = wakeword_score
                # Record the class predicted at the moment of highest wakeword confidence
                final_prediction = prediction
                
        return max_wakeword_score, final_prediction

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bcresnet_weights.onnx')
    parser.add_argument('--dataset', type=str, default='./dataset/testing')
    parser.add_argument('--output', type=str, default='streaming_results.txt')
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"❌ Testing dataset not found: {args.dataset}")
        return

    tester = OnnxStreamingTester(args.model)
    
    # Recursive search
    wav_files = glob.glob(os.path.join(args.dataset, '**', '*.wav'), recursive=True)
    if not wav_files:
        print("❌ No .wav files found.")
        return
        
    print(f"🔎 Found {len(wav_files)} files.")
    print(f"📝 Saving detailed results to: {args.output}")

    # Stats for console output
    stats = {
        'wakeword': {'total': 0, 'detected': 0},
        'unknown': {'total': 0, 'false_positives': 0},
        'silence': {'total': 0, 'false_positives': 0}
    }

    with open(args.output, 'w') as f:
        f.write("Filepath,Max_Wakeword_Score,Predicted_Class_At_Max\n")
        
        for wav in tqdm(wav_files):
            try:
                score, pred_class = tester.run_file(wav)
                
                # Use relative path for cleaner logs (e.g., "unknown/vacuum.wav")
                rel_path = os.path.relpath(wav, args.dataset)
                f.write(f"{rel_path},{score:.4f},{pred_class}\n")
                f.flush()
                
                # Update Stats
                is_detected = score >= DETECTION_THRESHOLD
                
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
    print("📊 STREAMING TEST SUMMARY")
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
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
