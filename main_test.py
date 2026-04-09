import os
import glob
import argparse
import json
import numpy as np
import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm
import warnings

# Use the official LiteRT runtime package
from ai_edge_litert.interpreter import Interpreter

# Suppress the torchaudio load warning for cleaner output
warnings.filterwarnings("ignore", message="An output with one or more elements was resized")

class Padding(object):
    """Pads or crops the audio to a specific target length in samples."""
    def __init__(self, target_len=22400):
        self.target_len = target_len

    def __call__(self, waveform):
        if waveform.shape[1] >= self.target_len:
            return waveform[:, :self.target_len]
        else:
            padding_len = self.target_len - waveform.shape[1]
            return torch.nn.functional.pad(waveform, (0, padding_len))

class TFLiteNonStreamingTester:
    def __init__(self, model_path, wakeword_idx, sample_rate=16000, clip_duration=1.4, n_mels=80):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        # Initialize the LiteRT Interpreter
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        
        self.wakeword_idx = wakeword_idx
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * clip_duration)
        
        # --- Audio Preprocessing (Log-Mel Spectrogram) ---
        # Must exactly match the Preprocess class in main.py
        self.pad_transform = Padding(target_len=self.target_samples)
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels, 
            n_fft=480,       
            hop_length=160   
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def process_file(self, filepath):
        """Runs non-streaming inference on a single audio file."""
        
        # 1. Load Audio (Using robust soundfile logic)
        audio_np, sr = sf.read(filepath, dtype='float32')
        waveform = torch.from_numpy(audio_np)
        
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # 2. Pad/Crop to exact target length
        waveform = self.pad_transform(waveform)
        
        # 3. Convert to Log-Mel Spectrogram
        mel_spec = self.melspec(waveform)
        log_mel = self.amplitude_to_db(mel_spec)
        
        # 4. Prepare input tensor shape: [Batch, Channel, Mels, Time] -> [1, 1, 80, 141]
        input_data = log_mel.unsqueeze(0).numpy()
        
        # 5. Handle INT8 Quantized Inputs if required by the model
        dtype = self.input_details['dtype']
        if dtype == np.int8:
            scale, zero_point = self.input_details['quantization']
            input_data = (input_data / scale + zero_point).astype(np.int8)
        else:
            input_data = input_data.astype(np.float32)

        # 6. Run LiteRT Inference
        self.interpreter.set_tensor(self.input_details['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details['index'])
        
        # 7. Handle INT8 Quantized Outputs if required
        if self.output_details['dtype'] == np.int8:
            scale, zero_point = self.output_details['quantization']
            output_data = (output_data.astype(np.float32) - zero_point) * scale

        logits = output_data[0]
        
        # Calculate Softmax Probabilities
        exp_logits = np.exp(logits - np.max(logits)) # Subtract max for numerical stability
        probs = exp_logits / np.sum(exp_logits)
        
        wakeword_score = probs[self.wakeword_idx]
        predicted_class = np.argmax(probs)
                
        return wakeword_score, predicted_class

def main():
    parser = argparse.ArgumentParser(description="Test Non-Streaming LiteRT Model")
    parser.add_argument('--model', type=str, default='bcresnet_int8.tflite', help='Path to .tflite model')
    parser.add_argument('--data_root', type=str, default='../gkws/data2', help='Path to dataset root folder')
    
    # Replicating main.py parameters
    parser.add_argument('--clip_duration', type=float, default=1.4)
    parser.add_argument('--n_mels', type=int, default=80)
    
    parser.add_argument('--threshold', type=float, default=0.85, help='Detection threshold to trigger a true positive')
    parser.add_argument('--labels', type=str, default='labels.json', help='Path to the labels.json file generated during training')
    
    args = parser.parse_args()

    # --- Load Class Mapping from Training ---
    if not os.path.exists(args.labels):
        print(f"❌ Error: {args.labels} not found. Ensure you are running this in the same directory where main.py saved it.")
        return
        
    with open(args.labels, 'r') as f:
        class_to_idx = json.load(f)
        
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    classes = list(class_to_idx.keys())
    
    print(f"📄 Loaded class mapping from {args.labels}: {class_to_idx}")

    # --- Deduce Wakeword ---
    static_background_names = ['unknown', 'noise', 'silence', '_background_noise_']
    wakeword_class = None
    
    for c in classes:
        if c.lower() not in static_background_names:
            wakeword_class = c
            break
            
    if not wakeword_class:
        wakeword_class = classes[0] # Fallback
        
    wakeword_idx = class_to_idx[wakeword_class]

    # --- Setup Testing Path ---
    test_dir = os.path.join(args.data_root, 'testing')
    if not os.path.exists(test_dir):
        print(f"⚠️ Warning: 'testing' folder not found. Falling back to 'validation'.")
        test_dir = os.path.join(args.data_root, 'validation')
        if not os.path.exists(test_dir):
            print(f"❌ Error: Neither testing nor validation folders found in {args.data_root}")
            return

    print("\n" + "="*40)
    print("🚀 TFLite Testing Configuration (Non-Streaming)")
    print(f"{'Model':<20} : {args.model}")
    print(f"{'Test Directory':<20} : {test_dir}")
    print(f"{'Threshold':<20} : {args.threshold}")
    print(f"{'Wakeword Class':<20} : '{wakeword_class}' (Index: {wakeword_idx})")
    print("="*40 + "\n")

    tester = TFLiteNonStreamingTester(
        model_path=args.model, 
        wakeword_idx=wakeword_idx,
        clip_duration=args.clip_duration,
        n_mels=args.n_mels
    )
    
    stats = {c: {'total': 0, 'detected': 0, 'false_positives': 0} for c in classes}
    
    print(f"🔍 Scanning dataset at {test_dir}...")
    wav_files = glob.glob(os.path.join(test_dir, '**', '*.wav'), recursive=True)
    
    if not wav_files:
        print("❌ No wav files found!")
        return

    with open("tflite_test_results.txt", "w") as log:
        log.write("Filepath,Wakeword_Score,Predicted_Class\n")
        
        for wav in tqdm(wav_files, desc="Evaluating files"):
            try:
                score, pred_idx = tester.process_file(wav)
                pred_class_name = idx_to_class[pred_idx]
                is_detected = (score > args.threshold)
                
                rel_path = os.path.relpath(wav, test_dir)
                log.write(f"{rel_path},{score:.4f},{pred_class_name}\n")
                
                # Determine ground truth from folder name
                true_class = None
                path_parts = rel_path.replace('\\', '/').split('/')
                for c in classes:
                    if c in path_parts:
                        true_class = c
                        break
                        
                if true_class:
                    stats[true_class]['total'] += 1
                    if is_detected: 
                        if true_class == wakeword_class:
                            stats[true_class]['detected'] += 1
                        else:
                            stats[true_class]['false_positives'] += 1
                        
            except Exception as e:
                print(f"\n⚠️ Error processing {wav}: {e}")

    print("\n" + "="*40)
    print("📊 LITERT (TFLITE) TEST SUMMARY")
    print("="*40)
    
    for c in classes:
        if stats[c]['total'] > 0:
            if c == wakeword_class:
                ww_rate = 100 * stats[c]['detected'] / stats[c]['total']
                print(f"✅ True Positives ({c}): {ww_rate:.2f}% ({stats[c]['detected']}/{stats[c]['total']})")
            else:
                fa_rate = 100 * stats[c]['false_positives'] / stats[c]['total']
                print(f"❌ False Positives ({c}):  {fa_rate:.2f}% ({stats[c]['false_positives']}/{stats[c]['total']})")
                
    print("\n📄 Detailed results saved to: tflite_test_results.txt")

if __name__ == "__main__":
    main()
