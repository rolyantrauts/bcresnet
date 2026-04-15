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
    def __init__(self, model_path, wakeword_idx, sample_rate=16000, clip_duration=1.4, n_mels=80, 
                 use_sigmoid=False, use_arcface=False, golden_weights=None):
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
        
        self.use_sigmoid = use_sigmoid
        self.use_arcface = use_arcface
        self.golden_weights = golden_weights
        
        # --- Audio Preprocessing (Log-Mel Spectrogram) ---
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
        
        # 1. Load Audio
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
            
        # 2. Pad/Crop
        waveform = self.pad_transform(waveform)
        
        # 3. Convert to Log-Mel Spectrogram
        mel_spec = self.melspec(waveform)
        log_mel = self.amplitude_to_db(mel_spec)
        
        # 4. Prepare input tensor
        input_data = log_mel.unsqueeze(0).numpy()
        
        # 5. Handle INT8 Quantized Inputs
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
        
        # 7. Handle INT8 Quantized Outputs (Dequantize back to Float32)
        if self.output_details['dtype'] == np.int8:
            scale, zero_point = self.output_details['quantization']
            output_data = (output_data.astype(np.float32) - zero_point) * scale

        logits_or_embedding = output_data[0]
        
        # --- ACTIVATION ROUTING ---
        if self.use_arcface:
            # 1. L2 Normalize the extracted audio fingerprint
            norm = np.linalg.norm(logits_or_embedding)
            if norm > 0:
                normalized_embedding = logits_or_embedding / norm
            else:
                normalized_embedding = logits_or_embedding
                
            # 2. Calculate Cosine Similarity against all Golden Weights (-1.0 to 1.0)
            scores = np.dot(self.golden_weights, normalized_embedding)
            
        elif self.use_sigmoid:
            # Sigmoid: Independent probabilities (0.0 to 1.0)
            scores = 1.0 / (1.0 + np.exp(-logits_or_embedding))
            
        else:
            # Softmax: Zero-sum probabilities (0.0 to 1.0)
            exp_logits = np.exp(logits_or_embedding - np.max(logits_or_embedding))
            scores = exp_logits / np.sum(exp_logits)
        
        wakeword_score = scores[self.wakeword_idx]
        predicted_class = np.argmax(scores)
                
        return wakeword_score, predicted_class, scores

def main():
    parser = argparse.ArgumentParser(description="Test Non-Streaming LiteRT Model")
    parser.add_argument('--model', type=str, default='bcresnet_int8.tflite', help='Path to .tflite model')
    parser.add_argument('--data_root', type=str, default='../gkws/data2', help='Path to dataset root folder')
    
    parser.add_argument('--clip_duration', type=float, default=1.4)
    parser.add_argument('--n_mels', type=int, default=80)
    
    parser.add_argument('--threshold', type=float, default=0.85, help='Detection threshold to trigger a true positive')
    parser.add_argument('--labels', type=str, default='labels.json', help='Path to labels.json')
    
    # --- ACTIVATION MODES ---
    parser.add_argument('--use_sigmoid', action='store_true', help='Use Sigmoid instead of Softmax')
    parser.add_argument('--arcface', action='store_true', help='Use Cosine Similarity with Golden Weights')
    parser.add_argument('--arcface_weights', type=str, default='arcface_golden_weights.json', help='Path to exported Golden Weights')
    
    args = parser.parse_args()

    # --- Load Class Mapping ---
    if not os.path.exists(args.labels):
        print(f"❌ Error: {args.labels} not found.")
        return
        
    with open(args.labels, 'r') as f:
        class_to_idx = json.load(f)
        
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    classes = list(class_to_idx.keys())

    # --- Load Golden Weights (If ArcFace) ---
    golden_weights_np = None
    if args.arcface:
        if not os.path.exists(args.arcface_weights):
            print(f"❌ Error: ArcFace mode enabled, but {args.arcface_weights} not found!")
            return
        with open(args.arcface_weights, 'r') as f:
            golden_weights_list = json.load(f)
            golden_weights_np = np.array(golden_weights_list, dtype=np.float32)
            print(f"📄 Loaded Golden Fingerprints (Shape: {golden_weights_np.shape})")

    # --- Deduce Wakeword & Unknown Class ---
    static_background_names = ['unknown', 'noise', 'silence', '_background_noise_']
    wakeword_class = None
    unknown_class = None
    
    for c in classes:
        if c.lower() not in static_background_names:
            wakeword_class = c
        elif c.lower() in ['unknown', '_unknown_']:
            unknown_class = c
            
    if not wakeword_class:
        wakeword_class = classes[0]
        
    wakeword_idx = class_to_idx[wakeword_class]
    unknown_idx = class_to_idx[unknown_class] if unknown_class else None

    # --- Setup Testing Path ---
    test_dir = os.path.join(args.data_root, 'testing')
    if not os.path.exists(test_dir):
        print(f"⚠️ Warning: 'testing' folder not found. Falling back to 'validation'.")
        test_dir = os.path.join(args.data_root, 'validation')
        if not os.path.exists(test_dir):
            print(f"❌ Error: Neither testing nor validation folders found in {args.data_root}")
            return

    # Determine activation string for logging
    if args.arcface:
        activation_str = "ArcFace (Cosine Similarity)"
    elif args.use_sigmoid:
        activation_str = "Sigmoid (Independent Probabilities)"
    else:
        activation_str = "Softmax (Zero-Sum Probabilities)"

    print("\n" + "="*40)
    print("🚀 TFLite Testing Configuration (Non-Streaming)")
    print(f"{'Model':<20} : {args.model}")
    print(f"{'Test Directory':<20} : {test_dir}")
    print(f"{'Threshold':<20} : {args.threshold}")
    print(f"{'Activation':<20} : {activation_str}")
    print(f"{'Wakeword Class':<20} : '{wakeword_class}' (Index: {wakeword_idx})")
    if unknown_class:
        print(f"{'Unknown Class':<20} : '{unknown_class}' (Index: {unknown_idx})")
    print("="*40 + "\n")

    tester = TFLiteNonStreamingTester(
        model_path=args.model, 
        wakeword_idx=wakeword_idx,
        clip_duration=args.clip_duration,
        n_mels=args.n_mels,
        use_sigmoid=args.use_sigmoid,
        use_arcface=args.arcface,
        golden_weights=golden_weights_np
    )
    
    stats = {c: {'total': 0, 'detected': 0, 'false_positives': 0} for c in classes}
    
    print(f"🔍 Scanning dataset at {test_dir}...")
    wav_files = glob.glob(os.path.join(test_dir, '**', '*.wav'), recursive=True)
    
    if not wav_files:
        print("❌ No wav files found!")
        return

    with open("tflite_test_results.txt", "w") as log:
        # Dynamic header based on the detected classes
        score_label = "Cosine_Similarity" if args.arcface else "Probability"
        header = f"Filepath,Predicted_Class,Wakeword_{score_label}," + ",".join([f"Score_{c}" for c in classes]) + "\n"
        log.write(header)
        
        for wav in tqdm(wav_files, desc="Evaluating files"):
            try:
                score, pred_idx, all_scores = tester.process_file(wav)
                pred_class_name = idx_to_class[pred_idx]
                
                # --- DETECTION LOGIC ---
                is_detected = (score > args.threshold)
                
                # Dynamic noise gate / Relative Margin check
                if (args.use_sigmoid or args.arcface) and unknown_idx is not None:
                    # The wakeword score must strictly beat the "unknown" centroid gravity
                    is_detected = is_detected and (score > all_scores[unknown_idx])
                
                rel_path = os.path.relpath(wav, test_dir)
                
                # Format scores to 4 decimal places
                scores_str = ",".join([f"{p:.4f}" for p in all_scores])
                
                log.write(f"{rel_path},{pred_class_name},{score:.4f},{scores_str}\n")
                
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