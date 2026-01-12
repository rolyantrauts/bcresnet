import os
import argparse
import json
import torch
import torchaudio
import onnxruntime as ort
import soundfile as sf
import numpy as np
from tqdm import tqdm
from utils import Preprocess, Padding

class Tester:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Batch Testing for Streaming ONNX Model")
        parser.add_argument("--data_root", default="./dataset/testing", type=str, help="Path to testing dataset")
        parser.add_argument("--model", default="streaming_bcresnet_float32.onnx", type=str, help="Path to Streaming ONNX model")
        parser.add_argument("--labels", default="labels.json", type=str, help="Path to labels.json")
        parser.add_argument("--output", default="streaming_results.txt", type=str, help="Output text file")
        
        # Audio Config
        parser.add_argument("--sample_rate", default=16000, type=int)
        parser.add_argument("--duration", default=1.0, type=float)
        parser.add_argument("--n_mels", default=40, type=int)
        
        args = parser.parse_args()
        self.__dict__.update(vars(args))
        
        self.target_samples = int(self.duration * self.sample_rate)
        self.device = torch.device('cpu')
        
        # 1. Load Labels
        if not os.path.exists(self.labels):
            print(f"[Error] {self.labels} not found. Run training first.")
            exit(1)
            
        with open(self.labels, 'r') as f:
            self.class_map = json.load(f)
            
        target_class_name = 'wakeword'
        if target_class_name not in self.class_map:
            print(f"[Error] Class '{target_class_name}' not found in labels.json.")
            exit(1)
            
        self.wakeword_idx = self.class_map[target_class_name]
        print(f"Target Class: '{target_class_name}' is Index {self.wakeword_idx}")

        # 2. Init Preprocessing
        self.padder = Padding(target_len=self.target_samples)
        self.preprocessor = Preprocess(self.device, sample_rate=self.sample_rate, n_mels=self.n_mels, specaug=False)
        self.preprocessor.eval()
        
        # 3. Init ONNX
        print(f"Loading ONNX Model: {self.model}...")
        if not os.path.exists(self.model):
            print(f"[Error] Model file {self.model} not found.")
            exit(1)

        # Set graph optimization to ensure speed
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(self.model, sess_options, providers=['CPUExecutionProvider'])
        
        # 4. Analyze Inputs
        self.inputs_info = self.sess.get_inputs()
        self.input_audio_name = self.inputs_info[0].name
        self.state_input_names = [x.name for x in self.inputs_info[1:]]
        
        print(f"Model Inputs: 1 Audio + {len(self.state_input_names)} States")

    def process_file(self, filepath):
        try:
            # 1. Load Audio
            audio, sr = sf.read(filepath, dtype='float32')
            
            # 2. Resample
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                audio_t = torch.from_numpy(audio)
                if audio_t.dim() == 1: audio_t = audio_t.unsqueeze(0)
                audio = resampler(audio_t).squeeze().numpy()

            # 3. Prepare Tensor
            wave = torch.from_numpy(audio)
            if wave.dim() == 1: wave = wave.unsqueeze(0)
            
            if wave.shape[0] > 1:
                wave = wave.mean(dim=0, keepdim=True)
                
            # 4. Pad/Crop & Preprocess
            wave = self.padder(wave).unsqueeze(0)
            with torch.no_grad():
                spec = self.preprocessor(wave) 
            
            # 5. Streaming Inference Loop
            spec_np = spec.numpy() # (1, 1, 40, T_frames)
            num_frames = spec_np.shape[3]
            
            # Reset States (Zeros)
            current_states = {}
            for inp in self.inputs_info[1:]:
                shape = [1 if (d is None or isinstance(d, str) or d < 0) else d for d in inp.shape]
                current_states[inp.name] = np.zeros(shape, dtype=np.float32)

            # Track MAX probability for wakeword over time
            max_wakeword_conf = 0.0

            for t in range(num_frames):
                # Extract single frame
                audio_frame = spec_np[:, :, :, t:t+1]
                
                # Inputs
                ort_inputs = {self.input_audio_name: audio_frame}
                ort_inputs.update(current_states)
                
                # Run Inference
                ort_outs = self.sess.run(None, ort_inputs)
                
                # Output[0] is Probabilities (Softmax was already applied in export)
                probs = ort_outs[0] 
                
                # Handle shapes like (1, NumClasses, 1, 1) or (1, NumClasses)
                probs = np.squeeze(probs) 
                
                # Update Max Score
                current_score = probs[self.wakeword_idx]
                if current_score > max_wakeword_conf:
                    max_wakeword_conf = current_score
                
                # Update States
                new_state_values = ort_outs[1:]
                for name, val in zip(self.state_input_names, new_state_values):
                    current_states[name] = val
            
            return max_wakeword_conf
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return 0.0

    def run(self):
        files = []
        for root, _, filenames in os.walk(self.data_root):
            for filename in filenames:
                if filename.lower().endswith(('.wav', '.flac', '.ogg')):
                    files.append(os.path.join(root, filename))
        
        print(f"Processing {len(files)} files...")
        
        with open(self.output, "w") as f:
            f.write("Filename, Wakeword_Confidence\n")
            for filepath in tqdm(files):
                score = self.process_file(filepath)
                f.write(f"{filepath}, {float(score):.6f}\n")
                
        print(f"\n[Success] Results saved to {self.output}")

if __name__ == "__main__":
    Tester().run()