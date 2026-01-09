import os
import argparse
import json
import torch
import onnxruntime as ort
import soundfile as sf
import numpy as np
from tqdm import tqdm
from utils import Preprocess, Padding

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

class Tester:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Batch Testing for Nuclear Option Model")
        parser.add_argument("--data_root", default="./dataset/testing", type=str, help="Path to testing dataset (e.g. ./dataset/testing)")
        parser.add_argument("--model", default="bcresnet_float32.onnx", type=str, help="Path to ONNX model")
        parser.add_argument("--labels", default="labels.json", type=str, help="Path to labels.json")
        parser.add_argument("--output", default="test_results.txt", type=str, help="Output text file")
        
        # Must match training config exactly!
        parser.add_argument("--sample_rate", default=16000, type=int)
        parser.add_argument("--duration", default=1.0, type=float)
        parser.add_argument("--n_mels", default=40, type=int)
        
        args = parser.parse_args()
        self.__dict__.update(vars(args))
        
        self.target_samples = int(self.duration * self.sample_rate)
        self.device = torch.device('cpu')
        
        # Load Labels
        if not os.path.exists(self.labels):
            print(f"[Error] {self.labels} not found. Run training first to generate it.")
            exit(1)
            
        with open(self.labels, 'r') as f:
            self.class_map = json.load(f) # {'silence': 0, 'wakeword': 2...}
            
        # Find the index specifically for the 'wakeword' class
        # We assume your folder name for the wake word is 'wakeword'. 
        # If you named it 'hey_computer', change this string!
        target_class_name = 'wakeword' 
        
        if target_class_name not in self.class_map:
            print(f"[Error] Class '{target_class_name}' not found in labels.json.")
            print(f"Available classes: {list(self.class_map.keys())}")
            exit(1)
            
        self.wakeword_idx = self.class_map[target_class_name]
        print(f"Target Class: '{target_class_name}' is Index {self.wakeword_idx}")

        # Initialize Components
        self.padder = Padding(target_len=self.target_samples)
        self.preprocessor = Preprocess(self.device, sample_rate=self.sample_rate, n_mels=self.n_mels, specaug=False)
        
        print(f"Loading ONNX Model: {self.model}...")
        self.sess = ort.InferenceSession(self.model, providers=['CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name

    def process_file(self, filepath):
        try:
            # 1. Load Audio
            audio, sr = sf.read(filepath, dtype='float32')
            
            # 2. Resample if necessary (Simple check)
            if sr != self.sample_rate:
                # Note: For production, use a proper resampler. 
                # For this script, we assume dataset matches config as per main2.py requirements.
                pass 

            # 3. Prepare Tensor
            wave = torch.from_numpy(audio).unsqueeze(0) # (1, T)
            
            # Ensure Mono
            if wave.dim() > 2 and wave.shape[0] > 1:
                wave = wave.mean(dim=0, keepdim=True)
                
            # 4. Pad/Crop
            wave = self.padder(wave).unsqueeze(0) # (1, 1, T)
            
            # 5. Generate Spectrogram (Nuclear Option)
            with torch.no_grad():
                spec = self.preprocessor(wave) # (1, 1, n_mels, time)
                
            # 6. Run Inference
            spec_np = spec.numpy()
            outputs = self.sess.run(None, {self.input_name: spec_np})
            
            # 7. Get Confidence
            probs = softmax(outputs[0])[0]
            wakeword_conf = probs[self.wakeword_idx]
            
            return wakeword_conf
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return 0.0

    def run(self):
        results = []
        
        # Collect all files
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
                
                # Write to file immediately
                # Format: path/to/file.wav, 0.9923
                f.write(f"{filepath}, {score:.6f}\n")
                
        print(f"\n[Success] Results saved to {self.output}")

if __name__ == "__main__":
    Tester().run()