import argparse
import numpy as np
import onnxruntime as ort
import sounddevice as sd
import queue
import sys
import time
import torch
from utils import Preprocess, Padding  # We reuse your training utils!

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

class RollingBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = np.zeros(size, dtype=np.float32)
        
    def extend(self, new_data):
        new_len = len(new_data)
        if new_len >= self.size:
            self.buffer[:] = new_data[-self.size:]
        else:
            self.buffer[:-new_len] = self.buffer[new_len:]
            self.buffer[-new_len:] = new_data

    def get(self):
        return self.buffer

def main():
    parser = argparse.ArgumentParser(description="Nuclear Option Inference (External Front-End)")
    parser.add_argument("--model", type=str, default="bcresnet_float32.onnx")
    
    # Must match training exactly!
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=1.0) # e.g. 1.0 or 1.5
    parser.add_argument("--n_mels", type=int, default=40)      # e.g. 40 or 80
    
    parser.add_argument("--wakeword_idx", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--poll_rate", type=float, default=0.1)
    parser.add_argument("--device", type=int, default=None)
    
    args = parser.parse_args()

    # 1. Load ONNX Model
    print(f"Loading Nuclear Model: {args.model}...")
    try:
        sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    input_name = sess.get_inputs()[0].name
    
    # 2. Initialize the Python Front-End (PyTorch)
    # We use the CPU to calculate the spectrogram, simulating what the ESP32 DSP will do.
    print("Initializing Audio Front End (Spectrogram)...")
    device = torch.device('cpu')
    preprocessor = Preprocess(device, sample_rate=args.sample_rate, n_mels=args.n_mels, specaug=False)
    
    # Calculate sizes
    BUFFER_SIZE = int(args.sample_rate * args.duration)
    
    print("\n" + "="*30)
    print(" NUCLEAR INFERENCE STARTED ")
    print("="*30)
    print(f"Sample Rate : {args.sample_rate}")
    print(f"Mel Bins    : {args.n_mels}")
    print(f"Window      : {args.duration}s")
    print("="*30 + "\n")

    # 3. Setup Audio
    q = queue.Queue()
    rolling_buffer = RollingBuffer(BUFFER_SIZE)

    def audio_callback(indata, frames, time, status):
        if status: print(status, file=sys.stderr)
        q.put(indata.copy())

    print("--- Listening... (Press Ctrl+C to stop) ---")
    block_size = int(args.sample_rate * args.poll_rate)

    try:
        with sd.InputStream(device=args.device, channels=1, samplerate=args.sample_rate,
                            callback=audio_callback, blocksize=block_size):
            
            while True:
                new_data = q.get().squeeze()
                rolling_buffer.extend(new_data)
                
                # --- STEP A: PRE-PROCESSING (Python side) ---
                # 1. Get raw audio
                raw_audio_np = rolling_buffer.get().astype(np.float32)
                
                # 2. Convert to Tensor for utils.py
                raw_audio_tensor = torch.from_numpy(raw_audio_np).unsqueeze(0).to(device)
                
                # 3. Generate Spectrogram
                # Output shape will be [1, 1, n_mels, time_steps]
                with torch.no_grad():
                    spectrogram_tensor = preprocessor(raw_audio_tensor)
                    
                # 4. Convert back to Numpy for ONNX
                spectrogram_np = spectrogram_tensor.numpy()

                # --- STEP B: INFERENCE (ONNX side) ---
                start_t = time.time()
                outputs = sess.run(None, {input_name: spectrogram_np})
                inference_ms = (time.time() - start_t) * 1000
                
                # --- STEP C: VISUALIZATION ---
                logits = outputs[0]
                probs = softmax(logits)[0]
                conf = probs[args.wakeword_idx] if len(probs) > args.wakeword_idx else 0.0
                
                bar_len = int(conf * 20)
                bar = "█" * bar_len + "-" * (20 - bar_len)
                status = " <<< DETECTED! >>>" if conf > args.threshold else ""
                
                print(f"\r[{bar}] {conf:.2f} | Inf: {inference_ms:.1f}ms {status}", end="", flush=True)

    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()