import argparse
import numpy as np
import onnxruntime as ort
import sounddevice as sd
import queue
import sys
import time
from collections import deque

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
    parser = argparse.ArgumentParser(description="Live Microphone Inference with ONNX")
    
    # Model Config
    parser.add_argument("--model", type=str, default="bcresnet_float32.onnx")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=1.5)
    
    # Inference Config
    parser.add_argument("--wakeword_idx", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--poll_rate", type=float, default=0.1)
    
    # --- NEW: Robustness Args ---
    parser.add_argument("--average_window", type=int, default=3, help="Smooth confidence over last N frames")
    parser.add_argument("--cooldown", type=float, default=2.0, help="Seconds to wait after detection")
    
    # Hardware
    parser.add_argument("--device", type=int, default=None)
    
    args = parser.parse_args()

    # 1. Load Model
    print(f"Loading model: {args.model}...")
    try:
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        sess = ort.InferenceSession(args.model, sess_options, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    input_name = sess.get_inputs()[0].name
    BUFFER_SIZE = int(args.sample_rate * args.duration)
    
    # Cooldown Logic
    cooldown_frames = int(args.cooldown / args.poll_rate)
    cooldown_counter = 0
    
    # Smoothing Logic (Deque holds last N confidence scores)
    confidence_history = deque(maxlen=args.average_window)

    print("\n" + "="*35)
    print(" LIVE STREAM CONFIGURATION ")
    print("="*35)
    print(f"Window       : {args.duration}s")
    print(f"Poll Rate    : {args.poll_rate}s")
    print(f"Smoothing    : Avg last {args.average_window} frames")
    print(f"Cooldown     : {args.cooldown}s after trigger")
    print(f"Threshold    : {args.threshold}")
    print("="*35 + "\n")

    # 2. Setup Audio
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
                
                # --- INFERENCE ---
                input_tensor = rolling_buffer.get().astype(np.float32)
                input_tensor = np.expand_dims(input_tensor, axis=0)
                
                start_t = time.time()
                outputs = sess.run(None, {input_name: input_tensor})
                inference_ms = (time.time() - start_t) * 1000
                
                # Get raw confidence
                logits = outputs[0]
                probs = softmax(logits)[0]
                raw_conf = probs[args.wakeword_idx] if len(probs) > args.wakeword_idx else 0.0
                
                # --- SMOOTHING ---
                confidence_history.append(raw_conf)
                avg_conf = sum(confidence_history) / len(confidence_history)
                
                # --- LOGIC ---
                status_msg = ""
                
                # Decrement Cooldown
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                    status_msg = f" [COOLDOWN {cooldown_counter}]"
                else:
                    # Check Trigger
                    if avg_conf > args.threshold:
                        status_msg = " <<< DETECTED! >>>"
                        cooldown_counter = cooldown_frames # Start Cooldown
                        
                        # --- OPTIONAL: TRIGGER ACTION HERE ---
                        # e.g., os.system("aplay beep.wav")
                
                # Visuals
                bar_len = int(avg_conf * 20)
                bar = "█" * bar_len + "-" * (20 - bar_len)
                
                print(f"\r[{bar}] {avg_conf:.2f} | Inf: {inference_ms:.1f}ms {status_msg}", end="", flush=True)

    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()