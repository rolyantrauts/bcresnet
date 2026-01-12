import argparse
import numpy as np
import onnxruntime as ort
import sounddevice as sd
import queue
import sys
import time
import torch
import torchaudio
import json
import os

# Import your existing utils
try:
    from utils import Preprocess
except ImportError:
    print("Error: utils.py not found. Please ensure it is in the same directory.")
    sys.exit(1)

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

class LiveStreamingTester:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cpu')
        
        # 1. Load Labels
        if not os.path.exists(args.labels):
            print(f"Error: {args.labels} not found.")
            sys.exit(1)
        with open(args.labels, 'r') as f:
            self.class_map = json.load(f)
            
        # Find Wakeword Index
        if 'wakeword' in self.class_map:
            self.wakeword_idx = self.class_map['wakeword']
            print(f"Target Class: 'wakeword' (Index {self.wakeword_idx})")
        else:
            print("Warning: 'wakeword' class not found. Using index 2 as default.")
            self.wakeword_idx = 2

        # 2. Load ONNX Model
        print(f"Loading Streaming ONNX Model: {args.model}")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1 # Optimize for latency on weak CPUs
        self.sess = ort.InferenceSession(args.model, sess_options, providers=['CPUExecutionProvider'])
        
        # Analyze Inputs
        self.inputs_info = self.sess.get_inputs()
        self.input_audio_name = self.inputs_info[0].name
        self.state_input_names = [x.name for x in self.inputs_info[1:]]
        
        # Initialize States to Zeros
        self.current_states = {}
        for inp in self.inputs_info[1:]:
            shape = [1 if (d is None or isinstance(d, str) or d < 0) else d for d in inp.shape]
            self.current_states[inp.name] = np.zeros(shape, dtype=np.float32)
            
        # 3. Preprocessor
        self.preprocessor = Preprocess(self.device, sample_rate=args.sample_rate, n_mels=args.n_mels, specaug=False)
        self.preprocessor.eval()

        # 4. Audio Buffer Management
        # --- CHANGED: 20ms Chunks ---
        self.chunk_size = int(args.sample_rate * 0.02)   # 20ms (320 samples)
        self.context_size = int(args.sample_rate * 0.02) # 20ms overlap (buffer previous chunk)
        
        # Initialize buffer with zeros
        self.audio_buffer = np.zeros(self.context_size, dtype=np.float32)
        
        self.q = queue.Queue()
        
        # Smoothing & Cooldown
        self.running_prob = 0.0
        self.alpha = args.smoothing
        self.cooldown_frames = 0
        # Cooldown limit: 1.5s / 0.02s chunk size = 75 chunks
        self.cooldown_limit = int(1.5 / 0.02) 

    def audio_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        self.q.put(indata.copy())

    def process_chunk(self, new_audio):
        # 1. Combine overlap + new audio
        # We combine 20ms Past + 20ms New = 40ms Total
        combined_audio = np.concatenate((self.audio_buffer, new_audio))
        
        # 2. Update overlap buffer for NEXT time
        # Save the *new* audio to be the *past* audio for the next step
        self.audio_buffer = new_audio[-self.context_size:]
        
        # 3. Preprocess -> Mel Spectrogram
        wave_tensor = torch.from_numpy(combined_audio).float().unsqueeze(0) # (1, samples)
        
        with torch.no_grad():
            spec = self.preprocessor(wave_tensor) # (1, 1, 40, T_total)
            
        # 4. Extract valid frames for this chunk
        spec_np = spec.numpy()
        
        # Calc: 320 samples / 160 hop = 2 new frames
        num_new_frames = self.chunk_size // 160
        input_frames = spec_np[:, :, :, -num_new_frames:] 

        # 5. Run Inference
        # We loop over the 2 frames because your ONNX expects [..., 1]
        # (Unless you changed export to 2). This loop is fast enough for 2 frames.
        
        max_prob_in_chunk = 0.0
        
        # Check if we exported with width 1 or 2. 
        # Safest way: Loop 1 frame at a time.
        for i in range(input_frames.shape[3]):
            frame = input_frames[:, :, :, i:i+1] # Take 1 frame: [1,1,40,1]
            
            ort_inputs = {self.input_audio_name: frame}
            ort_inputs.update(self.current_states)
            
            start_t = time.time()
            ort_outs = self.sess.run(None, ort_inputs)
            inference_time = (time.time() - start_t) * 1000
            
            logits = ort_outs[0]
            new_state_values = ort_outs[1:]
            
            # Update States
            for name, val in zip(self.state_input_names, new_state_values):
                self.current_states[name] = val
                
            # Get Prob
            squeezed = np.squeeze(logits)
            probs = softmax(squeezed.reshape(1, -1))[0]
            prob = probs[self.wakeword_idx]
            
            if prob > max_prob_in_chunk:
                max_prob_in_chunk = prob

        return max_prob_in_chunk, inference_time

    def run(self):
        print("\nStarting Microphone Stream (20ms Latency Mode)...")
        print(f"Chunk Size: {self.chunk_size} samples ({self.chunk_size/self.args.sample_rate:.3f}s)")
        
        try:
            with sd.InputStream(samplerate=self.args.sample_rate, 
                                blocksize=self.chunk_size, 
                                channels=1, 
                                callback=self.audio_callback):
                print("Listening... (Press Ctrl+C to stop)")
                print("-" * 60)
                
                while True:
                    raw_audio = self.q.get()
                    raw_audio = raw_audio.flatten().astype(np.float32)
                    
                    raw_prob, inf_ms = self.process_chunk(raw_audio)
                    
                    self.running_prob = (self.alpha * self.running_prob) + ((1.0 - self.alpha) * raw_prob)
                    
                    status = ""
                    if self.cooldown_frames > 0:
                        self.cooldown_frames -= 1
                        status = " [COOLDOWN]"
                    elif self.running_prob > self.args.threshold:
                        status = " <<< DETECTED! >>>"
                        self.cooldown_frames = self.cooldown_limit 
                        
                    bar_len = int(self.running_prob * 30)
                    bar = "█" * bar_len + "-" * (30 - bar_len)
                    
                    # Print slightly less frequently to avoid terminal flicker? 
                    # Or just print every chunk. 20ms is fast, terminal might blur.
                    print(f"\r[{bar}] {self.running_prob:.3f} | {inf_ms:.1f}ms{status}", end="", flush=True)

        except KeyboardInterrupt:
            print("\n\nStopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Streaming Wake Word Testing")
    parser.add_argument("--model", type=str, default="streaming_bcresnet_float32.onnx")
    parser.add_argument("--labels", type=str, default="labels.json")
    
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--smoothing", type=float, default=0.1, help="Lower smoothing for faster reaction (0.1)")
    
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=40)
    
    args = parser.parse_args()
    
    tester = LiveStreamingTester(args)
    tester.run()