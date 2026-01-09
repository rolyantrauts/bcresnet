# Live Microphone Inference (Stream Test)

This script (`stream_test.py`) allows you to run your trained BCResNet models (ONNX) on live microphone input in real-time. It features a rolling buffer, confidence smoothing, and cooldown logic to prevent false triggers.

## 📦 Requirements

You need a few extra libraries to handle real-time audio and ONNX inference:

```
pip install sounddevice numpy onnxruntime
```
Linux/Raspberry Pi users: You might need to install PortAudio at the system level
```
sudo apt-get install libportaudio2
```
🚀 Basic Usage1. Run with Standard Model (Float32)By default, the script looks for bcresnet_float32.onnx.Bashpython stream_test.py  
2. Run with Quantized Model (Int8)To run the compressed Int8 model (recommended for Raspberry Pi Zero 2), simply point to the Int8 file.Bashpython stream_test.py --model bcresnet_int8.onnx  
Note: The internal logic remains identical. ONNX Runtime automatically detects the Int8 quantization and applies CPU optimizations.  
🎛️ Configuration Parameters  
You can adjust the behavior of the stream to balance latency vs accuracy.  
Model Settings   
--model bcresnet_float32.onnx Path to your .onnx model file (Float32 or Int8).  
--sample_rate 16000 Audio sample rate. Must match your training configuration.  
--duration 1.5 Input window length in seconds. Must match training.  
--wakeword_idx 2 The class index of your wake word (usually 2 if folders are silence, unknown, wakeword).  
Inference Logic  
--poll_rate 0.1 How often to run the model (in seconds). Lower = more responsive but higher CPU usage.  
--threshold 0.85 Confidence (0.0 - 1.0) required to trigger a detection.  
Robustness (Smoothing & Cooldown)  
--average_window 3 Number of past frames to average. Removes glitches/spikes. Higher = smoother but adds slight lag.  
--cooldown 2.0 Seconds to wait after a detection before listening again. Prevents double-triggering on one word.  
Hardware   
--device None The numerical index of your microphone. Use python -m sounddevice to list available devices.  
⚡ Examples  
High-Performance Setup (Raspberry Pi 4 / PC)Scans audio every 50ms for ultra-fast response.
```
python stream_test.py --model bcresnet_float32.onnx --poll_rate 0.05
```
Low-Power Setup (Raspberry Pi Zero 2)Uses the Int8 model, scans every 200ms, and uses a longer averaging window to ensure stability on the weaker CPU.
```
python stream_test.py --model bcresnet_int8.onnx --poll_rate 0.2 --average_window 5
```
Custom Training SettingsIf you trained a model with specific settings (e.g., 1.0 second duration), you must match them here:
```
python stream_test.py --duration 1.0 --sample_rate 16000
```
❓ Troubleshooting  
"Error opening InputStream"  
Run python -m sounddevice to see your device list.Find the ID of your USB mic and run: python stream_test.py --device <ID>  
"IndexError: list index out of range"Your --wakeword_idx might be wrong.Check your training folders alphabetical order.Example: ['background', 'computer', 'speech'] -> 'computer' is index 1.Run: python stream_test.py --wakeword_idx 1"  
Model input shape mismatch"Ensure --duration and --sample_rate exactly match the arguments you used in main.py during training.
