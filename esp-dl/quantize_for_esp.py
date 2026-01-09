import os
import onnx
import torch
import soundfile as sf
import numpy as np
from ppq import *
from ppq.api import *
from utils import Padding, Preprocess # Reuse your existing utils

# 1. Config
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ONNX_PATH = "bcresnet_float32.onnx"
ESPDL_PATH = "bcresnet.espdl"
CALIB_DIR = "./dataset/validation/wakeword" # Use real data for calibration

# 2. Prepare Calibration Data
# We need ~32 samples to help the quantizer measure dynamic range
def calibration_dataloader():
    BATCH_SIZE = 1
    files = [os.path.join(CALIB_DIR, f) for f in os.listdir(CALIB_DIR) if f.endswith('.wav')][:32]
    
    # Reuse your Preprocessor to ensure data matches exactly
    preprocessor = Preprocess(DEVICE, sample_rate=16000, n_mels=80, specaug=False) # Ensure n_mels matches training
    padder = Padding(target_len=24000) # 1.5s * 16000
    
    calib_data = []
    for f in files:
        audio, sr = sf.read(f, dtype='float32')
        wave = torch.from_numpy(audio).unsqueeze(0) # (1, T)
        if wave.shape[1] > 1: wave = wave.mean(dim=0, keepdim=True) # Mono
        wave = padder(wave).unsqueeze(0).to(DEVICE) # (1, 1, T)
        
        # Preprocess to Spectrogram (BCResNet input)
        # Note: Your ONNX export includes the preprocessor? 
        # IF ONNX HAS PREPROCESSOR INSIDE: Return raw audio
        # IF ONNX IS MODEL ONLY: Return spectrogram
        
        # Based on your main.py, you exported 'EndToEndModel' which TAKES RAW AUDIO.
        # So we pass raw audio to the quantizer.
        calib_data.append(wave.cpu().numpy())
        
    return calib_data

# 3. Run Quantization
print("Quantizing for ESP32-S3...")
quantize_onnx_model(
    onnx_import_file=ONNX_PATH,
    calib_dataloader=calibration_dataloader(),
    calib_steps=32,
    input_shape=[1, 1, 24000], # [Batch, Ch, Time] (1.5s)
    platform=TargetPlatform.ESP32S3, # Change to ESP32 if using non-S3
    output_onnx_file=None, # We don't need the intermediate ONNX
    output_espdl_file=ESPDL_PATH,
    verbose=1
)
print(f"Success! Model saved to {ESPDL_PATH}")