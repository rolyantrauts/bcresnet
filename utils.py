import os
import glob
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
import numpy as np
import warnings
import soundfile as sf

class CustomAudioDataset(Dataset):
    def __init__(self, root_dir, subset=None, transform=None, sample_rate=16000):
        self.sample_rate = sample_rate
        self.transform = transform
        self.data_path = os.path.join(root_dir, subset) if subset else root_dir
        
        self.classes = sorted([d for d in os.listdir(self.data_path) 
                               if os.path.isdir(os.path.join(self.data_path, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.files = []
        for cls_name in self.classes:
            cls_folder = os.path.join(self.data_path, cls_name)
            wav_files = glob.glob(os.path.join(cls_folder, "*.wav"))
            for wav in wav_files:
                self.files.append((wav, self.class_to_idx[cls_name]))

        print(f"Found {len(self.files)} files for {subset if subset else 'all'} in {len(self.classes)} classes.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath, label = self.files[idx]
        
        # Use SoundFile for robust loading
        audio_np, sr = sf.read(filepath, dtype='float32')
        waveform = torch.from_numpy(audio_np)
        
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()
            
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        if sr != self.sample_rate:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = transform(waveform)
            
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label

class Padding(object):
    """Pads or crops the audio to a specific target length in samples."""
    def __init__(self, target_len=16000):
        self.target_len = target_len

    def __call__(self, waveform):
        if waveform.shape[1] >= self.target_len:
            return waveform[:, :self.target_len]
        else:
            padding_len = self.target_len - waveform.shape[1]
            return torch.nn.functional.pad(waveform, (0, padding_len))

class Preprocess(nn.Module):
    """Computes Log Mel Spectrogram and applies SpecAugment."""
    def __init__(self, device, sample_rate=16000, n_mels=40, specaug=False):
        super().__init__()
        self.device = device
        self.specaug = specaug
        
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels, 
            n_fft=480,       
            hop_length=160   
        ).to(device)
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(device)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=35).to(device)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=10).to(device)

    def forward(self, x, labels=None, augment=False):
        x = x.to(self.device)
        x = self.melspec(x)
        x = self.amplitude_to_db(x)

        if self.specaug and augment:
            x = self.freq_masking(x)
            x = self.time_masking(x)
            
        return x