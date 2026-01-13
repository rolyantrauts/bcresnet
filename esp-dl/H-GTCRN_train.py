import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
import torchaudio
import argparse
import os
import sys

# Add H-GTCRN to path to import the specific model components
# Ensure you've cloned https://github.com/Max1Wz/H-GTCRN
from models.h_gtcrn import H_GTCRN 

def audio_loader(path):
    # Load audio and ensure it is dual-channel for H-GTCRN
    speech, sr = torchaudio.load(path)
    if speech.shape[0] == 1:
        speech = speech.repeat(2, 1) # Convert mono to dual-channel if necessary
    return speech

class SI_SDR_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, est, target):
        # Flatten for SI-SDR calculation
        est = est.view(est.size(0), -1)
        target = target.view(target.size(0), -1)
        dot = torch.sum(est * target, dim=-1, keepdim=True)
        norm = torch.sum(target**2, dim=-1, keepdim=True) + 1e-8
        projection = dot * target / norm
        noise = est - projection
        ratio = torch.sum(projection**2, dim=-1) / (torch.sum(noise**2, dim=-1) + 1e-8)
        return -10 * torch.log10(ratio + 1e-8).mean()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='./dataset')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=16)
    return parser.parse_args()

def train():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Model Initialization
    # Parameters adjusted for real-time smart speaker constraints
    model = H_GTCRN(
        # These are example params based on GTCRN architecture
        # Adjust based on the actual model.py signature in the repo
    ).to(device)

    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint: {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))

    # 2. Export-Only Logic
    if args.export_only:
        model.eval()
        dummy_input = torch.randn(1, 2, 16000 * 2) # 2 seconds of dual-channel audio
        torch.onnx.export(model, dummy_input, "h_gtcrn_wakeword.onnx", opset_version=11)
        print("Export Complete.")
        return

    # 3. Data Loading
    # Uses DatasetFolder to automatically pick up subfolders as classes
    train_set = DatasetFolder(
        root=os.path.join(args.dataset_root, 'training'),
        loader=audio_loader,
        extensions=('.wav', '.flac')
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # 4. Training Loop with NaN Safety
    current_lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=current_lr)
    criterion = SI_SDR_Loss()

    model.train()
    for epoch in range(50):
        for i, (audio, labels) in enumerate(train_loader):
            audio = audio.to(device)
            
            optimizer.zero_grad()
            output = model(audio)
            
            # Note: For speech enhancement as a front-end, 'output' is cleaned audio
            # In a real wakeword scenario, you'd calculate loss against a 'clean' reference.
            # Here we assume audio is the target for a self-supervised or enhancement task.
            loss = criterion(output, audio) 

            if torch.isnan(loss):
                print(f"⚠️ NaN detected. Dropping LR to 0.001.")
                current_lr = 0.001
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                optimizer.zero_grad()
                continue

            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
