import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
import sys

# Import the MFCC-capable architecture
from bcresnet_mfcc_streaming import StreamingBCResNet, export_onnx

# --- MFCC Configuration (20 Bins) ---
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_N_MFCC = 20      
DEFAULT_N_MELS = 40      # Internal filterbank used to calculate MFCC
DEFAULT_N_FFT = 480      
DEFAULT_HOP_LENGTH = 480 

class SpeechCommandDataset(Dataset):
    def __init__(self, root_dir, split, clip_duration=1.5, sample_rate=16000):
        self.root_dir = os.path.join(root_dir, split)
        self.classes = ['silence', 'unknown', 'wakeword']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.files = []
        self.clip_duration = clip_duration
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * clip_duration)
        
        if not os.path.exists(self.root_dir):
            if split != 'training': return 
            print(f"❌ Error: Split folder not found: {self.root_dir}")
            sys.exit(1)

        print(f"--- Loading {split} set ---")
        for cls in self.classes:
            cls_path = os.path.join(self.root_dir, cls)
            if not os.path.exists(cls_path): continue
            wavs = glob.glob(os.path.join(cls_path, "*.wav"))
            print(f"   {cls}: {len(wavs)} files")
            for w in wavs:
                self.files.append((w, self.class_to_idx[cls]))
                
        if split == 'training' and len(self.files) == 0:
            print(f"❌ Error: No files found in {self.root_dir}")
            sys.exit(1)

        # --- MFCC Frontend ---
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=DEFAULT_N_MFCC,
            melkwargs={
                'n_fft': DEFAULT_N_FFT,
                'n_mels': DEFAULT_N_MELS,
                'hop_length': DEFAULT_HOP_LENGTH,
                'center': False,
                'power': 2.0
            }
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        waveform, sr = torchaudio.load(path)
        
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        current_samples = waveform.size(1)
        if current_samples > self.target_samples:
            start = torch.randint(0, current_samples - self.target_samples + 1, (1,)).item()
            waveform = waveform[:, start : start + self.target_samples]
        elif current_samples < self.target_samples:
            padding = self.target_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        # Extract MFCC (20 bins)
        mfcc = self.mfcc_transform(waveform)
        
        return mfcc, label

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, scheduler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}", unit="batch")
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        time_steps = inputs.size(3)
        
        optimizer.zero_grad()
        
        states = []
        for (c, h, p) in model.block_specs:
            s = torch.zeros(batch_size, c, h, p, device=device)
            states.append(s)
            
        logits = None
        for t in range(time_steps):
            frame = inputs[:, :, :, t:t+1]
            logits, *states = model(frame, *states)
            
        loss = criterion(logits, targets)
        
        if torch.isnan(loss):
            print("\n⚠️ NaN Loss detected! Skipping update.")
            optimizer.zero_grad()
            continue
            
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{100.*correct/total:.2f}%", lr=f"{current_lr:.5f}")

    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            time_steps = inputs.size(3)
            
            states = []
            for (c, h, p) in model.block_specs:
                states.append(torch.zeros(batch_size, c, h, p, device=device))
            
            logits = None
            for t in range(time_steps):
                frame = inputs[:, :, :, t:t+1]
                logits, *states = model(frame, *states)
                
            loss = criterion(logits, targets)
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return total_loss / len(loader), 100. * correct / total

def main():
    parser = argparse.ArgumentParser(description="Train Streaming BC-ResNet (MFCC 20-Bin)")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.1) 
    parser.add_argument('--dataset', type=str, default='./dataset')
    parser.add_argument('--clip_duration', type=float, default=1.5)
    parser.add_argument('--save_path', type=str, default='bcresnet_mfcc_stream.pth')
    
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=15)
    
    parser.add_argument('--export_only', action='store_true')
    
    args = parser.parse_args()

    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
        
    # --- Initialize Model with 20 Bins ---
    model = StreamingBCResNet(n_classes=3, tau=1.0, input_freq=DEFAULT_N_MFCC).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    time_steps = int(16000 * args.clip_duration / 480)

    print("\n" + "="*40)
    print(f"🚀 Training Configuration")
    print(f"{'Device':<20} : {device}")
    print(f"{'Input Tensor':<20} : (Batch, 1, 20, {time_steps})")
    print(f"{'Batch Size':<20} : {args.batch_size}")
    print(f"{'Optimizer':<20} : SGD (Momentum=0.9, WD=1e-3)")
    print(f"{'Max Epochs':<20} : {args.epochs}")
    print(f"{'Base LR':<20} : {args.lr}")
    print(f"{'Warmup / Patience':<20} : {args.warmup_epochs} / {args.patience}")
    print(f"{'Total Params':<20} : {total_params:,}")
    print("="*40 + "\n")

    if args.export_only:
        if not os.path.exists(args.save_path):
            print(f"❌ Error: Checkpoint not found at {args.save_path}")
            return
        print(f"🔄 Loading weights from {args.save_path}...")
        model.load_state_dict(torch.load(args.save_path, weights_only=True, map_location=device))
        export_onnx(model.to('cpu'), path=args.save_path.replace('.pth', '.onnx'))
        return

    train_set = SpeechCommandDataset(args.dataset, 'training', clip_duration=args.clip_duration)
    val_set = SpeechCommandDataset(args.dataset, 'validation', clip_duration=args.clip_duration)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

    # Scheduler (Steps)
    steps_per_epoch = len(train_loader)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch
    decay_steps = total_steps - warmup_steps

    warmup_scheduler = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_steps)
    decay_scheduler = CosineAnnealingLR(optimizer, T_max=decay_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps])

    best_val_loss = float('inf')
    best_acc = 0.0
    patience_counter = 0

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scheduler)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            print(f"Epoch {epoch}: Train Loss {train_loss:.4f} ({train_acc:.2f}%) | Val Loss {val_loss:.4f} ({val_acc:.2f}%)")
            
            if epoch == args.warmup_epochs + 1:
                print("    (Warmup Complete. Resetting Early Stopping Counter)")
                best_val_loss = val_loss
                patience_counter = 0

            if epoch > args.warmup_epochs:
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), args.save_path)
                    print(f"--> ⭐ New Best Accuracy! Saved to {args.save_path}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0 
                else:
                    patience_counter += 1
                    print(f"    ⏳ Patience: {patience_counter}/{args.patience}")
                    
                if patience_counter >= args.patience:
                    print(f"\n🛑 Early stopping triggered!")
                    break
            else:
                if val_acc > best_acc: best_acc = val_acc
                print(f"    (Warmup Phase)")
                
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    print(f"\n✅ Training Complete. Best Accuracy: {best_acc:.2f}%")
    
    if os.path.exists(args.save_path):
        print(f"\n🔄 Exporting...")
        model_cpu = StreamingBCResNet(n_classes=3, tau=1.0, input_freq=DEFAULT_N_MFCC)
        model_cpu.load_state_dict(torch.load(args.save_path, weights_only=True, map_location='cpu'))
        export_onnx(model_cpu, path=args.save_path.replace('.pth', '.onnx'))

if __name__ == "__main__":
    main()
