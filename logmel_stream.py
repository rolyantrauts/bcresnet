import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from tqdm import tqdm
import json
import gc
import traceback
import warnings

# --- MUTE WARNINGS AND PREVENT MACOS DEADLOCKS ---
warnings.filterwarnings("ignore", message="An output with one or more elements was resized")
torch.multiprocessing.set_sharing_strategy('file_system')

from utils import CustomAudioDataset, Padding, Preprocess
from bcresnet_logmel_streaming import StreamingBCResNets

def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def export_streaming_models(model, n_mels, save_path):
    print("\n" + "="*40)
    print("   EXPORTING STREAMING MODELS (LiteRT)   ")
    print("="*40)
    
    cpu_device = torch.device("cpu")
    model.to(cpu_device)
    model.eval()
    
    print(f"Input Frame Shape: [1, 1, {n_mels}, 1]")
    print(f"State Buffers: {len(model.block_specs)}")
    
    dummy_frame = torch.randn(1, 1, n_mels, 1, requires_grad=False).to(cpu_device)
    dummy_states = [torch.zeros(1, c, h, p, requires_grad=False).to(cpu_device) for (c, h, p) in model.block_specs]
    
    sample_inputs = (dummy_frame, *dummy_states)
    
    # ---------------------------------------------------------
    # EXPORT 1: LiteRT / TFLite (Float32)
    # ---------------------------------------------------------
    tflite_f32_path = "bcresnet_stream_float32.tflite"
    try:
        import litert_torch
        print("\nConverting PyTorch streaming model to LiteRT (Float32)...")
        edge_model_f32 = litert_torch.convert(model, sample_inputs)
        edge_model_f32.export(tflite_f32_path)
        print(f"[Success] Saved Float32 LiteRT: {tflite_f32_path}")
    except ImportError:
        print("\n[Warning] 'litert_torch' not installed. Skipping Float32 export.")
    except Exception as e:
        print(f"\n[Error] LiteRT Float32 Export Failed: {e}")
        return 

    # ---------------------------------------------------------
    # EXPORT 2: LiteRT / TFLite (INT8 Quantization)
    # ---------------------------------------------------------
    tflite_int8_path = "bcresnet_stream_int8.tflite"
    try:
        from ai_edge_quantizer import quantizer as aeq_quantizer
        from ai_edge_quantizer import recipe
        
        print("\nConverting Float32 LiteRT model to INT8 via ai_edge_quantizer...")
        qt = aeq_quantizer.Quantizer(tflite_f32_path)
        qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
        qt.quantize().export_model(tflite_int8_path)
        print(f"[Success] Saved INT8 LiteRT: {tflite_int8_path}")
        
    except ImportError:
        print("\n[Warning] 'ai-edge-quantizer' not installed. Run 'pip install ai-edge-quantizer'.")
    except Exception as e:
        print("\n" + "!"*40)
        print("[FATAL ERROR] LiteRT INT8 Export Failed! Full Traceback:")
        print("!"*40)
        traceback.print_exc()
        print("!"*40 + "\n")

    print("="*40 + "\n")

class Trainer:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Train Streaming BC-ResNet (Log-Mel)")
        parser.add_argument("--tau", default=1.0, type=float)
        parser.add_argument("--device", default="auto", type=str)
        parser.add_argument("--batch_size", default=64, type=int)
        parser.add_argument("--data_root", default="./dataset", type=str)
        parser.add_argument("--clip_duration", default=1.4, type=float)
        parser.add_argument("--sample_rate", default=16000, type=int)
        
        parser.add_argument("--n_mels", default=80, type=int)
        parser.add_argument("--no_ssn", action="store_true")
        parser.add_argument("--spec_prob", default=0.8, type=float)
        
        parser.add_argument("--epochs", default=100, type=int)
        parser.add_argument("--warmup_epochs", default=5, type=int)
        parser.add_argument("--lr", default=0.005, type=float)
        parser.add_argument("--patience", default=10, type=int)
        parser.add_argument("--dropout", default=0.3, type=float)
        parser.add_argument("--label_smoothing", default=0.0, type=float, help="Label smoothing epsilon")
        
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--start_epoch", default=1, type=int)
        parser.add_argument("--index", default="", type=str)
        parser.add_argument("--export", action="store_true")

        args = parser.parse_args()
        self.__dict__.update(vars(args))
        
        self.use_ssn = not self.no_ssn
        self.target_samples = int(self.clip_duration * self.sample_rate)
        # Hop length is fixed at 160 in utils.py Preprocess
        self.time_steps = int(1 + (self.target_samples / 160)) 
        
        if self.device == "auto":
            self.device_name = get_default_device()
        else:
            self.device_name = self.device
            
        print("\n" + "="*40)
        print("   TRAINING CONFIGURATION (Streaming)   ")
        print("="*40)
        print(f"Device          : {self.device_name}")
        print(f"Clip Duration   : {self.clip_duration}s ({self.target_samples} samples)")
        print(f"Mel Bins        : {self.n_mels}")
        print(f"Spec Shape      : [Batch, 1, {self.n_mels}, {self.time_steps}] (Sliced over time)")
        print(f"Model Tau       : {self.tau} (SSN={self.use_ssn})")
        print(f"Dropout         : {self.dropout}")
        print(f"Label Smoothing : {self.label_smoothing}")
        print(f"SpecAug Prob    : {self.spec_prob * 100:.1f}%")
        print(f"Class Index     : {self.index if self.index else 'Auto-detect'}")
        
        if self.export:
            print(f"Mode            : EXPORT ONLY")
        else:
            print(f"Start Epoch     : {self.start_epoch} / {self.epochs}")
            if self.resume:
                print(f"Resuming        : YES (Attempting to load checkpoint)")
        print("="*40 + "\n")
        
        if self.device_name == "cuda" and torch.cuda.is_available():
            self.device_name = "cuda:0"
        try:
            self.device = torch.device(self.device_name)
        except RuntimeError:
            self.device = torch.device("cpu")
            
        print(f"Running on device: {self.device}")
        
        self._load_data()
        self._load_model()

    def _load_data(self):
        print(f"Loading data from {self.data_root}...")
        transform = transforms.Compose([Padding(target_len=self.target_samples)])
        
        self.train_dataset = CustomAudioDataset(self.data_root, subset="training", transform=transform, sample_rate=self.sample_rate, index_str=self.index)
        self.valid_dataset = CustomAudioDataset(self.data_root, subset="validation", transform=transform, sample_rate=self.sample_rate, index_str=self.index)
        
        if os.path.exists(os.path.join(self.data_root, "testing")):
            self.test_dataset = CustomAudioDataset(self.data_root, subset="testing", transform=transform, sample_rate=self.sample_rate, index_str=self.index)
        else:
            self.test_dataset = self.valid_dataset

        self.num_classes = len(self.train_dataset.classes)
        print(f"Detected Classes: {self.train_dataset.classes} (Total: {self.num_classes})")
        
        try:
            with open("index.txt", "w") as f:
                for cls in self.train_dataset.classes:
                    f.write(f"{cls}\n")
            if not self.export:
                print(f"📄 Class mapping saved to index.txt")
        except Exception as e:
            print(f"⚠️ Warning: Could not save index.txt - {e}")

        # macOS Deadlock Fix
        if self.export:
            workers = 0
            use_persistent = False
        else:
            workers = 2 if os.name != 'nt' else 0
            use_persistent = (workers > 0)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=workers, persistent_workers=use_persistent)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=workers, persistent_workers=use_persistent)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=workers, persistent_workers=use_persistent)

        self.preprocess_train = Preprocess(self.device, sample_rate=self.sample_rate, n_mels=self.n_mels, specaug=True)
        self.preprocess_test = Preprocess(self.device, sample_rate=self.sample_rate, n_mels=self.n_mels, specaug=False)

    def _load_model(self):
        print(f"Building Streaming BCResNet-%.1f..." % self.tau)
        self.model = StreamingBCResNets(tau=self.tau, num_classes=self.num_classes, use_ssn=self.use_ssn, dropout=self.dropout, n_mels=self.n_mels).to(self.device)
        
        self.save_path = "best_streaming_bcresnet.pth"
        
        if (self.resume or self.export) and os.path.exists(self.save_path):
            print(f"🔄 Loading model weights from {self.save_path}...")
            checkpoint = torch.load(self.save_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        elif self.resume or self.export:
            print(f"⚠️ Warning: '{self.save_path}' not found. Using randomly initialized weights.")
            
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total Parameters: {total_params:,}")

    def save_labels(self):
        with open("labels.json", "w") as f:
            json.dump(self.train_dataset.class_to_idx, f, indent=4)

    def train_epoch(self, optimizer, warmup_scheduler, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")
        
        for inputs, labels in loop:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # 1. Full-clip Log-Mel extraction (Shared with main.py)
            should_augment = (np.random.rand() < self.spec_prob)
            inputs = self.preprocess_train(inputs, augment=should_augment)
            
            batch_size = inputs.size(0)
            time_steps = inputs.size(3)
            
            optimizer.zero_grad()
            
            # 2. Initialize Streaming Buffers
            states = [torch.zeros(batch_size, c, h, p, device=self.device) for (c, h, p) in self.model.block_specs]
            
            # 3. Stream through Time
            logits = None
            for t in range(time_steps):
                frame = inputs[:, :, :, t:t+1]
                logits, *states = self.model(frame, *states)
                
            loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
            
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue
                
            loss.backward()
            optimizer.step()
            
            if warmup_scheduler is not None:
                warmup_scheduler.step()
                
            total_loss += loss.item()
            predicted = torch.argmax(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            current_lr = optimizer.param_groups[0]['lr']
            loop.set_postfix(loss=loss.item(), lr=current_lr)
            
        return total_loss / len(self.train_loader), 100 * correct / total

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = self.preprocess_test(inputs) 
                
                batch_size = inputs.size(0)
                time_steps = inputs.size(3)
                
                states = [torch.zeros(batch_size, c, h, p, device=self.device) for (c, h, p) in self.model.block_specs]
                
                logits = None
                for t in range(time_steps):
                    frame = inputs[:, :, :, t:t+1]
                    logits, *states = self.model(frame, *states)
                    
                loss = F.cross_entropy(logits, labels)
                total_loss += loss.item()
                
                predicted = torch.argmax(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        return total_loss / len(self.valid_loader), 100 * correct / total

    def __call__(self):
        if self.export:
            # Force cleanup before XLA runs
            if hasattr(self, 'train_loader'):
                del self.train_loader
                del self.valid_loader
                del self.test_loader
                gc.collect()
            export_streaming_models(self.model, self.n_mels, self.save_path)
            self.save_labels()
            return
            
        # Updated weight_decay to 1e-3 to perfectly match Qualcomm official specification
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)
        
        steps_per_epoch = len(self.train_loader)
        warmup_steps = steps_per_epoch * self.warmup_epochs
        
        if self.start_epoch <= self.warmup_epochs:
            warmup_scheduler = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_steps)
        else:
            warmup_scheduler = None
            
        plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=self.patience, min_lr=1e-6)
        best_loss = float('inf')

        if self.resume and os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                plateau_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                best_loss = checkpoint.get("best_loss", float('inf'))
                self.start_epoch = checkpoint.get("epoch", 0) + 1
            else:
                print("   -> Loaded old weights only. Optimizer reset.")

        for epoch in range(self.start_epoch, self.epochs + 1):
            current_warmup = warmup_scheduler if epoch <= self.warmup_epochs else None
            
            train_loss, train_acc = self.train_epoch(optimizer, current_warmup, epoch)
            val_loss, val_acc = self.validate()
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")
            
            gc.collect()
            
            if epoch == self.warmup_epochs:
                print("    (Warmup Complete. Transitioning to ReduceLROnPlateau)")
                
            if epoch > self.warmup_epochs:
                plateau_scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": plateau_scheduler.state_dict(),
                    "best_loss": best_loss
                }
                torch.save(checkpoint, self.save_path)
                print(f"--> ⭐ New Best Loss! ({best_loss:.4f}) Saved full state to {self.save_path}")
        
        print(f"\nTraining Finished. Best Validation Loss: {best_loss:.4f}")
        self.save_labels()
        
        # Cleanup and Export automatically
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=0)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=0)
        gc.collect()
        
        if os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path, map_location=self.device)
            self.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        export_streaming_models(self.model, self.n_mels, self.save_path)

if __name__ == "__main__":
    trainer = Trainer()
    trainer()