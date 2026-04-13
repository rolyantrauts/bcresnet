import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from tqdm import tqdm
import json
import gc
import traceback
import warnings

# --- MUTE TORCHAUDIO RESIZE WARNING ---
warnings.filterwarnings("ignore", message="An output with one or more elements was resized")

# --- FIX FOR MACOS "TOO MANY OPEN FILES" ---
torch.multiprocessing.set_sharing_strategy('file_system')

from bcresnet import BCResNets
from utils import CustomAudioDataset, Padding, Preprocess, ArcMarginProduct

def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

class Trainer:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--tau", default=1.0, type=float)
        parser.add_argument("--device", default="auto", type=str)
        parser.add_argument("--batch_size", default=64, type=int)
        parser.add_argument("--data_root", default="./dataset", type=str)
        parser.add_argument("--clip_duration", default=1.0, type=float)
        parser.add_argument("--sample_rate", default=16000, type=int)
        
        parser.add_argument("--n_mels", default=40, type=int, help="Mel bins (Match your ESP-DL config)")
        parser.add_argument("--no_ssn", action="store_true", help="Disable SSN (Recommended for 40 mels)")
        
        parser.add_argument("--spec_prob", default=0.8, type=float, help="Probability (0.0-1.0) of applying SpecAugment")
        
        parser.add_argument("--epochs", default=100, type=int)
        parser.add_argument("--warmup_epochs", default=5, type=int)
        parser.add_argument("--lr", default=0.005, type=float)
        parser.add_argument("--patience", default=10, type=int, help="Epochs of plateau before reducing learning rate")
        
        parser.add_argument("--dropout", default=0.3, type=float, help="Dropout rate for the model classifier (default: 0.3)")
        parser.add_argument("--label_smoothing", default=0.0, type=float, help="Label smoothing epsilon for CrossEntropyLoss (default: 0.0)")
        
        parser.add_argument("--resume", action="store_true", help="Resume weights and state from checkpoint")
        parser.add_argument("--start_epoch", default=1, type=int, help="Epoch to start/resume training from")
        
        parser.add_argument("--index", default="", type=str, help='Comma-separated classes (e.g. "hey_jarvis,unknown,noise")')
        parser.add_argument("--export", action="store_true", help="Skip training, load best.pth, and export models")
        
        # --- ARCFACE TOGGLE ---
        parser.add_argument("--arcface", action="store_true", help="Use Cosine Similarity & Angular Margin (Outputs embeddings)")

        args = parser.parse_args()
        self.__dict__.update(vars(args))
        
        if self.n_mels < 80:
            print(f"[WARNING] n_mels={self.n_mels} is too small for Sub-Spectral Normalization.")
            print("          Disabling SSN automatically to prevent crash.")
            self.no_ssn = True

        self.use_ssn = not self.no_ssn
        self.target_samples = int(self.clip_duration * self.sample_rate)
        self.spec_width = int(1 + (self.target_samples / 160))
        
        print("\n" + "="*40)
        print("   TRAINING CONFIGURATION (External)   ")
        print("="*40)
        print(f"Device          : {self.device}")
        print(f"Clip Duration   : {self.clip_duration}s ({self.target_samples} samples)")
        print(f"Mel Bins        : {self.n_mels}")
        print(f"Spec Shape      : [1, 1, {self.n_mels}, {self.spec_width}] (Input for C++)")
        print(f"Model Tau       : {self.tau} (SSN={self.use_ssn})")
        print(f"Dropout         : {self.dropout}")
        print(f"Label Smoothing : {self.label_smoothing}")
        print(f"ArcFace         : {'ON (Outputting Embeddings)' if self.arcface else 'OFF (Linear Classifier)'}")
        print(f"SpecAug Prob    : {self.spec_prob * 100:.1f}%")
        print(f"Class Index     : {self.index if self.index else 'Auto-detect'}")
        
        if self.export:
            print(f"Mode            : EXPORT ONLY")
        else:
            print(f"Start Epoch     : {self.start_epoch} / {self.epochs}")
            if self.resume:
                print(f"Resuming        : YES (Attempting to load checkpoint)")
        print("="*40 + "\n")
        
        if self.device == "auto":
            self.device_name = get_default_device()
        else:
            self.device_name = self.device
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

        # --- MULTIPROCESSING DEADLOCK FIX ---
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
        print(f"Building BCResNet-%.1f..." % self.tau)
        self.model = BCResNets(tau=self.tau, num_classes=self.num_classes, use_ssn=self.use_ssn, dropout=self.dropout, use_arcface=self.arcface).to(self.device)
        
        if self.arcface:
            self.arcface_layer = ArcMarginProduct(in_features=self.model.embedding_dim, out_features=self.num_classes).to(self.device)
        else:
            self.arcface_layer = None
        
        if (self.resume or self.export) and os.path.exists("best_bcresnet.pth"):
            print("🔄 Loading model weights from best_bcresnet.pth...")
            checkpoint = torch.load("best_bcresnet.pth", map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                if self.arcface and "arcface_state_dict" in checkpoint:
                    self.arcface_layer.load_state_dict(checkpoint["arcface_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        elif self.resume or self.export:
            print("⚠️ Warning: 'best_bcresnet.pth' not found. Using randomly initialized weights.")
            
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total Parameters: {total_params:,}")

    def save_labels(self):
        with open("labels.txt", "w") as f:
            for i, cls in enumerate(self.train_dataset.classes):
                f.write(f"{i}: {cls}\n")
        with open("labels.json", "w") as f:
            json.dump(self.train_dataset.class_to_idx, f, indent=4)
        print("[Success] Saved 'labels.txt' and 'labels.json'")

    def Test(self, loader):
        self.model.eval()
        if self.arcface:
            self.arcface_layer.eval()
            
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = self.preprocess_test(inputs) 
                
                outputs = self.model(inputs)
                
                if self.arcface:
                    outputs = self.arcface_layer(outputs, label=None)
                
                # Label Smoothing is purely a training regularizer. 
                # We use strict cross_entropy for accurate validation loss metrics.
                loss = F.cross_entropy(outputs, labels)
                total_loss += loss.item()
                
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        return total_loss / len(loader), 100 * correct / total

    def export_models(self):
        # Destroy all multiprocessing iterators and force Garbage Collection
        if hasattr(self, 'train_loader'):
            del self.train_loader
            del self.valid_loader
            del self.test_loader
            gc.collect()

        print("\n" + "="*40)
        print("   EXPORTING MODELS (LiteRT / TFLite)   ")
        print("="*40)
        print("NOTE: This model expects a Mel Spectrogram as input.")
        print(f"Input Shape: [1, 1, {self.n_mels}, {self.spec_width}]")
        
        cpu_device = torch.device("cpu")
        self.model.to(cpu_device)
        self.model.eval()
        
        dummy_spectrogram = torch.randn(1, 1, self.n_mels, self.spec_width, requires_grad=False).to(cpu_device)
        sample_inputs = (dummy_spectrogram,)

        # ---------------------------------------------------------
        # EXPORT 1: LiteRT / TFLite (Float32)
        # ---------------------------------------------------------
        tflite_f32_path = "bcresnet_float32.tflite"
        try:
            import litert_torch
            print("\nConverting PyTorch model to LiteRT (Float32)...")
            edge_model_f32 = litert_torch.convert(self.model, sample_inputs)
            edge_model_f32.export(tflite_f32_path)
            print(f"[Success] Saved Float32 LiteRT: {tflite_f32_path}")
        except ImportError:
            print("\n[Warning] 'litert_torch' not installed. Skipping LiteRT (TFLite) Float32 export.")
        except Exception as e:
            print(f"\n[Error] LiteRT Float32 Export Failed: {e}")

        # ---------------------------------------------------------
        # EXPORT 2: LiteRT / TFLite (INT8 Quantization)
        # ---------------------------------------------------------
        tflite_int8_path = "bcresnet_int8.tflite"
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

        # ---------------------------------------------------------
        # EXPORT 3: Golden Weights (If ArcFace is enabled)
        # ---------------------------------------------------------
        if self.arcface:
            print("\nExtracting Golden Fingerprints for C++ Inference...")
            golden_weights = F.normalize(self.arcface_layer.weight.data, dim=1).cpu().numpy()
            
            with open("arcface_golden_weights.json", "w") as f:
                json.dump(golden_weights.tolist(), f)
            print(f"[Success] Saved Golden Fingerprints to arcface_golden_weights.json (Shape: {golden_weights.shape})")

        print("="*40 + "\n")

    def __call__(self):
        if self.export:
            self.export_models()
            self.save_labels()
            return
            
        if self.arcface:
            params = list(self.model.parameters()) + list(self.arcface_layer.parameters())
        else:
            params = self.model.parameters()
            
        # Updated weight_decay to 1e-3 to perfectly match Qualcomm official specification
        optimizer = torch.optim.SGD(params, lr=self.lr, weight_decay=1e-3, momentum=0.9)
        
        steps_per_epoch = len(self.train_loader)
        warmup_steps = steps_per_epoch * self.warmup_epochs
        
        if self.start_epoch <= self.warmup_epochs:
            warmup_scheduler = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_steps)
        else:
            warmup_scheduler = None
            
        plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=self.patience, min_lr=1e-6)
        best_loss = float('inf')

        if self.resume:
            checkpoint_path = "best_bcresnet.pth"
            if os.path.exists(checkpoint_path):
                print(f"🔄 Resuming from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    plateau_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    
                    best_loss = checkpoint.get("best_loss", float('inf'))
                    if best_loss != float('inf'):
                        print(f"   -> Loaded Full State (Best Loss: {best_loss:.4f})")
                    else:
                        print(f"   -> Loaded Full State. (Establishing new best_loss baseline this epoch)")
                    
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = self.lr
                else:
                    print("   -> Loaded Weights Only (Old save format). Optimizer memory reset.")
            else:
                print(f"⚠️ Warning: --resume passed but '{checkpoint_path}' not found. Starting from scratch.")

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.model.train()
            if self.arcface:
                self.arcface_layer.train()
                
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")
            
            for inputs, labels in loop:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                should_augment = (np.random.rand() < self.spec_prob)
                inputs = self.preprocess_train(inputs, augment=should_augment)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                
                if self.arcface:
                    outputs = self.arcface_layer(outputs, labels)
                    
                # Apply Label Smoothing during training updates
                loss = F.cross_entropy(outputs, labels, label_smoothing=self.label_smoothing)
                loss.backward()
                optimizer.step()
                
                if epoch <= self.warmup_epochs and warmup_scheduler is not None:
                    warmup_scheduler.step()
                    
                current_lr = optimizer.param_groups[0]['lr']
                loop.set_postfix(loss=loss.item(), lr=current_lr)

            val_loss, val_acc = self.Test(self.valid_loader)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")
            
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
                if self.arcface:
                    checkpoint["arcface_state_dict"] = self.arcface_layer.state_dict()
                    
                torch.save(checkpoint, "best_bcresnet.pth")
                print(f"--> ⭐ New Best Loss! ({best_loss:.4f}) Saved full state to best_bcresnet.pth")
        
        print(f"\nTraining Finished. Best Validation Loss: {best_loss:.4f}")
        self.save_labels()
        
        # --- DEADLOCK FIX FOR FULL TRAINING ---
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        gc.collect()
        
        if os.path.exists("best_bcresnet.pth"):
            checkpoint = torch.load("best_bcresnet.pth", map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                if self.arcface and "arcface_state_dict" in checkpoint:
                    self.arcface_layer.load_state_dict(checkpoint["arcface_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
                
        self.export_models()

if __name__ == "__main__":
    trainer = Trainer()
    trainer()