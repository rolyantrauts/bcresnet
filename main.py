import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import json

from bcresnet import BCResNets
from utils import CustomAudioDataset, Padding, Preprocess
import torch.onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

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
        parser.add_argument("--duration", default=1.0, type=float)
        parser.add_argument("--sample_rate", default=16000, type=int)
        parser.add_argument("--n_mels", default=80, type=int)
        parser.add_argument("--no_ssn", action="store_true")
        
        # Training Hyperparams
        parser.add_argument("--epochs", default=100, type=int, help="Total training epochs")
        parser.add_argument("--warmup_epochs", default=5, type=int, help="Epochs to warm up learning rate")
        parser.add_argument("--lr", default=0.1, type=float, help="Initial learning rate")
        parser.add_argument("--patience", default=15, type=int, help="Early stopping patience")

        args = parser.parse_args()
        self.__dict__.update(vars(args))
        
        self.use_ssn = not self.no_ssn
        self.target_samples = int(self.duration * self.sample_rate)
        
        print("\n" + "="*40)
        print("      TRAINING CONFIGURATION      ")
        print("="*40)
        print(f"Device       : {self.device}")
        print(f"Duration     : {self.duration}s ({self.target_samples} samples)")
        print(f"Mel Bins     : {self.n_mels}")
        print(f"Model Tau    : {self.tau} (SSN={self.use_ssn})")
        print(f"Training     : {self.epochs} Epochs")
        print(f"Patience     : {self.patience} Epochs")
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
        
        self.train_dataset = CustomAudioDataset(self.data_root, subset="training", transform=transform, sample_rate=self.sample_rate)
        self.valid_dataset = CustomAudioDataset(self.data_root, subset="validation", transform=transform, sample_rate=self.sample_rate)
        
        if os.path.exists(os.path.join(self.data_root, "testing")):
            self.test_dataset = CustomAudioDataset(self.data_root, subset="testing", transform=transform, sample_rate=self.sample_rate)
        else:
            self.test_dataset = self.valid_dataset

        self.num_classes = len(self.train_dataset.classes)
        print(f"Detected Classes: {self.train_dataset.classes} (Total: {self.num_classes})")

        workers = 2 if os.name != 'nt' else 0
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=workers)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=workers)

        self.preprocess_train = Preprocess(self.device, sample_rate=self.sample_rate, n_mels=self.n_mels, specaug=True)
        self.preprocess_test = Preprocess(self.device, sample_rate=self.sample_rate, n_mels=self.n_mels, specaug=False)

    def _load_model(self):
        print(f"Building BCResNet-%.1f..." % self.tau)
        self.model = BCResNets(tau=self.tau, num_classes=self.num_classes, use_ssn=self.use_ssn).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total Parameters: {total_params:,}")

    def save_labels(self):
        """Exports class names and indices to text and JSON files."""
        # 1. Text File (Human Readable)
        with open("labels.txt", "w") as f:
            for i, cls in enumerate(self.train_dataset.classes):
                f.write(f"{i}: {cls}\n")
        
        # 2. JSON File (Machine Readable)
        with open("labels.json", "w") as f:
            json.dump(self.train_dataset.class_to_idx, f, indent=4)
            
        print("[Success] Saved 'labels.txt' and 'labels.json'")

    def Test(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = self.preprocess_test(inputs) 
                outputs = self.model(inputs)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def export_onnx(self):
        print("\n--- Exporting to ONNX ---")
        class EndToEndModel(nn.Module):
            def __init__(self, preprocessor, model):
                super().__init__()
                self.preprocessor = preprocessor
                self.model = model
            def forward(self, x):
                if x.dim() == 2: x = x.unsqueeze(1)
                x = self.preprocessor(x) 
                return self.model(x)

        cpu_device = torch.device("cpu")
        preprocessor_cpu = Preprocess(cpu_device, sample_rate=self.sample_rate, n_mels=self.n_mels, specaug=False)
        self.model.to(cpu_device)
        self.model.eval()
        full_model = EndToEndModel(preprocessor_cpu, self.model)
        dummy_input = torch.randn(1, self.target_samples, requires_grad=False).to(cpu_device)
        
        f32_path = "bcresnet_float32.onnx"
        int8_path = "bcresnet_int8.onnx"

        try:
            # --- FIX: Use Opset 12 to handle STFT Complex numbers correctly ---
            torch.onnx.export(
                full_model, 
                dummy_input, 
                f32_path, 
                export_params=True, 
                opset_version=12,  # <--- CHANGED FROM 17 to 12
                do_constant_folding=False, # <--- CHANGED to False to prevent JIT errors
                input_names=['audio_input'], 
                output_names=['output'],
                dynamic_axes={'audio_input': {0: 'batch'}, 'output': {0: 'batch'}}
            )
            print(f"[Success] Saved Float32 ONNX: {f32_path}")
        except Exception as e:
            print(f"[Error] Float32 Export: {e}")
            return # Stop here if export fails

        try:
            print("Quantizing to Int8...")
            quantize_dynamic(f32_path, int8_path, weight_type=QuantType.QUInt8)
            print(f"[Success] Saved Int8 ONNX: {int8_path}")
            
            # Print Size Comparison
            size_f32 = os.path.getsize(f32_path) / 1024
            size_int8 = os.path.getsize(int8_path) / 1024
            print(f"Size Reduction: {size_f32:.2f} KB -> {size_int8:.2f} KB")
        except Exception as e:
            print(f"[Error] Quantization: {e}")

    def __call__(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0, weight_decay=1e-4, momentum=0.9)
        
        steps_per_epoch = len(self.train_loader)
        warmup_steps = steps_per_epoch * self.warmup_epochs
        total_steps = steps_per_epoch * self.epochs
        global_step = 0
        
        best_acc = 0.0
        patience_counter = 0  
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")
            
            for inputs, labels in loop:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = self.preprocess_train(inputs, augment=True)
                
                # Scheduler
                global_step += 1
                if global_step < warmup_steps:
                    lr = self.lr * global_step / warmup_steps
                else:
                    progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
                    lr = self.lr * 0.5 * (1 + np.cos(np.pi * progress))
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                
                loop.set_postfix(loss=loss.item(), lr=lr)

            # Validation
            val_acc = self.Test(self.valid_loader)
            
            if epoch > self.warmup_epochs:
                if val_acc > best_acc:
                    best_acc = val_acc
                    patience_counter = 0 
                    torch.save(self.model.state_dict(), "best_bcresnet.pth")
                    print(f"Epoch {epoch} | Val Acc: {val_acc:.2f}% (New Best!)")
                else:
                    patience_counter += 1
                    print(f"Epoch {epoch} | Val Acc: {val_acc:.2f}% (No Improvement. Patience: {patience_counter}/{self.patience})")
                    
                    if patience_counter >= self.patience:
                        print(f"\n>>> Early Stopping Triggered! (No improvement for {self.patience} epochs)")
                        break
            else:
                print(f"Epoch {epoch} | Val Acc: {val_acc:.2f}% (Warmup)")
        
        print(f"\nTraining Finished. Best Accuracy: {best_acc:.2f}%")
        
        # --- EXPORT ARTIFACTS ---
        self.save_labels()  # Save labels.txt and labels.json
        
        if os.path.exists("best_bcresnet.pth"):
            self.model.load_state_dict(torch.load("best_bcresnet.pth", map_location=self.device))
        self.export_onnx()

if __name__ == "__main__":
    trainer = Trainer()
    trainer()