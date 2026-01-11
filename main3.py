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
import copy
import sys
import warnings

# --- WARNING FILTER ---
# Suppress the specific future warning from torchaudio (Matches main2.py)
warnings.filterwarnings("ignore", message=".*load_with_torchcodec.*")
# ----------------------

# Import Causal Model instead of standard
from causal_bcresnet import BCResNets, CausalConv2d
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

# --- Streaming Wrapper for ONNX Export ---
class StreamingBCResNetWrapper(nn.Module):
    """
    Wraps the trained Causal BCResNet to expose state buffers as inputs/outputs
    """
    def __init__(self, trained_model):
        super().__init__()
        self.model = copy.deepcopy(trained_model)
        self.model.eval()
        self.causal_layers = [m for m in self.model.modules() if isinstance(m, CausalConv2d)]

    def forward(self, x, *states):
        new_states = []
        state_idx = 0
        
        def make_streaming_forward(layer, current_state):
            def custom_forward(input_x):
                # Concatenate [State] + [New Input]
                combined = torch.cat((current_state, input_x), dim=3)
                out = layer.conv(combined)
                
                # Update State (Keep trailing end)
                needed = layer.padding_width
                if needed > 0:
                    new_state = combined[..., -needed:]
                else:
                    new_state = current_state 
                new_states.append(new_state)
                return out
            return custom_forward

        original_forwards = {}
        for layer in self.causal_layers:
            original_forwards[layer] = layer.forward
            layer.forward = make_streaming_forward(layer, states[state_idx])
            state_idx += 1
            
        output = self.model(x)
        
        for layer in self.causal_layers:
            layer.forward = original_forwards[layer]
            
        output = torch.softmax(output, dim=1)
        return output, tuple(new_states)

class Trainer:
    def __init__(self):
        parser = argparse.ArgumentParser()
        
        # --- Model / Training Config ---
        parser.add_argument("--tau", default=1.0, type=float)
        parser.add_argument("--device", default="auto", type=str)
        parser.add_argument("--batch_size", default=64, type=int)
        parser.add_argument("--data_root", default="./dataset", type=str)
        parser.add_argument("--duration", default=1.0, type=float)
        parser.add_argument("--sample_rate", default=16000, type=int)
        
        parser.add_argument("--n_mels", default=40, type=int, help="Mel bins")
        parser.add_argument("--no_ssn", action="store_true", help="Disable SSN")
        parser.add_argument("--spec_prob", default=0.8, type=float, help="SpecAugment Probability")
        
        parser.add_argument("--epochs", default=100, type=int)
        parser.add_argument("--warmup_epochs", default=5, type=int)
        
        parser.add_argument("--lr", default=0.0005, type=float, help="Learning Rate") 
        parser.add_argument("--patience", default=15, type=int)
        
        parser.add_argument("--num_workers", default=0, type=int, help="Number of DataLoader workers")

        # --- Workflow Flags ---
        parser.add_argument("--export-only", action="store_true", 
                            help="Skip training and only export the model to ONNX")
        parser.add_argument("--checkpoint_path", type=str, default="best_streaming_bcresnet.pth", 
                            help="Path to save/load the model checkpoint")

        args = parser.parse_args()
        self.__dict__.update(vars(args))
        
        if self.n_mels < 80:
            print(f"[WARNING] n_mels={self.n_mels} is too small for SSN. Disabling SSN.")
            self.no_ssn = True

        self.use_ssn = not self.no_ssn
        self.target_samples = int(self.duration * self.sample_rate)
        
        print("\n" + "="*40)
        print("    TRAINING CONFIGURATION (STREAMING)    ")
        print("="*40)
        print(f"Device        : {self.device}")
        print(f"Duration      : {self.duration}s ({self.target_samples} samples)")
        print(f"Mel Bins      : {self.n_mels}")
        print(f"Model Tau     : {self.tau} (SSN={self.use_ssn})")
        print(f"Export Only   : {self.export_only}")
        print(f"Checkpoint    : {self.checkpoint_path}")
        print(f"Learning Rate : {self.lr}")
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
        
        # Load Data (Only if training, or we need class info)
        self._load_data()
        self._load_model()

    def _load_data(self):
        print(f"Loading data from {self.data_root}...")
        transform = transforms.Compose([Padding(target_len=self.target_samples)])
        
        # We always load training set to get class definitions
        self.train_dataset = CustomAudioDataset(self.data_root, subset="training", transform=transform, sample_rate=self.sample_rate)
        
        self.num_classes = len(self.train_dataset.classes)
        print(f"Detected Classes: {self.train_dataset.classes} (Total: {self.num_classes})")

        if not self.export_only:
            self.valid_dataset = CustomAudioDataset(self.data_root, subset="validation", transform=transform, sample_rate=self.sample_rate)
            
            if os.path.exists(os.path.join(self.data_root, "testing")):
                self.test_dataset = CustomAudioDataset(self.data_root, subset="testing", transform=transform, sample_rate=self.sample_rate)
            else:
                self.test_dataset = self.valid_dataset

            # Use the CLI argument for workers
            workers = self.num_workers
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=workers)
            self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=workers)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=workers)

            self.preprocess_train = Preprocess(self.device, sample_rate=self.sample_rate, n_mels=self.n_mels, specaug=True)
            self.preprocess_test = Preprocess(self.device, sample_rate=self.sample_rate, n_mels=self.n_mels, specaug=False)

    def _load_model(self):
        print(f"Building Causal BCResNet-%.1f..." % self.tau)
        self.model = BCResNets(tau=self.tau, num_classes=self.num_classes, use_ssn=self.use_ssn).to(self.device)
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
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = self.preprocess_test(inputs) 
                outputs = self.model(inputs)
                
                # Causal Model returns [Batch, Class, Time]
                # We mean-pool over Time to get clip classification
                outputs_pooled = outputs.mean(dim=-1)
                
                predicted = torch.argmax(outputs_pooled, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        if total == 0: return 0.0
        return 100 * correct / total

    def export_onnx(self):
        print("\n--- Exporting to Streaming ONNX (Stateful) ---")
        print("NOTE: This exports a TRUE STREAMING model.")
        print(f"Input Shape: [1, 1, {self.n_mels}, 1] (Single Time Step)")
        
        cpu_device = torch.device("cpu")
        # Copy model to CPU for export
        model_cpu = copy.deepcopy(self.model).to(cpu_device)
        wrapper = StreamingBCResNetWrapper(model_cpu)
        
        # 1. Dummy Input (Single Frame)
        dummy_input = torch.randn(1, 1, self.n_mels, 1).to(cpu_device)
        
        # 2. Dummy States
        dummy_states = []
        for layer in wrapper.causal_layers:
            # State buffer: [1, Channels, Freq, PaddingWidth]
            state = torch.zeros(1, layer.conv.in_channels, self.n_mels, layer.padding_width).to(cpu_device)
            dummy_states.append(state)
            
        input_names = ['input'] + [f'state_in_{i}' for i in range(len(dummy_states))]
        output_names = ['output'] + [f'state_out_{i}' for i in range(len(dummy_states))]
        
        f32_path = "streaming_bcresnet_float32.onnx"
        int8_path = "streaming_bcresnet_int8.onnx"

        try:
            torch.onnx.export(
                wrapper, 
                (dummy_input, *tuple(dummy_states)), 
                f32_path, 
                export_params=True, 
                opset_version=13, 
                do_constant_folding=True,
                input_names=input_names, 
                output_names=output_names
            )
            print(f"[Success] Saved Float32 Streaming ONNX: {f32_path}")
        except Exception as e:
            print(f"[Error] Float32 Export: {e}")
            return

        try:
            print("Quantizing to Int8...")
            quantize_dynamic(f32_path, int8_path, weight_type=QuantType.QUInt8)
            print(f"[Success] Saved Int8 Streaming ONNX: {int8_path}")
            
            size_f32 = os.path.getsize(f32_path) / 1024
            size_int8 = os.path.getsize(int8_path) / 1024
            print(f"Size Reduction: {size_f32:.2f} KB -> {size_int8:.2f} KB")
        except Exception as e:
            print(f"[Error] Quantization: {e}")

    def __call__(self):
        # -------------------------------------------------------------
        # BRANCH A: Export Only (Skip Training)
        # -------------------------------------------------------------
        if self.export_only:
            print(f"Skipping training. Loading weights from {self.checkpoint_path}...")
            if not os.path.exists(self.checkpoint_path):
                print(f"[Error] Checkpoint not found: {self.checkpoint_path}")
                sys.exit(1)
            
            try:
                state_dict = torch.load(self.checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print("Weights loaded successfully.")
            except Exception as e:
                print(f"[Error] Failed to load weights: {e}")
                sys.exit(1)

        # -------------------------------------------------------------
        # BRANCH B: Training Loop
        # -------------------------------------------------------------
        else:
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
                    
                    should_augment = (np.random.rand() < self.spec_prob)
                    inputs = self.preprocess_train(inputs, augment=should_augment)
                    
                    global_step += 1
                    # Custom LR Scheduler
                    if global_step < warmup_steps:
                        lr = self.lr * global_step / warmup_steps
                    else:
                        progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
                        lr = self.lr * 0.5 * (1 + np.cos(np.pi * progress))
                    
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    
                    # Pool time dimension for training loss
                    outputs_pooled = outputs.mean(dim=-1)
                    
                    loss = F.cross_entropy(outputs_pooled, labels)
                    loss.backward()
                    optimizer.step()
                    loop.set_postfix(loss=loss.item(), lr=lr)

                # Validation
                val_acc = self.Test(self.valid_loader)
                
                # --- FIXED LOGIC: Strict Warmup Isolation ---
                if epoch <= self.warmup_epochs:
                    # STRICT WARMUP: Do NOT check for best, do NOT save, do NOT patience.
                    print(f"Epoch {epoch} | Val Acc: {val_acc:.2f}% (Warmup - Saving Frozen)")
                else:
                    # NORMAL TRAINING
                    if val_acc > best_acc:
                        best_acc = val_acc
                        patience_counter = 0
                        torch.save(self.model.state_dict(), self.checkpoint_path)
                        print(f"Epoch {epoch} | Val Acc: {val_acc:.2f}% (New Best!)")
                    else:
                        patience_counter += 1
                        print(f"Epoch {epoch} | Val Acc: {val_acc:.2f}% (Patience: {patience_counter}/{self.patience})")
                        if patience_counter >= self.patience:
                            print(f"\n>>> Early Stopping Triggered!")
                            break
            
            print(f"\nTraining Finished. Best Accuracy: {best_acc:.2f}%")
            self.save_labels()
            
            # Load best weights before export
            if os.path.exists(self.checkpoint_path):
                print(f"Loading best weights from {self.checkpoint_path}...")
                self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))

        # -------------------------------------------------------------
        # 3. Export to ONNX (Runs for both branches)
        # -------------------------------------------------------------
        self.export_onnx()

if __name__ == "__main__":
    trainer = Trainer()
    trainer()