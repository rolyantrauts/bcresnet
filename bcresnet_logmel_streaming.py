import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import onnx
import onnxruntime as ort
import numpy as np

# --- Basic Components ---

class StreamingBCResBlock(nn.Module):
    def __init__(self, in_plane, out_plane, stage_idx, stride=(1,1)):
        super().__init__()
        self.stride = stride
        self.dilation = int(2**stage_idx)
        self.temporal_kernel = 3
        self.padding_needed = (self.temporal_kernel - 1) * self.dilation
        
        # --- Channel/Group Logic ---
        if in_plane != out_plane:
            f2_groups = 1
            f1_groups = 1 
        else:
            f2_groups = in_plane # Depthwise
            f1_groups = out_plane # Depthwise
            
        # --- Branch 2 (F2): Frequency + Temporal (Non-dilated) ---
        self.f2 = nn.Sequential(
            nn.Conv2d(in_plane, out_plane, kernel_size=(3, 1), stride=(stride[0], 1), 
                      padding=(1, 0), groups=f2_groups, bias=False),
            nn.BatchNorm2d(out_plane)
        )
        
        # --- Branch 1 (F1): Temporal Only (Dilated) ---
        self.f1_conv1 = nn.Conv2d(out_plane, out_plane, kernel_size=(1, 3), 
                                  stride=(1, 1), padding=0, 
                                  dilation=(1, self.dilation), groups=f1_groups, bias=False)
        self.f1_bn1 = nn.BatchNorm2d(out_plane)
        self.f1_act = nn.SiLU(True)
        self.f1_conv2 = nn.Conv2d(out_plane, out_plane, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(0.1)

        # Projection if channels/resolution change
        self.use_projection = (in_plane != out_plane) or (stride[0] > 1)
        if self.use_projection:
            self.proj = nn.Conv2d(in_plane, out_plane, kernel_size=1, stride=(stride[0], 1), bias=False)
            self.proj_bn = nn.BatchNorm2d(out_plane)

    def forward(self, x, state):
        # 1. Main Shortcut
        identity = x
        if self.use_projection:
            identity = self.proj_bn(self.proj(identity))
            
        # 2. F2 Branch
        out_f2 = self.f2(x)
        
        # 3. Global Avg Pool for F1 branch input
        out_pooled = torch.mean(out_f2, dim=2, keepdim=True) 
        
        # 4. Temporal State Update
        combined = torch.cat((state, out_pooled), dim=3)
        new_state = combined[:, :, :, 1:]
        
        # 5. F1 Branch
        out_f1 = self.f1_conv1(combined)
        out_f1 = self.f1_bn1(out_f1)
        out_f1 = self.f1_act(out_f1)
        out_f1 = self.f1_conv2(out_f1)
        out_f1 = self.dropout(out_f1)
        
        # 6. Add & Activation
        out = out_f1 + out_f2 + identity
        return F.relu(out, True), new_state

# --- Full Network (BC-ResNet-1 tau=1) ---

class StreamingBCResNet(nn.Module):
    def __init__(self, n_classes=12, tau=1.0):
        super().__init__()
        
        base_c = int(8 * tau) 
        self.stage_channels = [base_c * 2, base_c, int(base_c * 1.5), base_c * 2, int(base_c * 2.5), base_c * 4]
        self.num_blocks = [2, 2, 4, 4] 
        
        self.stem = nn.Sequential(
            nn.Conv2d(1, self.stage_channels[0], kernel_size=(5, 1), stride=(2, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(self.stage_channels[0]),
            nn.ReLU(True)
        )
        
        self.blocks = nn.ModuleList()
        self.block_specs = [] 
        
        # Stage Construction
        self._make_stage(0, self.num_blocks[0], self.stage_channels[0], self.stage_channels[1])
        self._make_stage(1, self.num_blocks[1], self.stage_channels[1], self.stage_channels[2], stride=(2, 1))
        self._make_stage(2, self.num_blocks[2], self.stage_channels[2], self.stage_channels[3], stride=(2, 1))
        self._make_stage(3, self.num_blocks[3], self.stage_channels[3], self.stage_channels[4])

        # Classifier
        last_c = self.stage_channels[4]
        final_c = self.stage_channels[5]
        self.classifier = nn.Sequential(
            # Reduces Freq=5 down to 1
            nn.Conv2d(last_c, last_c, kernel_size=(5, 1), padding=(0, 0), groups=last_c, bias=False),
            nn.Conv2d(last_c, final_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(final_c),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(final_c, n_classes, kernel_size=1)
        )

    def _make_stage(self, stage_idx, num_blocks, in_c, out_c, stride=(1, 1)):
        for i in range(num_blocks):
            s = stride if i == 0 else (1, 1)
            inp = in_c if i == 0 else out_c
            block = StreamingBCResBlock(inp, out_c, stage_idx, stride=s)
            self.blocks.append(block)
            self.block_specs.append((out_c, 1, block.padding_needed))

    def forward(self, x, *states):
        new_states = []
        x = self.stem(x)
        
        for i, block in enumerate(self.blocks):
            s_in = states[i]
            x, s_out = block(x, s_in)
            new_states.append(s_out)
        
        # No Global Pooling. Classifier handles 5x1 input.
        x = self.classifier(x)
        logits = x.view(x.size(0), -1)
        
        return (logits, *new_states)

# --- Export & Verify Helper Function ---

def export_onnx(model, path="bcresnet_streaming_tau1.onnx"):
    model.eval()
    
    dummy_input = torch.randn(1, 1, 40, 1)
    
    dummy_states = []
    input_names = ['input']
    output_names = ['logits']
    
    # Legacy dynamic axes (works best with dynamo=False)
    dynamic_axes = {'input': {0: 'batch'}}
    
    for i, (c, h, p) in enumerate(model.block_specs):
        s = torch.zeros(1, c, h, p) 
        dummy_states.append(s)
        
        s_name = f'state_in_{i}'
        input_names.append(s_name)
        output_names.append(f'state_out_{i}')
        
        dynamic_axes[s_name] = {0: 'batch'}

    model_inputs = (dummy_input, *dummy_states)
    
    print(f"Exporting model with {len(dummy_states)} state buffers...")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print("Using Stable (Legacy) Exporter for stability...")

    torch.onnx.export(
        model, 
        model_inputs,
        path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,
        verbose=True 
    )
    print(f"✅ Exported to {path}")


