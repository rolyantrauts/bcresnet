import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Replicated from bcresnet.py for standalone structural integrity ---
class SubSpectralNorm(nn.Module):
    def __init__(self, num_features, spec_groups=16, dim=2):
        super().__init__()
        self.spec_groups = spec_groups
        self.bn = nn.BatchNorm2d(num_features * spec_groups)
        self.dim = dim

    def forward(self, x):
        b, c, h, w = x.size()
        if self.dim == 2: 
            x = x.view(b, c * self.spec_groups, h // self.spec_groups, w)
            x = self.bn(x)
            x = x.view(b, c, h, w)
        return x

class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1, use_ssn=False):
        super().__init__()
        padding = (kernel_size - 1) // 2
        # Freq-only 1D Convolution
        self.conv = nn.Conv2d(in_planes, out_planes, (kernel_size, 1), 
                              stride=(stride, 1), padding=(padding, 0), groups=groups, bias=False)
        
        if use_ssn:
            self.bn = SubSpectralNorm(out_planes, spec_groups=5)
        else:
            self.bn = nn.BatchNorm2d(out_planes)
            
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# --- Streaming Architecture ---

class StreamingBCResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, use_ssn=False):
        super().__init__()
        # 2D Branch (Frequency Processing - Independent per frame)
        self.f2 = nn.Sequential(
            ConvBNReLU(in_planes, out_planes, kernel_size=3, stride=stride, use_ssn=use_ssn),
            ConvBNReLU(out_planes, out_planes, kernel_size=3, stride=1, use_ssn=use_ssn)
        )
        
        self.temporal_pad = 2
        
        # 1D Branch (Temporal Processing)
        self.f1_conv1 = nn.Conv2d(out_planes, out_planes, kernel_size=(1, 3), 
                                 padding=0, groups=out_planes, bias=False)
        self.f1_bn = nn.BatchNorm2d(out_planes)
        self.f1_act = nn.SiLU()

    def forward(self, x, state):
        # 1. Process F2 (Frequency)
        out_f2 = self.f2(x)
        
        # 2. Pool Frequency Dimension down to 1
        out_pooled = torch.mean(out_f2, dim=2, keepdim=True) 
        
        # 3. Concatenate Temporal State
        combined = torch.cat((state, out_pooled), dim=3)
        new_state = combined[:, :, :, 1:] # Shift buffer
        
        # 4. Process F1 (Time) on the 3 buffered frames
        res = self.f1_conv1(combined)
        res = self.f1_bn(res)
        res = self.f1_act(res)
        
        # 5. Residual Add
        out = out_f2 + res 
        return F.relu(out, inplace=True), new_state

class StreamingBCResNet(nn.Module):
    def __init__(self, num_classes=3, base_channels=8, multipliers=[1, 1.5, 2, 2.5], use_ssn=True, dropout=0.3, n_mels=80):
        super().__init__()
        self.block_specs = []
        
        # --- STEM (conv1) ---
        self.block_specs.append((1, n_mels, 4))
        self.conv1 = nn.Conv2d(1, base_channels, kernel_size=(5, 5), stride=(2, 1), padding=(2, 0))
        
        self.blocks = nn.ModuleList()
        
        in_planes = base_channels
        for i, m in enumerate(multipliers):
            out_planes = int(base_channels * m)
            # Block 1 (Strided)
            stride = 1 if i == 0 else 2
            b1 = StreamingBCResBlock(in_planes, out_planes, stride=stride, use_ssn=use_ssn)
            self.blocks.append(b1)
            self.block_specs.append((out_planes, 1, b1.temporal_pad))
            
            # Block 2 (Unstrided)
            in_planes = out_planes
            b2 = StreamingBCResBlock(in_planes, out_planes, stride=1, use_ssn=use_ssn)
            self.blocks.append(b2)
            self.block_specs.append((out_planes, 1, b2.temporal_pad))
            
        # --- CLASSIFIER (conv2) ---
        freq_dim = n_mels // 16 
        self.block_specs.append((in_planes, freq_dim, 4))
        self.conv2 = nn.Conv2d(in_planes, int(in_planes*1.5), kernel_size=(5, 5), padding=(2, 0))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_dim = int(in_planes*1.5)
        
        # Global Classifier Dropout
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x, *states):
        new_states = []
        
        # 1. Stem
        s_conv1 = states[0]
        comb_conv1 = torch.cat((s_conv1, x), dim=3)
        new_states.append(comb_conv1[:, :, :, 1:])
        x = self.conv1(comb_conv1)
        
        # 2. Residual Blocks
        state_idx = 1
        for block in self.blocks:
            x, s_out = block(x, states[state_idx])
            new_states.append(s_out)
            state_idx += 1
            
        # 3. Classifier Feature Extraction
        s_conv2 = states[state_idx]
        comb_conv2 = torch.cat((s_conv2, x), dim=3)
        new_states.append(comb_conv2[:, :, :, 1:])
        
        x = self.conv2(comb_conv2)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        logits = self.fc(x)
        
        return (logits, *new_states)

def StreamingBCResNets(tau=1.0, num_classes=3, use_ssn=True, dropout=0.3, n_mels=80):
    base = int(8 * tau)
    return StreamingBCResNet(num_classes=num_classes, base_channels=base, use_ssn=use_ssn, dropout=dropout, n_mels=n_mels)