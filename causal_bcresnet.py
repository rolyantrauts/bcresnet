import torch
import torch.nn as nn
import torch.nn.functional as F

class SubSpectralNorm(nn.Module):
    def __init__(self, num_features, spec_groups=5, eps=1e-5):
        super().__init__()
        self.spec_groups = spec_groups
        self.eps = eps
        self.bn = nn.BatchNorm2d(num_features * spec_groups)

    def forward(self, x):
        # x: [Batch, Channel, Freq, Time]
        b, c, f, t = x.size()
        if f % self.spec_groups != 0:
            return x
        x = x.view(b, c * self.spec_groups, f // self.spec_groups, t)
        x = self.bn(x)
        return x.view(b, c, f, t)

class CausalConv2d(nn.Module):
    """
    1D Temporal Convolution that strictly pads the past (Left side).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
        super().__init__()
        self.kernel_size = kernel_size 
        self.dilation = dilation
        
        # Calculate strict left-side padding
        # kernel_size is usually (1, K), dilation is (1, D)
        k = kernel_size[1] if isinstance(kernel_size, tuple) else kernel_size
        d = dilation[1] if isinstance(dilation, tuple) else dilation
        
        self.padding_width = (k - 1) * d
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=0, dilation=dilation, groups=groups, bias=False)

    def forward(self, x):
        # Training Mode: Pad strictly on the left
        if self.padding_width > 0:
            x = F.pad(x, (self.padding_width, 0, 0, 0), value=0.0)
        return self.conv(x)

class BCResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilation=1, dropout=0.1, use_ssn=True):
        super().__init__()
        
        # Norm 1
        self.norm1 = SubSpectralNorm(in_channels) if use_ssn else nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.relu1 = nn.ReLU()
        
        # Norm 2 + Causal Conv
        self.norm2 = SubSpectralNorm(out_channels) if use_ssn else nn.BatchNorm2d(out_channels)
        self.conv2 = CausalConv2d(out_channels, out_channels, kernel_size=(1, 3), 
                                  stride=stride, dilation=(1, dilation), groups=out_channels)
        self.relu2 = nn.ReLU()
        
        # Norm 3
        self.norm3 = SubSpectralNorm(out_channels) if use_ssn else nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)

    def forward(self, x):
        identity = x
        out = self.norm1(x)
        out = self.conv1(out)
        out = self.relu1(out)
        
        out = self.norm2(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        
        out = self.norm3(out)
        out = self.conv3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        return out

class BCResNet(nn.Module):
    def __init__(self, tau=1.0, num_classes=12, use_ssn=True):
        super().__init__()
        
        # Calculate Base Channels using Tau
        base_channels = int(8 * tau)
        
        # Initial Causal Conv (5x5 context)
        self.init_conv = CausalConv2d(1, base_channels, kernel_size=(1, 5), stride=(1, 1))
        
        self.blocks = nn.Sequential(
            BCResBlock(base_channels, base_channels*2, stride=1, dilation=1, use_ssn=use_ssn),
            BCResBlock(base_channels*2, base_channels*2, stride=1, dilation=2, use_ssn=use_ssn),
            BCResBlock(base_channels*2, base_channels*4, stride=1, dilation=4, use_ssn=use_ssn),
            BCResBlock(base_channels*4, base_channels*4, stride=1, dilation=8, use_ssn=use_ssn),
            BCResBlock(base_channels*4, base_channels*8, stride=1, dilation=1, use_ssn=use_ssn),
            BCResBlock(base_channels*8, base_channels*8, stride=1, dilation=2, use_ssn=use_ssn),
        )
        
        last_channel = base_channels * 8
        
        # Classifier: 1x1 Conv replaces Global Average Pooling + Linear
        # This preserves the time dimension for streaming
        self.classifier = nn.Conv2d(last_channel, num_classes, kernel_size=1)

    def forward(self, x):
        # x: [Batch, 1, Freq, Time]
        x = self.init_conv(x)
        x = self.blocks(x)
        x = self.classifier(x)
        
        # If Frequency dimension still exists (e.g. 1), mean it out.
        # But KEEP the Time dimension.
        if x.shape[2] > 1:
             x = x.mean(dim=2, keepdim=True)
             
        # Squeeze Freq (dim 2) -> [Batch, Class, Time]
        return x.squeeze(2)

# Alias for compatibility with main2.py style instantiation
BCResNets = BCResNet