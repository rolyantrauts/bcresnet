import torch
import torch.nn as nn
import torch.nn.functional as F

class SubSpectralNorm(nn.Module):
    def __init__(self, num_features, spec_groups=16, dim=2):
        super().__init__()
        self.spec_groups = spec_groups
        self.bn = nn.BatchNorm2d(num_features * spec_groups)
        self.dim = dim

    def forward(self, x):
        b, c, h, w = x.size()
        if self.dim == 2: 
            # Interleave frequency bands for independent normalization
            x = x.view(b, c * self.spec_groups, h // self.spec_groups, w)
            x = self.bn(x)
            x = x.view(b, c, h, w)
        return x

class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1, use_ssn=False):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_planes, out_planes, (kernel_size, 1), 
                              stride=(stride, 1), padding=(padding, 0), groups=groups, bias=False)
        
        if use_ssn:
            # SSN splits frequency into 5 groups (optimal per paper)
            self.bn = SubSpectralNorm(out_planes, spec_groups=5)
        else:
            self.bn = nn.BatchNorm2d(out_planes)
            
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class BCResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, use_ssn=False):
        super().__init__()
        # 2D Branch (Frequency Processing)
        self.f2 = nn.Sequential(
            ConvBNReLU(in_planes, out_planes, kernel_size=3, stride=stride, use_ssn=use_ssn),
            ConvBNReLU(out_planes, out_planes, kernel_size=3, stride=1, use_ssn=use_ssn)
        )
        # 1D Branch (Temporal Processing)
        self.f1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(out_planes, out_planes, kernel_size=(1, 3), padding=(0, 1), groups=out_planes, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.SiLU() 
        )
        self.add = nn.Conv2d(out_planes, out_planes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.f2(x)
        res = self.f1(out) 
        out = out + res 
        return self.relu(out)

class BCResNet(nn.Module):
    def __init__(self, num_classes=3, base_channels=8, multipliers=[1, 1.5, 2, 2.5], use_ssn=True):
        super().__init__()
        self.conv1 = nn.Conv2d(1, base_channels, 5, stride=(2, 1), padding=(2, 2))
        self.blocks = nn.ModuleList()
        
        in_planes = base_channels
        for i, m in enumerate(multipliers):
            out_planes = int(base_channels * m)
            self.blocks.append(BCResBlock(in_planes, out_planes, stride=1 if i==0 else 2, use_ssn=use_ssn))
            in_planes = out_planes
            self.blocks.append(BCResBlock(in_planes, out_planes, use_ssn=use_ssn))
            
        self.conv2 = nn.Conv2d(in_planes, int(in_planes*1.5), 5, padding=(2, 2))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(in_planes*1.5), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

def BCResNets(tau=1.0, num_classes=3, use_ssn=True):
    base = int(8 * tau)
    return BCResNet(num_classes=num_classes, base_channels=base, use_ssn=use_ssn)