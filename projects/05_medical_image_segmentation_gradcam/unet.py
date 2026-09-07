"""
PyTorch U-Net Segmentation Architecture.
"""

import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU())
        self.final = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        b = self.bottleneck(self.pool(e1))
        d1 = self.dec1(self.up(b))
        out = self.final(d1)
        return out
