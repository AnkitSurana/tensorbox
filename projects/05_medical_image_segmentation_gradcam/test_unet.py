"""
Tests for UNet segmentation.
"""

import torch
from unet import UNet

def test_unet_output_shape():
    model = UNet(in_channels=1, num_classes=1)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)
