"""
Spatial-Temporal 1D Convolutional Front-End for Wearable EEG.
Combines:
  1. Temporal Convolution: learns frequency-selective filters across time.
  2. Spatial Depthwise Convolution: learns spatial montage topographies across wearable scalp channels.
  3. Separable Convolutions: decouples temporal frequency extraction from feature recombination.
"""

import torch
import torch.nn as nn
from config.settings import ModelConfig

class SpatialTemporalCNNBlock(nn.Module):
    """
    Lightweight, highly efficient spatial-temporal feature extractor tailored for
    low-channel wearable EEG (AF3, AF4, F3, F4, T7, T8, P7, P8).
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_channels = config.num_channels
        self.f1 = config.temporal_filters
        self.d = config.spatial_depth_multiplier
        self.f2 = self.f1 * self.d
        self.kernel_size = config.temporal_kernel_size

        # Stage 1: Temporal Convolution (Kernel covers ~250ms of EEG wave dynamics)
        # Input shape: (Batch, 1, Channels, TimeSamples)
        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.f1,
            kernel_size=(1, self.kernel_size),
            padding=(0, self.kernel_size // 2),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.f1)

        # Stage 2: Spatial Depthwise Convolution (Mixes wearable scalp channels)
        self.spatial_conv = nn.Conv2d(
            in_channels=self.f1,
            out_channels=self.f2,
            kernel_size=(self.num_channels, 1),
            groups=self.f1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(self.f2)
        self.elu = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout1 = nn.Dropout(0.25)

        # Stage 3: Separable Temporal Convolution
        self.depthwise_sep = nn.Conv2d(
            in_channels=self.f2,
            out_channels=self.f2,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=self.f2,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels=self.f2,
            out_channels=self.f2,
            kernel_size=(1, 1),
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.f2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout2 = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Channels, TimeSamples)
        Returns:
            features: Tensor of shape (Batch, Reduced_TimeSamples, Features)
        """
        # Reshape to (Batch, 1, Channels, TimeSamples)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        out = self.temporal_conv(x)
        out = self.bn1(out)

        out = self.spatial_conv(out)
        out = self.bn2(out)
        out = self.elu(out)
        out = self.pool1(out)
        out = self.dropout1(out)

        out = self.depthwise_sep(out)
        out = self.pointwise(out)
        out = self.bn3(out)
        out = self.elu(out)
        out = self.pool2(out)
        out = self.dropout2(out)

        # out shape is (Batch, F2, 1, Reduced_TimeSamples)
        out = out.squeeze(2) # (Batch, F2, Reduced_TimeSamples)
        out = out.permute(0, 2, 1) # (Batch, Reduced_TimeSamples, F2)
        return out
