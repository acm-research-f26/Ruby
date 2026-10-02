"""A small lens/non-lens classifier CNN, in the spirit of OU-66 (Wilde et al. 2022):
stacked Conv2D + ReLU + MaxPool2D + Dropout blocks feeding a linear head."""
from __future__ import annotations

import torch
import torch.nn as nn


class SimpleLensCNN(nn.Module):
    def __init__(self, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        # BatchNorm after every conv: without it, this net is prone to a
        # dead-ReLU collapse on an unlucky random init (observed in practice --
        # loss stuck exactly at ln(2) for 80 epochs on one seed while an
        # identical config/data converged fine on another). BN keeps pre-ReLU
        # activations well-scaled regardless of initialization.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(dropout),           # 64 -> 32

            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(dropout),           # 32 -> 16

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(dropout),           # e.g. 25 -> 12
        )
        # Adaptive pool makes the classifier head independent of input resolution
        # (64x64, 100x100, ...) so changing image size doesn't break the flatten shape.
        # 4x4 (not 8x8) because MPS's adaptive_avg_pool2d requires the input spatial
        # size to be evenly divisible by the output size, and 100x100 -> 12x12 after
        # three stride-2 pools isn't divisible by 8.
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
