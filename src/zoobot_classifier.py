"""Lens/non-lens classifier built on a frozen, pretrained Zoobot (galaxy
morphology) encoder: 100x100 grayscale -> upsample to 224x224 -> replicate to
3ch -> ImageNet-normalize -> frozen convnext_nano encoder (640-dim) -> a small
trainable linear head. Only the head is trained (head_only), since Zoobot's
encoder already saw millions of real galaxy images and our ~6000 simulated
lenses would be nowhere near enough to safely fine-tune the whole thing."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

ENCODER_NAME = "hf_hub:mwalmsley/zoobot-encoder-convnext_nano"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ZoobotLensClassifier(nn.Module):
    def __init__(self, freeze_encoder: bool = True, dropout: float = 0.3):
        super().__init__()
        self.encoder = timm.create_model(ENCODER_NAME, num_classes=0, pretrained=True)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()
        self.freeze_encoder = freeze_encoder
        self.register_buffer("mean", IMAGENET_MEAN)
        self.register_buffer("std", IMAGENET_STD)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder.num_features, 1),
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W) in [0, 1]
        x = F.interpolate(x, size=224, mode="bicubic", align_corners=False)
        x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        if self.freeze_encoder:
            with torch.no_grad():
                feats = self.encoder(x)
        else:
            feats = self.encoder(x)
        return self.head(feats)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()  # keep frozen BN/etc. in eval mode even in train()
        return self
