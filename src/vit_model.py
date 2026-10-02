"""A small Vision Transformer classifier, trained from scratch on the simulated
single-lens / non-lens images (100x100 grayscale). Sized down from standard ViT
configs (vit_tiny etc.) to fit our ~800-image dataset and 1-channel input."""
from __future__ import annotations

from timm.models.vision_transformer import VisionTransformer


def build_lens_vit(img_size: int = 100, patch_size: int = 10, in_chans: int = 1,
                    embed_dim: int = 128, depth: int = 4, num_heads: int = 4,
                    drop_rate: float = 0.1, attn_drop_rate: float = 0.1) -> VisionTransformer:
    assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=1,            # single logit: lens vs. non-lens
        global_pool="token",      # classify from the CLS token
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
    )
