"""Run DeepDream on the trained small ViT lens classifier.

The ViT has a fixed 100x100 input (patch + position embeddings are sized for
it), so octave-based multi-scale DeepDream -- which resizes the working image
between octaves -- isn't compatible here; every dream runs at a single scale
(octaves=1).

Two kinds of dream:
  - "head" target: maximizes the lens-classification logit directly -- shows
    what the model considers maximally lens-like.
  - an intermediate transformer block: maximizes mean token activation --
    a class-agnostic "texture" dream showing lower/mid-level learned features.

Each is run from three seed images: random noise, a real simulated lens, and
a real simulated non-lens (to see whether the model hallucinates lens-like
structure into a non-lens image when pushed).
"""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from PIL import Image

from vit_model import build_lens_vit
from train_cnn import pick_device, load_dataset
from deepdream import DeepDreamer

ROOT = Path(__file__).resolve().parent.parent

# The ViT has a fixed 100x100 input (patch + position embeddings are sized for
# it), so octaves=1 everywhere below -- multi-scale DeepDream would resize the
# working image between octaves, which this architecture can't accept.
VIT_DREAM_KWARGS = dict(octaves=1, iterations=200, lr=0.03, jitter=4,
                         tv_weight=0.02, grad_blur_sigma=0.6,
                         image_blur_every=10, image_blur_sigma=0.4)


def load_vit(ckpt_path=ROOT / "outputs" / "lens_vit.pt"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_lens_vit(img_size=ckpt["img_size"])
    model.load_state_dict(ckpt["state_dict"])
    return model


def save_gray(tensor: torch.Tensor, path: Path):
    arr = tensor.squeeze().numpy()
    arr = np.flipud(arr)  # match export_images.py's row convention
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L").save(path)


def main():
    device = pick_device("auto")
    model = load_vit()

    images, labels = load_dataset(ROOT / "data" / "lens_dataset.npz")
    lens_img = torch.from_numpy(images[labels == 1][0]).view(1, 1, *images.shape[1:])
    nonlens_img = torch.from_numpy(images[labels == 0][0]).view(1, 1, *images.shape[1:])
    noise_img = torch.rand(1, 1, *images.shape[1:])

    out_dir = ROOT / "outputs" / "deepdream_vit"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = {"noise": noise_img, "real_lens": lens_img, "real_nonlens": nonlens_img}

    # 1) maximize the lens-classification logit ("head")
    dreamer_head = DeepDreamer(model, layer_name="head", device=device)
    for name, seed in seeds.items():
        result = dreamer_head.dream(seed, **VIT_DREAM_KWARGS)
        save_gray(result, out_dir / f"head_maximize_{name}.png")
        print(f"saved head_maximize_{name}.png")

    # 2) class-agnostic texture dream on an intermediate transformer block
    dreamer_block = DeepDreamer(model, layer_name="blocks.2", device=device)
    for name, seed in seeds.items():
        result = dreamer_block.dream(seed, **VIT_DREAM_KWARGS)
        save_gray(result, out_dir / f"block2_texture_{name}.png")
        print(f"saved block2_texture_{name}.png")


if __name__ == "__main__":
    main()
