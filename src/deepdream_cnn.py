"""Run DeepDream on the trained SimpleLensCNN classifier, for comparison against
the ViT dreams (same seeds, analogous targets: final logit vs. an intermediate
layer). CNNs support the full multi-octave DeepDreamer since they don't have a
fixed-size position embedding."""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from PIL import Image

from models import SimpleLensCNN
from train_cnn import pick_device, load_dataset
from deepdream import DeepDreamer

ROOT = Path(__file__).resolve().parent.parent


def load_cnn(ckpt_path=ROOT / "outputs" / "simple_lens_cnn.pt"):
    model = SimpleLensCNN(in_channels=1)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model


def save_gray(tensor: torch.Tensor, path: Path):
    arr = tensor.squeeze().numpy()
    arr = np.flipud(arr)
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L").save(path)


def main():
    # CPU, not MPS: octave resizing produces spatial sizes that don't evenly
    # divide the adaptive-pool output, which MPS's adaptive_avg_pool2d rejects.
    device = pick_device("cpu")
    model = load_cnn()

    images, labels = load_dataset(ROOT / "data" / "lens_dataset.npz")
    lens_img = torch.from_numpy(images[labels == 1][0]).view(1, 1, *images.shape[1:])
    nonlens_img = torch.from_numpy(images[labels == 0][0]).view(1, 1, *images.shape[1:])
    noise_img = torch.rand(1, 1, *images.shape[1:])
    seeds = {"noise": noise_img, "real_lens": lens_img, "real_nonlens": nonlens_img}

    out_dir = ROOT / "outputs" / "deepdream_cnn"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Same regularization recipe as the ViT dreamer: without it, the CNN's
    # unconstrained multi-octave ascent saturates into near-binary black/white
    # blobs rather than interpretable texture.
    reg_kwargs = dict(tv_weight=0.02, grad_blur_sigma=0.6,
                       image_blur_every=10, image_blur_sigma=0.4)

    # 1) maximize the final lens-classification logit
    dreamer_head = DeepDreamer(model, layer_name="classifier.4", device=device)
    for name, seed in seeds.items():
        result = dreamer_head.dream(seed, channel=None, iterations=40, lr=0.08,
                                     octaves=3, octave_scale=1.4, jitter=4, **reg_kwargs)
        save_gray(result, out_dir / f"head_maximize_{name}.png")
        print(f"saved head_maximize_{name}.png")

    # 2) class-agnostic texture dream on the last conv block (64ch, before final pool)
    # index 21, not 15: models.py adds a BatchNorm2d after every Conv2d
    dreamer_block = DeepDreamer(model, layer_name="features.21", device=device)
    for name, seed in seeds.items():
        result = dreamer_block.dream(seed, channel=None, iterations=40, lr=0.08,
                                      octaves=3, octave_scale=1.4, jitter=4, **reg_kwargs)
        save_gray(result, out_dir / f"conv_texture_{name}.png")
        print(f"saved conv_texture_{name}.png")


if __name__ == "__main__":
    main()
