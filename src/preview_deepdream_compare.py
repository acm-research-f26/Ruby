"""Side-by-side grid: CNN vs. ViT vs. Zoobot DeepDream, same seeds/targets."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from train_cnn import load_dataset

ROOT = Path(__file__).resolve().parent.parent


def load_img(path):
    return np.flipud(np.array(Image.open(path)))


def main():
    images, labels = load_dataset(ROOT / "data" / "lens_dataset.npz")
    lens_img = images[labels == 1][0]
    nonlens_img = images[labels == 0][0]
    rng = np.random.default_rng(0)
    noise_img = rng.random(images.shape[1:], dtype=np.float32)
    seeds = {"noise": noise_img, "real_lens": lens_img, "real_nonlens": nonlens_img}

    cnn_dir = ROOT / "outputs" / "deepdream_cnn"
    vit_dir = ROOT / "outputs" / "deepdream_vit"
    zoobot_dir = ROOT / "outputs" / "deepdream_zoobot"

    rows = [
        ("seed image", None, None),
        ("CNN: maximize logit", cnn_dir, "head_maximize_{name}.png"),
        ("CNN: conv-layer texture", cnn_dir, "conv_texture_{name}.png"),
        ("ViT: maximize logit", vit_dir, "head_maximize_{name}.png"),
        ("ViT: block2 texture", vit_dir, "block2_texture_{name}.png"),
        ("Zoobot: maximize logit", zoobot_dir, "head_maximize_{name}.png"),
        ("Zoobot: stage2 texture", zoobot_dir, "stage2_texture_{name}.png"),
    ]

    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 21.5))
    for col, (name, seed) in enumerate(seeds.items()):
        axes[0, col].set_title(name, fontsize=10)
        for row, (label, d, pattern) in enumerate(rows):
            ax = axes[row, col]
            if d is None:
                ax.imshow(seed, cmap="gray", origin="lower", vmin=0, vmax=1)
            else:
                ax.imshow(load_img(d / pattern.format(name=name)), cmap="gray", origin="lower")
            ax.set_xticks([]); ax.set_yticks([])

    for row, (label, _, _) in enumerate(rows):
        axes[row, 0].set_ylabel(label, fontsize=10)

    fig.suptitle("DeepDream comparison: CNN vs. ViT vs. Zoobot lens classifier\n(all trained on 6000 simulated images, regularized dreams)")
    fig.tight_layout()
    out_path = ROOT / "outputs" / "deepdream_compare_grid.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
