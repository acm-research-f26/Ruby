"""Grid comparing original seeds to their ViT DeepDream results."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from train_cnn import load_dataset

ROOT = Path(__file__).resolve().parent.parent


def main():
    images, labels = load_dataset(ROOT / "data" / "lens_dataset.npz")
    lens_img = images[labels == 1][0]
    nonlens_img = images[labels == 0][0]
    rng = np.random.default_rng(0)
    noise_img = rng.random(images.shape[1:], dtype=np.float32)

    seeds = {"noise": noise_img, "real_lens": lens_img, "real_nonlens": nonlens_img}
    dd_dir = ROOT / "outputs" / "deepdream_vit"

    fig, axes = plt.subplots(3, 3, figsize=(9, 9.5))
    row_labels = ["seed image", "head-maximize dream", "block2 texture dream"]
    for col, (name, seed) in enumerate(seeds.items()):
        axes[0, col].imshow(seed, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[0, col].set_title(name, fontsize=10)

        head_img = np.array(Image.open(dd_dir / f"head_maximize_{name}.png"))
        axes[1, col].imshow(np.flipud(head_img), cmap="gray", origin="lower")

        block_img = np.array(Image.open(dd_dir / f"block2_texture_{name}.png"))
        axes[2, col].imshow(np.flipud(block_img), cmap="gray", origin="lower")

    for row in range(3):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("DeepDream on the small ViT lens classifier")
    fig.tight_layout()
    out_path = ROOT / "outputs" / "deepdream_vit_grid.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
