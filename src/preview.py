"""Save a grid of simulated single lenses vs. non-lenses for visual inspection."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def save_preview_grid(data_path=ROOT / "data" / "lens_dataset.npz",
                       out_path=ROOT / "outputs" / "sample_grid.png",
                       n_per_class: int = 4):
    data = np.load(data_path)
    images, labels = data["images"], data["labels"]

    lens_idx = np.where(labels == 1)[0][:n_per_class]
    nonlens_idx = np.where(labels == 0)[0][:n_per_class]

    fig, axes = plt.subplots(2, n_per_class, figsize=(3 * n_per_class, 6.5))
    for col, idx in enumerate(lens_idx):
        axes[0, col].imshow(images[idx], cmap="gray", origin="lower")
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
    for col, idx in enumerate(nonlens_idx):
        axes[1, col].imshow(images[idx], cmap="gray", origin="lower")
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])

    axes[0, 0].set_ylabel("Single lenses", fontsize=12)
    axes[1, 0].set_ylabel("Non-lenses", fontsize=12)
    fig.suptitle("Simulated single (galaxy-galaxy) lenses vs. non-lenses (lenstronomy, SIE+Sersic)")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    path = save_preview_grid()
    print(f"Saved preview grid to {path}")
