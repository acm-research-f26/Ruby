"""Grid of the preprocessed real Euclid Q1 cutouts (lens candidates vs. non-lenses)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    data = np.load(ROOT / "data" / "real_euclid_dataset.npz")
    images, labels = data["images"], data["labels"]

    lens_idx = np.where(labels == 1)[0]
    nonlens_idx = np.where(labels == 0)[0]
    n = min(len(lens_idx), len(nonlens_idx), 6)

    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 5))
    for col in range(n):
        axes[0, col].imshow(images[lens_idx[col]], cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[1, col].imshow(images[nonlens_idx[col]], cmap="gray", origin="lower", vmin=0, vmax=1)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("real lens candidates", fontsize=10)
    axes[1, 0].set_ylabel("real non-lenses", fontsize=10)
    fig.suptitle("Real Euclid Q1 cutouts (Walmsley et al. 2025 discovery engine)")
    fig.tight_layout()
    out_path = ROOT / "outputs" / "real_euclid_grid.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
