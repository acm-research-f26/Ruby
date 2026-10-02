"""Dump simulated images from the .npz dataset as individual grayscale PNG files
for browsing."""
from __future__ import annotations

import numpy as np
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _to_grayscale_png(image: np.ndarray) -> Image.Image:
    # images are already scaled to [0, 1]; flip vertically since arrays are
    # stored with row 0 at the bottom (origin="lower" convention used elsewhere).
    arr = np.flipud(image)
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8), mode="L")


def export_images(data_path=ROOT / "data" / "lens_dataset.npz",
                   out_dir=ROOT / "outputs" / "images",
                   n_per_class: int | None = None):
    data = np.load(data_path)
    images, labels = data["images"], data["labels"]

    lens_dir = Path(out_dir) / "lens"
    nonlens_dir = Path(out_dir) / "nonlens"
    lens_dir.mkdir(parents=True, exist_ok=True)
    nonlens_dir.mkdir(parents=True, exist_ok=True)

    lens_idx = np.where(labels == 1)[0]
    nonlens_idx = np.where(labels == 0)[0]
    if n_per_class is not None:
        lens_idx = lens_idx[:n_per_class]
        nonlens_idx = nonlens_idx[:n_per_class]

    for i in lens_idx:
        _to_grayscale_png(images[i]).save(lens_dir / f"lens_{i:04d}.png")
    for i in nonlens_idx:
        _to_grayscale_png(images[i]).save(nonlens_dir / f"nonlens_{i:04d}.png")

    return lens_dir, nonlens_dir, len(lens_idx), len(nonlens_idx)


if __name__ == "__main__":
    lens_dir, nonlens_dir, n_lens, n_nonlens = export_images()
    print(f"Wrote {n_lens} lens images to {lens_dir}")
    print(f"Wrote {n_nonlens} non-lens images to {nonlens_dir}")
