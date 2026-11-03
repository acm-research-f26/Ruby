"""Grid of images + their radial profile/residual, with detected peak count,
for the 12 real lens images and a sample of simulated single lenses (ground
truth: exactly 1 ring, so this is the false-positive-rate check)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from radial_profile import count_rings

ROOT = Path(__file__).resolve().parent.parent


def build_grid(images, names, out_path, title):
    n = len(images)
    fig, axes = plt.subplots(n, 2, figsize=(8, 2.6 * n))
    if n == 1:
        axes = axes[None, :]

    n_found_list = []
    for i, (img, name) in enumerate(zip(images, names)):
        n_ring_like, classifications, radii, profile, residual = count_rings(img)
        n_found_list.append(n_ring_like)

        axes[i, 0].imshow(img, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[i, 0].set_title(name, fontsize=9)
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])

        axes[i, 1].plot(radii, residual, color="0.3", lw=1)
        axes[i, 1].axhline(0, color="0.8", lw=0.8)
        for c in classifications:
            is_ring = c["classification"] == "ring-like"
            color = "crimson" if is_ring else "darkorange"
            axes[i, 1].axvline(c["radius"], color=color, lw=1, ls="--")
            label = f"{c['angular_coverage']:.0%}" + (" edge!" if c["near_edge"] else "")
            axes[i, 1].text(c["radius"], axes[i, 1].get_ylim()[1] * 0.85,
                             label, fontsize=7, color=color, ha="center")
        row_title = f"{n_ring_like} ring-like peak(s)" + (" <-- CHECK" if n_ring_like >= 2 else "")
        axes[i, 1].set_title(row_title, fontsize=9, color="crimson" if n_ring_like >= 2 else "black")
        axes[i, 1].set_xlabel("radius (px)  [red=ring-like, orange=point-like/companion]", fontsize=7)

    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved {out_path}")
    return n_found_list


def main():
    real = np.load(ROOT / "data" / "real_euclid_dataset.npz")
    lens_imgs = real["images"][real["labels"] == 1]
    lens_names = [f"real lens {i}" for i in range(len(lens_imgs))]
    n_found_real = build_grid(lens_imgs, lens_names,
                               ROOT / "outputs" / "radial_profile_real_lenses.png",
                               "Radial profile peak count -- 12 real verified single lenses")
    print(f"Real lenses: peak counts = {n_found_real}")
    print(f"  -> {sum(1 for n in n_found_real if n >= 2)}/{len(n_found_real)} flagged as possibly compound (>=2 peaks)")

    sim = np.load(ROOT / "data" / "lens_dataset_large.npz")
    sim_lens_imgs = sim["images"][sim["labels"] == 1][:12]
    sim_names = [f"sim single lens {i}" for i in range(len(sim_lens_imgs))]
    n_found_sim = build_grid(sim_lens_imgs, sim_names,
                              ROOT / "outputs" / "radial_profile_sim_lenses.png",
                              "Radial profile peak count -- 12 simulated single lenses (ground truth: 1 ring)")
    print(f"Simulated single lenses: peak counts = {n_found_sim}")
    print(f"  -> {sum(1 for n in n_found_sim if n >= 2)}/{len(n_found_sim)} FALSE POSITIVES (>=2 peaks found, but only 1 exists)")


if __name__ == "__main__":
    main()
