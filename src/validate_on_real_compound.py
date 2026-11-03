"""The real validation: run our compound-lens detector on actual archival
imaging of 3 known real compound lenses (SDSSJ0946+1006 'Jackpot', SL2SJ02176-0513,
J1148+1930 'Cosmic Horseshoe'), not just the synthetic two-arc proxy.

Preprocessing: each FITS cutout is resampled from its native HST pixel scale
to our pipeline's 0.1"/px via scipy.ndimage.zoom, then center-cropped to
100x100px (10"x10" FOV, matching every other image in this project), then
stretched with a 1st-99.5th percentile scale to [0,1] for this one-off
visual/detection check (NOT the fixed-scale-across-dataset rule used for
training data -- that rule exists to keep pixel values comparable across many
training images; a single one-off real-image validation check doesn't need
it, it just needs enough dynamic range to see the ring structure).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import zoom

from radial_profile import count_rings

ROOT = Path(__file__).resolve().parent.parent
TARGET_SCALE = 0.1  # arcsec/px, matches the rest of this project
TARGET_SIZE = 100

SYSTEMS = [
    ("SDSSJ0946+1006 (Jackpot Lens)", "data/real_compound_lenses/SDSSJ0946+1006/SDSSJ0946+1006_F814W.fits"),
    ("SL2SJ02176-0513", "data/real_compound_lenses/SL2SJ02176-0513/SL2SJ02176-0513_F814W.fits"),
    ("J1148+1930 (Cosmic Horseshoe)*", "data/real_compound_lenses/J1148+1930/J1148+1930_F814W.fits"),
]


def load_and_preprocess(fits_path: Path) -> np.ndarray:
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(np.float64)
        header = hdul[0].header
        w = WCS(header)
    native_scale = np.sqrt(np.sum(w.pixel_scale_matrix ** 2, axis=0)).mean() * 3600
    zoom_factor = native_scale / TARGET_SCALE

    resampled = zoom(np.nan_to_num(data), zoom_factor, order=1)
    h, w_ = resampled.shape
    cy, cx = h // 2, w_ // 2
    half = TARGET_SIZE // 2
    cropped = resampled[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
    # pad if the crop came up short (resampled image smaller than target)
    if cropped.shape != (TARGET_SIZE, TARGET_SIZE):
        padded = np.zeros((TARGET_SIZE, TARGET_SIZE))
        oy, ox = (TARGET_SIZE - cropped.shape[0]) // 2, (TARGET_SIZE - cropped.shape[1]) // 2
        padded[oy:oy + cropped.shape[0], ox:ox + cropped.shape[1]] = cropped
        cropped = padded

    lo, hi = np.percentile(cropped, [1.0, 99.5])
    return np.clip((cropped - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def main():
    fig, axes = plt.subplots(len(SYSTEMS), 2, figsize=(8, 2.8 * len(SYSTEMS)))

    print("=== Compound lens detector on REAL archival imaging ===\n")
    for i, (name, rel_path) in enumerate(SYSTEMS):
        img = load_and_preprocess(ROOT / rel_path)
        n_ring_like, classifications, radii, profile, residual = count_rings(img)

        print(f"{name}:")
        for c in classifications:
            print(f"  radius={c['radius']:.1f}px  coverage={c['angular_coverage']:.1%}  "
                  f"-> {c['classification']}" + ("  [near edge]" if c["near_edge"] else ""))
        verdict = "DETECTED compound (2+ ring-like peaks)" if n_ring_like >= 2 else \
                  f"only {n_ring_like} ring-like peak(s) found -- MISSED"
        print(f"  Verdict: {verdict}\n")

        axes[i, 0].imshow(img, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[i, 0].set_title(name, fontsize=9)
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])

        axes[i, 1].plot(radii, residual, color="0.3", lw=1)
        axes[i, 1].axhline(0, color="0.8", lw=0.8)
        for c in classifications:
            color = "crimson" if c["classification"] == "ring-like" else "darkorange"
            axes[i, 1].axvline(c["radius"], color=color, lw=1, ls="--")
        row_title = f"{n_ring_like} ring-like peak(s)" + (" <-- COMPOUND DETECTED" if n_ring_like >= 2 else "")
        axes[i, 1].set_title(row_title, fontsize=9, color="green" if n_ring_like >= 2 else "crimson")
        axes[i, 1].set_xlabel("radius (px)", fontsize=8)

    fig.suptitle("Compound lens detector tested on REAL known compound lenses (archival HST)",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = ROOT / "outputs" / "real_compound_lens_validation.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
