"""Turn the downloaded real Euclid Q1 FITS cutouts into a dataset matching our
simulated images' format: 100x100 px (center-cropped from the 200x200 FITS,
which keeps the same 0.1"/px scale -> same 10"x10" field of view as the sims),
single fixed linear intensity stretch applied identically across the whole
real set (not per-image -- same reasoning as lens_sim.fixed_linear_scale, but
calibrated separately since real Euclid flux units have nothing to do with
our simulated ADU counts)."""
from __future__ import annotations

import csv
import numpy as np
from pathlib import Path
from astropy.io import fits
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
REAL_DIR = ROOT / "data" / "real_euclid"


def center_crop(img: np.ndarray, size: int = 100) -> np.ndarray:
    h, w = img.shape
    top = (h - size) // 2
    left = (w - size) // 2
    return img[top:top + size, left:left + size]


def load_metadata():
    rows = []
    with open(REAL_DIR / "metadata.csv") as f:
        for row in csv.DictReader(f):
            if row["filename"].endswith(".fits"):
                rows.append(row)
    return rows


def main(lo_pct: float = 1.0, hi_pct: float = 99.5):
    rows = load_metadata()

    raw_crops, labels, names, ras, decs = [], [], [], [], []
    for row in rows:
        label_dir = "lens" if row["label"] == "lens" else "nonlens"
        path = REAL_DIR / label_dir / row["filename"]
        data = fits.getdata(path).astype(np.float32)
        data = np.nan_to_num(data, nan=0.0)
        crop = center_crop(data, 100)
        raw_crops.append(crop)
        labels.append(1 if row["label"] == "lens" else 0)
        names.append(row["filename"])
        ras.append(float(row["ra"]))
        decs.append(float(row["dec"]))

    raw_crops = np.stack(raw_crops)
    labels = np.array(labels, dtype=np.int64)

    # ONE fixed stretch computed across the whole real set, applied to every
    # image identically -- analogous to lens_sim.calibrate_global_scale, but
    # a separate calibration since real flux units != simulated ADU counts.
    lo, hi = np.percentile(raw_crops, [lo_pct, hi_pct])
    print(f"Real-data fixed scale: low={lo:.5f} high={hi:.5f} (from {lo_pct}/{hi_pct} percentiles)")
    images = np.clip((raw_crops - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    out_path = ROOT / "data" / "real_euclid_dataset.npz"
    np.savez_compressed(out_path, images=images, labels=labels,
                         filenames=np.array(names), ra=np.array(ras), dec=np.array(decs))
    print(f"Saved {len(images)} real images ({labels.sum()} lens, {(labels == 0).sum()} nonlens) to {out_path}")

    preview_dir = ROOT / "outputs" / "real_euclid_preview"
    (preview_dir / "lens").mkdir(parents=True, exist_ok=True)
    (preview_dir / "nonlens").mkdir(parents=True, exist_ok=True)
    for img, label, name in zip(images, labels, names):
        sub = "lens" if label == 1 else "nonlens"
        arr = np.flipud(img)
        Image.fromarray((arr * 255).astype(np.uint8), mode="L").save(
            preview_dir / sub / (Path(name).stem + ".png"))
    print(f"Saved grayscale previews to {preview_dir}")


if __name__ == "__main__":
    main()
