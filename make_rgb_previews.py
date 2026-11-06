"""Build RGB color-composite previews from the multi-band DSPL arrays.

The per-band PNGs are normalized image-by-image, which destroys relative
band flux and therefore color. This script rebuilds composites from the raw
.npy arrays with a single asinh stretch shared across all three bands of a
scene, so band ratios (i.e. colors) survive.

Channel mapping follows the usual astronomical convention:
red channel = i band, green channel = r band, blue channel = g band.
Redder objects therefore appear redder in the composites.

Outputs (not tracked in Git):
- dspl_color_experiment/rgb/{experiment}/{structure}/{image_id}_rgb.png
- dspl_color_experiment/analysis/fig0_rgb_gallery.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

BANDS = ("g", "r", "i")
CHANNEL_FOR_BAND = {"i": 0, "r": 1, "g": 2}  # RGB channel index per band


def load_rgb(row: pd.Series, output_root: Path, stretch: float = 3.0) -> np.ndarray:
    """Return a float RGB array in [0, 1] with one stretch shared across bands."""
    bands = {b: np.load(output_root / row[f"array_path_{b}"]) for b in BANDS}
    scale = float(np.percentile(np.stack([bands[b] for b in BANDS]), 99.5))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    rgb = np.zeros((*bands["g"].shape, 3), dtype=float)
    for band, channel in CHANNEL_FOR_BAND.items():
        rgb[..., channel] = np.arcsinh(stretch * bands[band] / scale) / np.arcsinh(stretch)
    return np.clip(rgb, 0.0, 1.0)


def save_png(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB").save(path)


def make_gallery(df: pd.DataFrame, output_root: Path, analysis_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = df[(df["experiment"] == "faintness") & (df["replicate"] == 0)]
    conditions = sorted(picks["condition_value"].unique())
    fig, axes = plt.subplots(len(conditions), 2, figsize=(6, 3 * len(conditions)))
    for row_index, value in enumerate(conditions):
        for col_index, structure in enumerate(("simple", "compound")):
            match = picks[
                (picks["condition_value"] == value) & (picks["lens_structure"] == structure)
            ].iloc[0]
            ax = axes[row_index, col_index]
            ax.imshow(load_rgb(match, output_root), origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            label = "control" if value == 0 else f"+{value:g} mag fainter"
            ax.set_title(f"{structure}, {label}", fontsize=9)
    fig.suptitle("RGB composites (i->R, r->G, g->B), shared asinh stretch")
    fig.tight_layout()
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out = analysis_dir / "fig0_rgb_gallery.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="dspl_color_experiment/simulation_manifest.csv")
    parser.add_argument("--output-dir", default=None, help="default: <manifest folder>/rgb")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N rows")
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    output_root = manifest.parent
    rgb_root = Path(args.output_dir) if args.output_dir else output_root / "rgb"

    df = pd.read_csv(manifest)
    todo = df.head(args.limit) if args.limit else df
    for _, row in todo.iterrows():
        rgb = load_rgb(row, output_root)
        path = rgb_root / row["experiment"] / row["lens_structure"] / f"{row['image_id']}_rgb.png"
        save_png(rgb, path)
    print(f"Wrote {len(todo)} RGB composites under {rgb_root}")

    make_gallery(df, output_root, output_root / "analysis")


if __name__ == "__main__":
    main()
