#!/usr/bin/env python3
"""Unravel DSPL rings into radial profiles and color profiles.

Reads the raw .npy flux arrays from generate_color_dspl_experiment.py (never
the normalized PNGs, which lose relative band flux) and produces:

- metrics_per_image.csv: radial-profile and color metrics per scene, designed
  to separate simple vs compound lens structures without any model scores.
- summary_by_condition.csv: mean metrics per experiment/condition/structure.
- Plots: example ring profiles, sweep responses, color profiles, and a
  morphology scatter showing simple-vs-compound separability.

All metrics are morphology descriptions, not model detections or accuracy.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


BANDS = ("g", "r", "i")
PIXELS = 100
PIXEL_SCALE = 0.1
ANNULUS_WIDTH_PIX = 2
RADIAL_BINS = np.arange(0.0, 51.0, ANNULUS_WIDTH_PIX)  # bin edges in pixels
BIN_CENTERS_PIX = 0.5 * (RADIAL_BINS[:-1] + RADIAL_BINS[1:])
BIN_CENTERS_ARCSEC = BIN_CENTERS_PIX * PIXEL_SCALE

_Y, _X = np.indices((PIXELS, PIXELS))
CENTER = (PIXELS - 1) / 2
RADIUS_MAP = np.hypot(_Y - CENTER, _X - CENTER)
ANNULUS_INDEX = np.digitize(RADIUS_MAP, RADIAL_BINS) - 1  # 0..24 inside, else out of range


def radial_profile(image: np.ndarray) -> np.ndarray:
    """Mean flux per annulus around the lens center (x = radius, y = flux)."""
    profile = np.full(len(BIN_CENTERS_PIX), np.nan)
    for annulus in range(len(BIN_CENTERS_PIX)):
        mask = ANNULUS_INDEX == annulus
        if mask.any():
            profile[annulus] = float(image[mask].mean())
    return profile


def annulus_integrated_flux(image: np.ndarray, inner: int, outer: int) -> float:
    mask = (RADIUS_MAP >= inner) & (RADIUS_MAP < outer)
    return float(image[mask].sum())


def color_magnitude(flux_a: float, flux_b: float) -> float:
    """a - b color from integrated fluxes; NaN if either side is non-positive."""
    if flux_a <= 0 or flux_b <= 0:
        return math.nan
    return -2.5 * math.log10(flux_a / flux_b)


def count_local_maxima(profile: np.ndarray, min_bin: int = 3) -> int:
    """Count radial-profile peaks outside the central galaxy region."""
    valid = profile[min_bin:]
    valid = valid[~np.isnan(valid)]
    if len(valid) < 3:
        return 0
    peaks = sum(
        1
        for index in range(1, len(valid) - 1)
        if valid[index] > valid[index - 1] and valid[index] > valid[index + 1]
    )
    return peaks


def quadrant_asymmetry(image: np.ndarray, inner: float = 6.0, outer: float = 22.0) -> float:
    """(max - min) / (max + min) of ring flux across four quadrants."""
    ring = (RADIUS_MAP >= inner) & (RADIUS_MAP < outer)
    quadrants = [
        ring & (_X >= CENTER) & (_Y >= CENTER),
        ring & (_X >= CENTER) & (_Y < CENTER),
        ring & (_X < CENTER) & (_Y >= CENTER),
        ring & (_X < CENTER) & (_Y < CENTER),
    ]
    fluxes = [float(image[q].sum()) for q in quadrants]
    total = max(fluxes) + min(fluxes)
    return (max(fluxes) - min(fluxes)) / total if total > 0 else math.nan


def image_metrics(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    profiles = {band: radial_profile(arrays[band]) for band in BANDS}
    r_profile = profiles["r"]

    inner_flux = annulus_integrated_flux(arrays["r"], 0.0, 6.0)
    outer_flux = annulus_integrated_flux(arrays["r"], 6.0, 22.0)
    outer_profile = r_profile[3:11]  # 6-22 arcsec-ish annuli
    peak_bin = int(np.nanargmax(outer_profile)) + 3
    ring_contrast = (
        float(np.nanmax(outer_profile)) / float(np.nanmean(r_profile[:3]))
        if np.nanmean(r_profile[:3]) > 0
        else math.nan
    )

    # Color profile: g-r in inner (lens-dominated) vs outer (arc-dominated) annulus.
    inner_color = color_magnitude(
        annulus_integrated_flux(arrays["g"], 0.0, 6.0),
        annulus_integrated_flux(arrays["r"], 0.0, 6.0),
    )
    outer_color = color_magnitude(
        annulus_integrated_flux(arrays["g"], 6.0, 22.0),
        annulus_integrated_flux(arrays["r"], 6.0, 22.0),
    )
    color_gradient = (
        outer_color - inner_color
        if not (math.isnan(inner_color) or math.isnan(outer_color))
        else math.nan
    )

    return {
        "peak_radius_pix": float(BIN_CENTERS_PIX[peak_bin]),
        "peak_radius_arcsec": float(BIN_CENTERS_ARCSEC[peak_bin]),
        "ring_contrast": ring_contrast,
        "outer_to_inner_flux_ratio": outer_flux / inner_flux if inner_flux > 0 else math.nan,
        "quadrant_asymmetry": quadrant_asymmetry(arrays["r"]),
        "radial_peak_count": float(count_local_maxima(r_profile)),
        "inner_g_minus_r": inner_color,
        "outer_g_minus_r": outer_color,
        "color_gradient_g_minus_r": color_gradient,
        "radial_profile_g": ";".join(f"{v:.6g}" for v in profiles["g"]),
        "radial_profile_r": ";".join(f"{v:.6g}" for v in profiles["r"]),
        "radial_profile_i": ";".join(f"{v:.6g}" for v in profiles["i"]),
    }


def save_image_profile_plot(arrays: dict[str, np.ndarray], title: str, path: Path) -> None:
    """Per-image unraveled ring: g/r/i annulus profiles plus the g-r color profile."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profiles = {band: radial_profile(arrays[band]) for band in BANDS}
    valid = (profiles["g"] > 0) & (profiles["r"] > 0)
    color_profile = np.where(
        valid, -2.5 * np.log10(np.abs(profiles["g"] / profiles["r"])), np.nan
    )
    color_profile[~valid] = np.nan

    fig, axes = plt.subplots(
        2, 1, figsize=(6.5, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    # Band colors match the RGB composites: g blue, r green, i red.
    for band, color in (("g", "#3B82C4"), ("r", "#55A868"), ("i", "#C44E52")):
        axes[0].plot(BIN_CENTERS_ARCSEC, profiles[band], color=color, label=f"{band} band")
    axes[0].set_ylabel("mean flux per annulus")
    axes[0].set_title(title, fontsize=9)
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(BIN_CENTERS_ARCSEC, color_profile, color="#8172B3")
    axes[1].set_xlabel("radius from lens center (arcsec)")
    axes[1].set_ylabel("g - r color")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def parse_profile(row: dict[str, str], band: str) -> np.ndarray:
    return np.array([float(v) for v in row[f"radial_profile_{band}"].split(";")])


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["experiment"], row["condition_value"], row["lens_structure"])].append(row)

    metric_names = [
        "ring_contrast",
        "outer_to_inner_flux_ratio",
        "quadrant_asymmetry",
        "peak_radius_arcsec",
        "color_gradient_g_minus_r",
    ]
    summaries = []
    for (experiment, condition_value, structure), group in sorted(groups.items()):
        entry: dict[str, object] = {
            "experiment": experiment,
            "condition_value": condition_value,
            "condition_unit": group[0]["condition_unit"],
            "lens_structure": structure,
            "label": group[0]["label"],
            "n_images": len(group),
        }
        for metric in metric_names:
            values = [
                float(row[metric])
                for row in group
                if row[metric] not in ("", "nan") and math.isfinite(float(row[metric]))
            ]
            entry[f"{metric}_mean"] = statistics.mean(values) if values else math.nan
            entry[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else math.nan
        summaries.append(entry)
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(rows: list[dict[str, str]], output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    positives = [row for row in rows if row["label"] == "1"]

    def control_rows(structure: str) -> list[dict[str, str]]:
        return [
            row
            for row in positives
            if row["lens_structure"] == structure
            and (
                (row["experiment"] == "faintness" and float(row["condition_value"]) == 0.0)
                or (row["experiment"] == "arc_size" and float(row["condition_value"]) == 1.0)
                or (row["experiment"] == "blending" and float(row["condition_value"]) == 1.0)
            )
        ]

    # 1. Example radial profiles: one simple and one compound control, all bands.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, structure in zip(axes, ("simple", "compound")):
        example = control_rows(structure)[0]
        for band, color in zip(BANDS, ("#3B82C4", "#C44E52", "#55A868")):
            axis.plot(
                BIN_CENTERS_ARCSEC,
                parse_profile(example, band),
                color=color,
                label=f"{band} band",
            )
        axis.set_title(f"{structure} lens, control condition")
        axis.set_xlabel("radius from lens center (arcsec)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("mean flux per annulus")
    axes[0].legend()
    fig.suptitle("Unraveled ring profiles (annulus mean flux vs radius)")
    fig.tight_layout()
    path = output_dir / "fig1_example_radial_profiles.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)

    # 2. Sweep response: mean ring contrast per condition, split by structure.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    sweep_titles = {"faintness": "Faintness (mag offset)", "arc_size": "Arc size (theta-E scale)", "blending": "Blending (lens-light flux)"}
    for axis, (experiment, title) in zip(axes, sweep_titles.items()):
        for structure, color in (("simple", "#007C91"), ("compound", "#D55E00")):
            condition_rows = [
                row for row in positives if row["experiment"] == experiment and row["lens_structure"] == structure
            ]
            by_condition: dict[float, list[float]] = defaultdict(list)
            for row in condition_rows:
                by_condition[float(row["condition_value"])].append(float(row["ring_contrast"]))
            x_values = sorted(by_condition)
            means = [statistics.mean(by_condition[x]) for x in x_values]
            stds = [statistics.stdev(by_condition[x]) if len(by_condition[x]) > 1 else 0.0 for x in x_values]
            axis.errorbar(x_values, means, yerr=stds, marker="o", capsize=3, label=structure, color=color)
        axis.set_title(title)
        axis.set_xlabel(condition_rows[0]["condition_unit"] if condition_rows else "condition")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("ring contrast (morphology metric)")
    axes[0].legend()
    fig.suptitle("Ring contrast vs experimental condition")
    fig.tight_layout()
    path = output_dir / "fig2_ring_contrast_by_condition.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)

    # 3. Color profiles: mean g-r vs radius, simple vs compound (control only).
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    for structure, color in (("simple", "#007C91"), ("compound", "#D55E00")):
        g_profiles = np.stack([parse_profile(row, "g") for row in control_rows(structure)])
        r_profiles = np.stack([parse_profile(row, "r") for row in control_rows(structure)])
        # Skip annuli where noise makes mean flux non-positive; color is undefined there.
        valid = (g_profiles > 0) & (r_profiles > 0)
        color_profiles = np.where(valid, -2.5 * np.log10(np.abs(g_profiles / r_profiles)), np.nan)
        color_profiles[~valid] = np.nan
        with np.errstate(invalid="ignore"):
            mean_color = np.nanmean(color_profiles, axis=0)
            std_color = np.nanstd(color_profiles, axis=0)
        axis.plot(BIN_CENTERS_ARCSEC, mean_color, color=color, label=structure)
        axis.fill_between(BIN_CENTERS_ARCSEC, mean_color - std_color, mean_color + std_color, color=color, alpha=0.2)
    axis.set_xlabel("radius from lens center (arcsec)")
    axis.set_ylabel("g - r color of annulus")
    axis.set_title("Color profile vs radius (control condition)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path = output_dir / "fig3_color_profile_by_structure.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)

    # 4. Morphology scatter: asymmetry vs ring contrast, colored by structure.
    fig, axis = plt.subplots(figsize=(6.5, 5))
    for structure, color in (("simple", "#007C91"), ("compound", "#D55E00")):
        group = [row for row in positives if row["lens_structure"] == structure]
        axis.scatter(
            [float(row["ring_contrast"]) for row in group],
            [float(row["quadrant_asymmetry"]) for row in group],
            s=18,
            alpha=0.7,
            color=color,
            label=structure,
        )
    negatives = [row for row in rows if row["label"] == "0"]
    axis.scatter(
        [float(row["ring_contrast"]) for row in negatives],
        [float(row["quadrant_asymmetry"]) for row in negatives],
        s=28,
        marker="x",
        color="0.35",
        label="negative control",
    )
    axis.set_xlabel("ring contrast")
    axis.set_ylabel("quadrant asymmetry of ring flux")
    axis.set_title("Simple vs compound morphology (positive images)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path = output_dir / "fig4_structure_metric_scatter.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)

    # 5-6. Class comparison on the control condition only: non-lens, simple lens,
    # compound lens, and all lenses pooled. Compare like with like.
    def is_control_condition(row: dict[str, str]) -> bool:
        return (
            (row["experiment"] == "faintness" and float(row["condition_value"]) == 0.0)
            or (row["experiment"] == "arc_size" and float(row["condition_value"]) == 1.0)
            or (row["experiment"] == "blending" and float(row["condition_value"]) == 1.0)
        )

    simple_controls = [
        row for row in positives if row["lens_structure"] == "simple" and is_control_condition(row)
    ]
    compound_controls = [
        row for row in positives if row["lens_structure"] == "compound" and is_control_condition(row)
    ]
    class_groups = [
        ("non-lens", negatives, "0.35"),
        ("simple lens", simple_controls, "#007C91"),
        ("compound lens", compound_controls, "#D55E00"),
        ("all lenses", simple_controls + compound_controls, "#8172B3"),
    ]

    # 5. Mean unraveled-ring profile per band (rows) and class (columns).
    fig, axes = plt.subplots(len(BANDS), len(class_groups), figsize=(15, 8.5), sharex=True)
    for col, (class_name, group, color) in enumerate(class_groups):
        for band_row, band in enumerate(BANDS):
            axis = axes[band_row, col]
            band_profiles = np.stack([parse_profile(row, band) for row in group])
            with np.errstate(invalid="ignore"):
                mean_profile = np.nanmean(band_profiles, axis=0)
                std_profile = np.nanstd(band_profiles, axis=0)
            axis.plot(BIN_CENTERS_ARCSEC, mean_profile, color=color)
            axis.fill_between(
                BIN_CENTERS_ARCSEC,
                mean_profile - std_profile,
                mean_profile + std_profile,
                color=color,
                alpha=0.2,
            )
            axis.grid(alpha=0.25)
            if band_row == 0:
                axis.set_title(f"{class_name} (n={len(group)})")
            if col == 0:
                axis.set_ylabel(f"{band} band mean flux")
            if band_row == len(BANDS) - 1:
                axis.set_xlabel("radius from lens center (arcsec)")
    fig.suptitle("Unraveled ring profiles by class, control condition (mean +/- 1 std)")
    fig.tight_layout()
    path = output_dir / "fig5_class_radial_profiles.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)

    # 6. Color variation by class: mean g-r vs radius for each class.
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for class_name, group, color in class_groups:
        g_profiles = np.stack([parse_profile(row, "g") for row in group])
        r_profiles = np.stack([parse_profile(row, "r") for row in group])
        valid = (g_profiles > 0) & (r_profiles > 0)
        color_profiles = np.where(valid, -2.5 * np.log10(np.abs(g_profiles / r_profiles)), np.nan)
        color_profiles[~valid] = np.nan
        with np.errstate(invalid="ignore"):
            mean_color = np.nanmean(color_profiles, axis=0)
            std_color = np.nanstd(color_profiles, axis=0)
        axis.plot(BIN_CENTERS_ARCSEC, mean_color, color=color, label=f"{class_name} (n={len(group)})")
        axis.fill_between(
            BIN_CENTERS_ARCSEC, mean_color - std_color, mean_color + std_color, color=color, alpha=0.15
        )
    axis.set_xlabel("radius from lens center (arcsec)")
    axis.set_ylabel("g - r color of annulus")
    axis.set_title("Color profile by class, control condition (mean +/- 1 std)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path = output_dir / "fig6_class_color_profiles.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("dspl_color_experiment/simulation_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("dspl_color_experiment/analysis"))
    parser.add_argument(
        "--skip-per-image-plots",
        action="store_true",
        help="skip the per-image ring-profile plots under <output-dir>/profiles/",
    )
    args = parser.parse_args()

    if not args.manifest.is_file():
        parser.error(f"Manifest not found: {args.manifest}. Run generate_color_dspl_experiment.py first.")

    manifest_root = args.manifest.resolve().parent
    rows = load_manifest(args.manifest)
    metrics_rows: list[dict[str, object]] = []
    for row in rows:
        arrays = {}
        for band in BANDS:
            array_path = (manifest_root / row[f"array_path_{band}"]).resolve()
            if not array_path.is_file():
                raise FileNotFoundError(f"Raw flux array missing: {array_path}")
            arrays[band] = np.load(array_path)
        metrics = image_metrics(arrays)
        if not args.skip_per_image_plots:
            kind = "non-lens control" if row["label"] == "0" else f"{row['lens_structure']} lens"
            save_image_profile_plot(
                arrays,
                f"{row['image_id']} ({kind}, {row['experiment']}={row['condition_value']})",
                args.output_dir
                / "profiles"
                / row["experiment"]
                / row["lens_structure"]
                / f"{row['image_id']}_profile.png",
            )
        metrics_rows.append(
            {
                "image_id": row["image_id"],
                "label": row["label"],
                "experiment": row["experiment"],
                "condition_value": row["condition_value"],
                "condition_unit": row["condition_unit"],
                "lens_structure": row["lens_structure"],
                "seed": row["seed"],
                **metrics,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics_per_image.csv"
    write_csv(metrics_path, metrics_rows)
    summaries = summarize(metrics_rows)
    summary_path = args.output_dir / "summary_by_condition.csv"
    write_csv(summary_path, summaries)
    figures = make_plots(metrics_rows, args.output_dir)

    print(f"Wrote per-image metrics: {metrics_path}")
    print(f"Wrote condition summary: {summary_path}")
    profile_dir = args.output_dir / "profiles"
    if profile_dir.is_dir():
        count = len(list(profile_dir.rglob("*_profile.png")))
        print(f"Wrote {count} per-image ring-profile plots under {profile_dir}")
    for figure in figures:
        print(f"Wrote plot: {figure}")
    print("Metrics are morphology descriptions from raw flux arrays, not model scores or accuracy.")


if __name__ == "__main__":
    main()
