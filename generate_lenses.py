#!/usr/bin/env python3
"""
This script creates a CSV file with:
- 1,000 lens samples
- 1,000 non-lens samples
- 500 compound lenses among the 1,000 lens samples
- 500 simple lenses among the 1,000 lens samples
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from lenstronomy.SimulationAPI.sim_api import SimAPI
from PIL import Image


FEATURE_NAMES = [
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
    "feature_5",
    "feature_6",
]


def make_lens_features(compound: bool, rng: random.Random) -> dict[str, float]:
    """Create a synthetic feature vector for a lens image.

    Compound lenses get a stronger secondary signal to mimic multiple image arcs.
    """
    base = {
        "feature_1": rng.uniform(0.7, 1.4),
        "feature_2": rng.uniform(0.8, 1.5),
        "feature_3": rng.uniform(0.6, 1.3),
        "feature_4": rng.uniform(0.7, 1.4),
        "feature_5": rng.uniform(0.8, 1.6),
        "feature_6": rng.uniform(0.9, 1.7),
    }

    if compound:
        # Compound lenses are made more structured and stronger in multiple features.
        compound_shift = {
            "feature_1": rng.uniform(0.35, 0.8),
            "feature_2": rng.uniform(0.45, 0.9),
            "feature_3": rng.uniform(0.25, 0.7),
            "feature_4": rng.uniform(0.30, 0.75),
            "feature_5": rng.uniform(0.50, 1.0),
            "feature_6": rng.uniform(0.60, 1.05),
        }
        for key in base:
            base[key] += compound_shift[key]
    else:
        # Simpler lenses are still lens-like, but less extreme than compound systems.
        for key in base:
            base[key] += rng.uniform(-0.15, 0.15)

    return base


def make_non_lens_features(rng: random.Random) -> dict[str, float]:
    """Create a synthetic feature vector for a non-lens image."""
    return {
        "feature_1": rng.uniform(-1.0, 1.0),
        "feature_2": rng.uniform(-1.0, 1.0),
        "feature_3": rng.uniform(-1.0, 1.0),
        "feature_4": rng.uniform(-1.0, 1.0),
        "feature_5": rng.uniform(-1.0, 1.0),
        "feature_6": rng.uniform(-1.0, 1.0),
    }


def make_simulator(compound: bool = False) -> SimAPI:
    """Create the 64x64 Euclid-like simulation used by the notebook."""
    single_band = {
        "pixel_scale": 0.1,
        "exposure_time": 90,
        "magnitude_zero_point": 27.0,
        "read_noise": 7,
        "ccd_gain": 6.083,
        "sky_brightness": 23.5,
        "seeing": 0.16,
        "num_exposures": 1,
        "psf_type": "GAUSSIAN",
    }
    lens_count = 2 if compound else 1
    model = {
        "lens_model_list": ["SIE"] * lens_count,
        "source_light_model_list": ["SERSIC_ELLIPSE"],
        "lens_light_model_list": ["SERSIC_ELLIPSE"] * lens_count,
    }
    return SimAPI(64, single_band, model)


def make_lens_image(seed: int, compound: bool) -> tuple[np.ndarray, float]:
    """Simulate a lens and return the noisy image and source redshift."""
    np.random.seed(seed)
    sim = make_simulator(compound)
    source_redshift = np.random.uniform(0.3, 5.0)

    if compound:
        centers = [
            (np.random.uniform(-0.35, -0.15), np.random.uniform(-0.25, 0.25)),
            (np.random.uniform(0.15, 0.35), np.random.uniform(-0.25, 0.25)),
        ]
    else:
        centers = [(0.0, 0.0)]

    kwargs_lens = []
    kwargs_lens_light_mag = []
    for center_x, center_y in centers:
        kwargs_lens.append(
            {
                "theta_E": np.random.uniform(0.65, 1.15) if compound else np.random.uniform(1.0, 2.0),
                "e1": 0.1,
                "e2": -0.1,
                "center_x": center_x,
                "center_y": center_y,
            }
        )
        kwargs_lens_light_mag.append(
            {
                "magnitude": np.random.uniform(20.0, 21.5),
                "R_sersic": 1.0,
                "n_sersic": 4.0,
                "e1": 0.1,
                "e2": -0.1,
                "center_x": center_x,
                "center_y": center_y,
            }
        )

    kwargs_source_mag = [
        {
            "magnitude": 20.0 + 1.5 * source_redshift,
            "R_sersic": 0.3 / (1 + 0.5 * source_redshift),
            "n_sersic": 2.0,
            "e1": 0.1,
            "e2": -0.1,
            "center_x": np.random.uniform(0.05, 0.2),
            "center_y": np.random.uniform(0.05, 0.2),
        }
    ]
    kwargs_lens_light, kwargs_source, _ = sim.magnitude2amplitude(
        kwargs_lens_light_mag=kwargs_lens_light_mag,
        kwargs_source_mag=kwargs_source_mag,
    )
    image_model = sim.image_model_class({"supersampling_factor": 1})
    image = image_model.image(
        kwargs_lens=kwargs_lens,
        kwargs_source=kwargs_source,
        kwargs_lens_light=kwargs_lens_light,
    )
    noise = sim.noise_for_model(
        model=image,
        background_noise=True,
        poisson_noise=True,
    )
    return image + noise, source_redshift


def make_non_lens_image(seed: int) -> np.ndarray:
    """Simulate the notebook's hard non-lens: a chance-aligned companion."""
    np.random.seed(seed)
    sim = make_simulator()
    kwargs_lens_light_mag = [
        {
            "magnitude": np.random.uniform(20.0, 21.5),
            "R_sersic": 1.0,
            "n_sersic": 4.0,
            "e1": 0.1,
            "e2": -0.1,
            "center_x": 0.0,
            "center_y": 0.0,
        }
    ]
    kwargs_source_mag = [
        {
            "magnitude": np.random.uniform(20.0, 22.0),
            "R_sersic": np.random.uniform(0.2, 0.5),
            "n_sersic": np.random.uniform(1.0, 4.0),
            "e1": np.random.uniform(-0.3, 0.3),
            "e2": np.random.uniform(-0.3, 0.3),
            "center_x": np.random.uniform(-1.0, 1.0),
            "center_y": np.random.uniform(-1.0, 1.0),
        }
    ]
    kwargs_lens = [
        {"theta_E": 0.01, "e1": 0.0, "e2": 0.0, "center_x": 0.0, "center_y": 0.0}
    ]
    kwargs_lens_light, kwargs_source, _ = sim.magnitude2amplitude(
        kwargs_lens_light_mag=kwargs_lens_light_mag,
        kwargs_source_mag=kwargs_source_mag,
    )
    image_model = sim.image_model_class({"supersampling_factor": 1})
    image = image_model.image(
        kwargs_lens=kwargs_lens,
        kwargs_source=kwargs_source,
        kwargs_lens_light=kwargs_lens_light,
    )
    noise = sim.noise_for_model(
        model=image,
        background_noise=True,
        poisson_noise=True,
    )
    return image + noise


def normalize_and_save(image_array: np.ndarray, output_path: Path) -> None:
    """Normalize a simulated pixel array to a viewable grayscale PNG."""
    low, high = np.percentile(image_array, (0.5, 99.5))
    image = np.clip(image_array, low, high)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((image * 255).astype(np.uint8)).save(output_path)


def render_sample_image(output_path: Path, lens_type: str, seed: int) -> float | None:
    """Simulate and save one sample; return source redshift for lenses."""
    if lens_type == "none":
        image = make_non_lens_image(seed)
        redshift = None
    else:
        image, redshift = make_lens_image(seed, compound=lens_type == "compound")
    normalize_and_save(image, output_path)
    return redshift


def generate_dataset(output_path: Path) -> None:
    rng = random.Random(42)
    image_root = output_path.parent / f"{output_path.stem}_images"
    rows: list[dict[str, object]] = []

    # 1,000 lens samples: 500 compound + 500 simple
    for idx in range(1000):
        is_compound = idx < 500
        lens_type = "compound" if is_compound else "simple"
        features = make_lens_features(is_compound, rng)
        sample_id = f"lens_{idx + 1:04d}"
        image_path = image_root / "lens" / lens_type / f"{sample_id}.png"
        redshift = render_sample_image(image_path, lens_type, idx)
        rows.append(
            {
                "id": sample_id,
                "label": 1,
                "lens_type": lens_type,
                "image_path": image_path.relative_to(output_path.parent).as_posix(),
                "redshift": round(redshift, 6),
                **{name: round(features[name], 6) for name in FEATURE_NAMES},
            }
        )

    # 1,000 non-lens samples
    for idx in range(1000):
        features = make_non_lens_features(rng)
        sample_id = f"nonlens_{idx + 1:04d}"
        image_path = image_root / "non_lens" / f"{sample_id}.png"
        render_sample_image(image_path, "none", 1000 + idx)
        rows.append(
            {
                "id": sample_id,
                "label": 0,
                "lens_type": "none",
                "image_path": image_path.relative_to(output_path.parent).as_posix(),
                "redshift": "",
                **{name: round(features[name], 6) for name in FEATURE_NAMES},
            }
        )

    rng.shuffle(rows)

    fieldnames = ["id", "label", "lens_type", "image_path", "redshift", *FEATURE_NAMES]
    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {len(rows)} samples at {output_path}")
    print(f"Lens count: {sum(row['label'] == 1 for row in rows)}")
    print(f"Non-lens count: {sum(row['label'] == 0 for row in rows)}")
    print(f"Compound lenses: {sum(row['label'] == 1 and row['lens_type'] == 'compound' for row in rows)}")
    print(f"Image files: {sum(1 for _ in image_root.rglob('*.png'))} in {image_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic lens/non-lens data.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("synthetic_lenses.csv"),
        help="Output CSV path for the generated dataset.",
    )
    args = parser.parse_args()
    generate_dataset(args.output)
