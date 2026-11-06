#!/usr/bin/env python3
"""Generate multi-band (g/r/i) DSPL-like images with redshift-dependent color.

Extends generate_dspl_experiment.py with:
- Three simulated bands (g, r, i) sharing the project's camera-style settings.
- A simple, explicitly placeholder color-redshift model: background sources get
  redder with redshift via linear g-r and r-i laws. This is NOT a K-correction;
  real survey colors need SED templates.
- A structure axis: "simple" (one SIE deflector) vs "compound" (two SIE
  deflectors on the same lens plane, matching generate_lenses.py's compound
  convention -- not a multi-plane compound model).
- Raw flux arrays saved as .npy next to viewable PNGs. Per-image PNG
  normalization destroys relative band flux, so all quantitative color or
  radial-profile analysis must use the .npy arrays, not the PNGs.

These are pilot assumptions for pipeline development, not calibrated
astrophysical parameter ranges.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.SimulationAPI.sim_api import SimAPI
from PIL import Image


PIXELS = 100
PIXEL_SCALE = 0.1
LENS_REDSHIFT = 0.3
SOURCE_REDSHIFTS = (1.0, 2.0)
BASE_THETA_E = 1.2
BASE_LENS_MAG_R = 20.75
BASE_SOURCE_MAG_R = 20.0
BASE_SOURCE_RADIUS = 0.3
BANDS = ("g", "r", "i")

# Placeholder linear color-redshift laws (pilot assumptions, not SED-derived).
# Sources get redder with redshift; the elliptical-like lens galaxy has fixed red colors.
SOURCE_G_MINUS_R = lambda z: 0.2 + 0.25 * z
SOURCE_R_MINUS_I = lambda z: 0.15 + 0.15 * z
LENS_G_MINUS_R = 1.0
LENS_R_MINUS_I = 0.6

IMAGE_SETTINGS = {
    "pixel_scale": PIXEL_SCALE,
    "exposure_time": 90,
    "magnitude_zero_point": 27.0,
    "read_noise": 7,
    "ccd_gain": 6.083,
    "sky_brightness": 23.5,
    "seeing": 0.16,
    "num_exposures": 1,
    "psf_type": "GAUSSIAN",
}

SWEEPS = {
    "faintness": ("source_magnitude_offset", (0.0, 1.0, 2.0, 3.0), "magnitude"),
    "arc_size": ("theta_e_scale", (0.50, 0.55, 0.70, 0.85, 1.0, 1.15, 1.30), "scale"),
    "blending": ("lens_light_flux_scale", (0.5, 1.0, 2.0, 4.0), "flux_scale"),
}


def sample_scene(seed: int) -> dict[str, float]:
    """Sample nuisance and structure parameters once so conditions pair by seed."""
    rng = np.random.default_rng(seed)
    return {
        "source1_x": float(rng.uniform(-0.18, 0.18)),
        "source1_y": float(rng.uniform(-0.18, 0.18)),
        "source2_x": float(rng.uniform(-0.18, 0.18)),
        "source2_y": float(rng.uniform(-0.18, 0.18)),
        "simple_theta_e": float(rng.uniform(1.0, 2.0)),
        "compound_theta_e_1": float(rng.uniform(0.65, 1.15)),
        "compound_theta_e_2": float(rng.uniform(0.65, 1.15)),
        "compound_lens1_x": float(rng.uniform(-0.35, -0.15)),
        "compound_lens1_y": float(rng.uniform(-0.25, 0.25)),
        "compound_lens2_x": float(rng.uniform(0.15, 0.35)),
        "compound_lens2_y": float(rng.uniform(-0.25, 0.25)),
    }


def band_magnitudes(base_r_magnitude: float, redshift: float) -> dict[str, float]:
    """Apply the placeholder color law: g = r + (g-r), i = r - (r-i)."""
    return {
        "g": base_r_magnitude + SOURCE_G_MINUS_R(redshift),
        "r": base_r_magnitude,
        "i": base_r_magnitude - SOURCE_R_MINUS_I(redshift),
    }


def lens_band_magnitudes(lens_flux_scale: float) -> dict[str, float]:
    base_r = BASE_LENS_MAG_R - 2.5 * math.log10(lens_flux_scale)
    return {
        "g": base_r + LENS_G_MINUS_R,
        "r": base_r,
        "i": base_r - LENS_R_MINUS_I,
    }


def lens_components(structure: str, scene: dict[str, float], theta_e_scale: float) -> list[dict[str, float]]:
    """Return the SIE component list for a simple or compound structure."""
    if structure == "simple":
        return [
            {
                "theta_E": scene["simple_theta_e"] * theta_e_scale,
                "e1": 0.1,
                "e2": -0.1,
                "center_x": 0.0,
                "center_y": 0.0,
            }
        ]
    return [
        {
            "theta_E": scene["compound_theta_e_1"] * theta_e_scale,
            "e1": 0.1,
            "e2": -0.1,
            "center_x": scene["compound_lens1_x"],
            "center_y": scene["compound_lens1_y"],
        },
        {
            "theta_E": scene["compound_theta_e_2"] * theta_e_scale,
            "e1": 0.1,
            "e2": -0.1,
            "center_x": scene["compound_lens2_x"],
            "center_y": scene["compound_lens2_y"],
        },
    ]


def make_image_model(structure: str) -> tuple[ImageModel, SimAPI]:
    """Build the multi-plane model with two source planes for one structure."""
    n_lenses = 1 if structure == "simple" else 2
    transform = np.array([[PIXEL_SCALE, 0.0], [0.0, PIXEL_SCALE]])
    image_data = ImageData(
        image_data=np.zeros((PIXELS, PIXELS), dtype=np.float64),
        exposure_time=IMAGE_SETTINGS["exposure_time"],
        transform_pix2angle=transform,
        ra_at_xy_0=-(PIXELS / 2) * PIXEL_SCALE,
        dec_at_xy_0=-(PIXELS / 2) * PIXEL_SCALE,
    )
    psf = PSF(psf_type="GAUSSIAN", fwhm=IMAGE_SETTINGS["seeing"], pixel_size=PIXEL_SCALE)
    lens_model = LensModel(
        lens_model_list=["SIE"] * n_lenses,
        z_lens=LENS_REDSHIFT,
        z_source=max(SOURCE_REDSHIFTS),
        lens_redshift_list=[LENS_REDSHIFT] * n_lenses,
        multi_plane=True,
    )
    source_model = LightModel(
        light_model_list=["SERSIC_ELLIPSE", "SERSIC_ELLIPSE"],
        source_redshift_list=list(SOURCE_REDSHIFTS),
    )
    lens_light_model = LightModel(light_model_list=["SERSIC_ELLIPSE"] * n_lenses)
    image_model = ImageModel(
        image_data,
        psf,
        lens_model_class=lens_model,
        source_model_class=source_model,
        lens_light_model_class=lens_light_model,
        kwargs_numerics={"supersampling_factor": 1},
    )
    converter = SimAPI(
        PIXELS,
        IMAGE_SETTINGS,
        {
            "lens_model_list": ["SIE"] * n_lenses,
            "source_light_model_list": ["SERSIC_ELLIPSE", "SERSIC_ELLIPSE"],
            "lens_light_model_list": ["SERSIC_ELLIPSE"] * n_lenses,
        },
    )
    return image_model, converter


def magnitude_to_amplitude(
    converter: SimAPI,
    structure: str,
    scene: dict[str, float],
    lens_mag_band: float,
    source_mags_band: tuple[float, float],
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Convert per-band magnitudes to amplitudes with the project's SimAPI convention."""
    lens_centers = (
        [(0.0, 0.0)]
        if structure == "simple"
        else [
            (scene["compound_lens1_x"], scene["compound_lens1_y"]),
            (scene["compound_lens2_x"], scene["compound_lens2_y"]),
        ]
    )
    lens_light_mag = [
        {
            "magnitude": lens_mag_band,
            "R_sersic": 1.0,
            "n_sersic": 4.0,
            "e1": 0.1,
            "e2": -0.1,
            "center_x": center_x,
            "center_y": center_y,
        }
        for center_x, center_y in lens_centers
    ]
    source_positions = (
        (scene["source1_x"], scene["source1_y"]),
        (scene["source2_x"], scene["source2_y"]),
    )
    source_mag = [
        {
            "magnitude": source_mags_band[index],
            "R_sersic": BASE_SOURCE_RADIUS / (1 + 0.5 * SOURCE_REDSHIFTS[index]),
            "n_sersic": 2.0,
            "e1": 0.1,
            "e2": -0.1,
            "center_x": source_positions[index][0],
            "center_y": source_positions[index][1],
        }
        for index in range(2)
    ]
    kwargs_lens_light, kwargs_source, _ = converter.magnitude2amplitude(
        kwargs_lens_light_mag=lens_light_mag,
        kwargs_source_mag=source_mag,
    )
    return kwargs_lens_light, kwargs_source


def normalize_and_save(image: np.ndarray, path: Path) -> None:
    """Viewable PNG only; quantitative analysis must use the saved .npy arrays."""
    lower, upper = np.percentile(image, (0.5, 99.5))
    clipped = np.clip(image, lower, upper)
    scaled = (clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((scaled * 255).astype(np.uint8), mode="L").save(path)


def render_scene(
    *,
    seed: int,
    scene: dict[str, float],
    structure: str,
    source_magnitude_offset: float,
    theta_e_scale: float,
    lens_light_flux_scale: float,
    negative: bool,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    """Render one scene in g/r/i; returns images and per-band source fluxes."""
    image_model, converter = make_image_model(structure)
    kwargs_lens = lens_components(structure, scene, theta_e_scale)
    if negative:
        for component in kwargs_lens:
            component["theta_E"] = 0.01

    source_r_mags = tuple(
        BASE_SOURCE_MAG_R + 1.5 * redshift + source_magnitude_offset
        for redshift in SOURCE_REDSHIFTS
    )
    lens_mags = lens_band_magnitudes(lens_light_flux_scale)
    source_mags = tuple(band_magnitudes(mag, z) for mag, z in zip(source_r_mags, SOURCE_REDSHIFTS))

    images: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    for band_index, band in enumerate(BANDS):
        kwargs_lens_light, kwargs_source = magnitude_to_amplitude(
            converter,
            structure,
            scene,
            lens_mags[band],
            (source_mags[0][band], source_mags[1][band]),
        )
        noiseless = image_model.image(
            kwargs_lens=kwargs_lens,
            kwargs_source=kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
        )
        source_fluxes = [
            float(
                image_model.source_surface_brightness(
                    kwargs_source=kwargs_source,
                    kwargs_lens=kwargs_lens,
                    k=source_index,
                ).sum()
            )
            for source_index in range(2)
        ]
        np.random.seed(seed + 100_000 + band_index)
        noise = converter.noise_for_model(
            model=noiseless,
            background_noise=True,
            poisson_noise=True,
        )
        images[band] = noiseless + noise
        metrics[band] = {
            "source1_flux": source_fluxes[0],
            "source2_flux": source_fluxes[1],
            "source1_magnitude": source_mags[0][band],
            "source2_magnitude": source_mags[1][band],
            "lens_magnitude": lens_mags[band],
        }
    return images, metrics


def make_row(
    *,
    output_root: Path,
    image_root: Path,
    array_root: Path,
    experiment: str,
    condition_value: float,
    condition_unit: str,
    replicate: int,
    seed: int,
    structure: str,
    source_magnitude_offset: float = 0.0,
    theta_e_scale: float = 1.0,
    lens_light_flux_scale: float = 1.0,
    negative: bool = False,
) -> dict[str, object]:
    scene = sample_scene(seed)
    images, metrics = render_scene(
        seed=seed,
        scene=scene,
        structure=structure,
        source_magnitude_offset=source_magnitude_offset,
        theta_e_scale=theta_e_scale,
        lens_light_flux_scale=lens_light_flux_scale,
        negative=negative,
    )
    kwargs_lens = lens_components(structure, scene, theta_e_scale)
    sample_id = f"{experiment}_{condition_value:g}_{structure}_rep{replicate:02d}"
    if negative:
        sample_id = f"negative_control_{structure}_rep{replicate:02d}"

    row: dict[str, object] = {
        "image_id": sample_id,
        "label": 0 if negative else 1,
        "experiment": experiment,
        "condition_value": condition_value,
        "condition_unit": condition_unit,
        "replicate": replicate,
        "seed": seed,
        "lens_structure": structure,
        "lens_redshift": LENS_REDSHIFT,
        "source1_redshift": SOURCE_REDSHIFTS[0],
        "source2_redshift": SOURCE_REDSHIFTS[1],
        "source_magnitude_offset": source_magnitude_offset,
        "theta_e_scale": theta_e_scale,
        "lens_light_flux_scale": lens_light_flux_scale,
        "theta_e_1_arcsec": kwargs_lens[0]["theta_E"],
        "theta_e_2_arcsec": kwargs_lens[1]["theta_E"] if len(kwargs_lens) > 1 else "",
        "lens1_x_arcsec": kwargs_lens[0]["center_x"],
        "lens1_y_arcsec": kwargs_lens[0]["center_y"],
        "lens2_x_arcsec": kwargs_lens[1]["center_x"] if len(kwargs_lens) > 1 else "",
        "lens2_y_arcsec": kwargs_lens[1]["center_y"] if len(kwargs_lens) > 1 else "",
        "g_r_color_source1": SOURCE_G_MINUS_R(SOURCE_REDSHIFTS[0]),
        "g_r_color_source2": SOURCE_G_MINUS_R(SOURCE_REDSHIFTS[1]),
        "r_i_color_source1": SOURCE_R_MINUS_I(SOURCE_REDSHIFTS[0]),
        "r_i_color_source2": SOURCE_R_MINUS_I(SOURCE_REDSHIFTS[1]),
        "source1_x_arcsec": scene["source1_x"],
        "source1_y_arcsec": scene["source1_y"],
        "source2_x_arcsec": scene["source2_x"],
        "source2_y_arcsec": scene["source2_y"],
        "image_width": PIXELS,
        "image_height": PIXELS,
        **IMAGE_SETTINGS,
    }
    for band in BANDS:
        image_path = image_root / band / experiment / structure / f"{sample_id}.png"
        array_path = array_root / band / experiment / structure / f"{sample_id}.npy"
        array_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(array_path, images[band])
        normalize_and_save(images[band], image_path)
        row[f"image_path_{band}"] = Path(os.path.relpath(image_path, output_root)).as_posix()
        row[f"array_path_{band}"] = Path(os.path.relpath(array_path, output_root)).as_posix()
        for key, value in metrics[band].items():
            row[f"{band}_{key}"] = value
    return row


def generate(
    output_csv: Path,
    replicates: int,
    negative_controls: int,
    image_dir: Path | None = None,
    array_dir: Path | None = None,
) -> None:
    output_csv = output_csv.resolve()
    output_root = output_csv.parent
    image_root = (image_dir or output_root / "images").resolve()
    array_root = (array_dir or output_root / "arrays").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    base_seed = 20261105

    for experiment, (parameter, values, unit) in SWEEPS.items():
        for value in values:
            for structure in ("simple", "compound"):
                for replicate in range(replicates):
                    seed = base_seed + replicate
                    settings: dict[str, float] = {
                        "source_magnitude_offset": 0.0,
                        "theta_e_scale": 1.0,
                        "lens_light_flux_scale": 1.0,
                    }
                    settings[parameter] = value
                    rows.append(
                        make_row(
                            output_root=output_root,
                            image_root=image_root,
                            array_root=array_root,
                            experiment=experiment,
                            condition_value=value,
                            condition_unit=unit,
                            replicate=replicate,
                            seed=seed,
                            structure=structure,
                            **settings,
                        )
                    )

    for structure in ("simple", "compound"):
        for replicate in range(negative_controls):
            rows.append(
                make_row(
                    output_root=output_root,
                    image_root=image_root,
                    array_root=array_root,
                    experiment="negative_control",
                    condition_value=1.0,
                    condition_unit="control",
                    replicate=replicate,
                    seed=base_seed + replicate,
                    structure=structure,
                    negative=True,
                )
            )

    fieldnames = list(rows[0])
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lens_rows = [row for row in rows if row["label"] == 1]
    for band in BANDS:
        assert all(
            row[f"{band}_source1_flux"] > 0 and row[f"{band}_source2_flux"] > 0
            for row in lens_rows
        )
    print(f"Generated {len(rows)} scenes x {len(BANDS)} bands; manifest: {output_csv}")
    print(f"Lenses: {len(lens_rows)}; negative controls: {len(rows) - len(lens_rows)}")
    print("Each positive contains nonzero flux from both source planes in every band.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dspl_color_experiment/simulation_manifest.csv"),
        help="CSV manifest path; images/arrays are saved beside it.",
    )
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument(
        "--negative-controls",
        type=int,
        default=None,
        help="Number of negative controls per structure; defaults to --replicates.",
    )
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--array-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.replicates < 2:
        parser.error("--replicates must be at least 2 for condition comparisons")
    negative_controls = args.replicates if args.negative_controls is None else args.negative_controls
    if negative_controls < 1:
        parser.error("--negative-controls must be at least 1")
    generate(args.output, args.replicates, negative_controls, args.image_dir, args.array_dir)
