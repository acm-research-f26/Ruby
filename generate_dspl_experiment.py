#!/usr/bin/env python3
"""Generate a small, reproducible DSPL-like image experiment."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.SimulationAPI.sim_api import SimAPI
from PIL import Image


ROOT = Path(__file__).resolve().parent
PIXELS = 100
PIXEL_SCALE = 0.1
LENS_REDSHIFT = 0.3
SOURCE_REDSHIFTS = (1.0, 2.0)
BASE_THETA_E = 1.2
BASE_LENS_MAG = 20.75
BASE_SOURCE_MAGNITUDE = 20.0
BASE_SOURCE_RADIUS = 0.3
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
    "arc_size": ("theta_e_scale", (0.70, 0.85, 1.0, 1.15, 1.30), "scale"),
    "blending": ("lens_light_flux_scale", (0.5, 1.0, 2.0, 4.0), "flux_scale"),
}


def sample_scene(seed: int) -> dict[str, float]:
    """Sample nuisance positions once so each condition is paired by seed."""
    rng = np.random.default_rng(seed)
    return {
        "source1_x": float(rng.uniform(-0.18, 0.18)),
        "source1_y": float(rng.uniform(-0.18, 0.18)),
        "source2_x": float(rng.uniform(-0.18, 0.18)),
        "source2_y": float(rng.uniform(-0.18, 0.18)),
    }


def make_image_model() -> tuple[ImageModel, SimAPI]:
    """Build one common lens plane and two source planes with Lenstronomy."""
    transform = np.array([[PIXEL_SCALE, 0.0], [0.0, PIXEL_SCALE]])
    image_data = ImageData(
        image_data=np.zeros((PIXELS, PIXELS), dtype=np.float64),
        exposure_time=IMAGE_SETTINGS["exposure_time"],
        transform_pix2angle=transform,
        ra_at_xy_0=-(PIXELS / 2) * PIXEL_SCALE,
        dec_at_xy_0=-(PIXELS / 2) * PIXEL_SCALE,
    )
    psf = PSF(
        psf_type=IMAGE_SETTINGS["psf_type"],
        fwhm=IMAGE_SETTINGS["seeing"],
        pixel_size=PIXEL_SCALE,
    )
    lens_model = LensModel(
        lens_model_list=["SIE"],
        z_lens=LENS_REDSHIFT,
        z_source=max(SOURCE_REDSHIFTS),
        lens_redshift_list=[LENS_REDSHIFT],
        multi_plane=True,
    )
    source_model = LightModel(
        light_model_list=["SERSIC_ELLIPSE", "SERSIC_ELLIPSE"],
        source_redshift_list=list(SOURCE_REDSHIFTS),
    )
    lens_light_model = LightModel(light_model_list=["SERSIC_ELLIPSE"])
    image_model = ImageModel(
        image_data,
        psf,
        lens_model_class=lens_model,
        source_model_class=source_model,
        lens_light_model_class=lens_light_model,
        kwargs_numerics={"supersampling_factor": 1},
    )

    amplitude_converter = SimAPI(
        PIXELS,
        IMAGE_SETTINGS,
        {
            "lens_model_list": ["SIE"],
            "source_light_model_list": ["SERSIC_ELLIPSE", "SERSIC_ELLIPSE"],
            "lens_light_model_list": ["SERSIC_ELLIPSE"],
        },
    )
    return image_model, amplitude_converter


def magnitude_to_amplitude(
    converter: SimAPI,
    lens_magnitude: float,
    source_magnitudes: tuple[float, float],
    source_positions: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Use the project's existing SimAPI magnitude-to-amplitude convention."""
    lens_light_mag = [
        {
            "magnitude": lens_magnitude,
            "R_sersic": 1.0,
            "n_sersic": 4.0,
            "e1": 0.1,
            "e2": -0.1,
            "center_x": 0.0,
            "center_y": 0.0,
        }
    ]
    source_mag = [
        {
            "magnitude": source_magnitudes[index],
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
    """Match the existing project convention for viewable grayscale PNGs."""
    lower, upper = np.percentile(image, (0.5, 99.5))
    clipped = np.clip(image, lower, upper)
    scaled = (clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((scaled * 255).astype(np.uint8), mode="L").save(path)


def render_image(
    *,
    seed: int,
    scene: dict[str, float],
    source_magnitude_offset: float = 0.0,
    theta_e_scale: float = 1.0,
    lens_light_flux_scale: float = 1.0,
    negative: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    image_model, converter = make_image_model()
    source_positions = (
        (scene["source1_x"], scene["source1_y"]),
        (scene["source2_x"], scene["source2_y"]),
    )
    source_magnitudes = tuple(
        BASE_SOURCE_MAGNITUDE + 1.5 * redshift + source_magnitude_offset
        for redshift in SOURCE_REDSHIFTS
    )
    lens_magnitude = BASE_LENS_MAG - 2.5 * math.log10(lens_light_flux_scale)
    kwargs_lens_light, kwargs_source = magnitude_to_amplitude(
        converter,
        lens_magnitude,
        source_magnitudes,
        source_positions,
    )
    theta_e = 0.01 if negative else BASE_THETA_E * theta_e_scale
    kwargs_lens = [
        {
            "theta_E": theta_e,
            "e1": 0.1,
            "e2": -0.1,
            "center_x": 0.0,
            "center_y": 0.0,
        }
    ]
    noiseless = image_model.image(
        kwargs_lens=kwargs_lens,
        kwargs_source=kwargs_source,
        kwargs_lens_light=kwargs_lens_light,
    )

    # Record per-plane lensed light before adding noise to verify both sources contribute.
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
    noise_simulator = SimAPI(
        PIXELS,
        IMAGE_SETTINGS,
        {
            "lens_model_list": ["SIE"],
            "source_light_model_list": ["SERSIC_ELLIPSE"],
            "lens_light_model_list": ["SERSIC_ELLIPSE"],
        },
    )
    np.random.seed(seed + 100_000)
    noise = noise_simulator.noise_for_model(
        model=noiseless,
        background_noise=True,
        poisson_noise=True,
    )
    return noiseless + noise, {
        "source1_flux": source_fluxes[0],
        "source2_flux": source_fluxes[1],
        "theta_e": theta_e,
        "lens_light_magnitude": lens_magnitude,
        "source1_magnitude": source_magnitudes[0],
        "source2_magnitude": source_magnitudes[1],
    }


def make_row(
    *,
    output_root: Path,
    experiment: str,
    condition_value: float,
    condition_unit: str,
    replicate: int,
    seed: int,
    source_magnitude_offset: float = 0.0,
    theta_e_scale: float = 1.0,
    lens_light_flux_scale: float = 1.0,
    negative: bool = False,
) -> dict[str, object]:
    scene = sample_scene(seed)
    image, measured = render_image(
        seed=seed,
        scene=scene,
        source_magnitude_offset=source_magnitude_offset,
        theta_e_scale=theta_e_scale,
        lens_light_flux_scale=lens_light_flux_scale,
        negative=negative,
    )
    sample_id = f"{experiment}_{condition_value:g}_rep{replicate:02d}"
    if negative:
        sample_id = f"negative_control_rep{replicate:02d}"
    relative_path = Path("images") / experiment / f"{sample_id}.png"
    normalize_and_save(image, output_root / relative_path)
    return {
        "image_id": sample_id,
        "image_path": relative_path.as_posix(),
        "label": 0 if negative else 1,
        "experiment": experiment,
        "condition_value": condition_value,
        "condition_unit": condition_unit,
        "replicate": replicate,
        "seed": seed,
        "lens_redshift": LENS_REDSHIFT,
        "source1_redshift": SOURCE_REDSHIFTS[0],
        "source2_redshift": SOURCE_REDSHIFTS[1],
        "source_magnitude_offset": source_magnitude_offset,
        "theta_e_scale": theta_e_scale,
        "theta_e_arcsec": measured["theta_e"],
        "lens_light_flux_scale": lens_light_flux_scale,
        "lens_light_magnitude": measured["lens_light_magnitude"],
        "source1_magnitude": measured["source1_magnitude"],
        "source2_magnitude": measured["source2_magnitude"],
        "source1_x_arcsec": scene["source1_x"],
        "source1_y_arcsec": scene["source1_y"],
        "source2_x_arcsec": scene["source2_x"],
        "source2_y_arcsec": scene["source2_y"],
        "source1_lensed_flux": measured["source1_flux"],
        "source2_lensed_flux": measured["source2_flux"],
        "image_width": PIXELS,
        "image_height": PIXELS,
        **IMAGE_SETTINGS,
    }


def generate(output_csv: Path, replicates: int, negative_controls: int) -> None:
    output_csv = output_csv.resolve()
    output_root = output_csv.parent
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    base_seed = 20261016

    for experiment, (parameter, values, unit) in SWEEPS.items():
        for value in values:
            for replicate in range(replicates):
                seed = base_seed + replicate
                settings: dict[str, float] = {
                    "source_magnitude_offset": 0.0,
                    "theta_e_scale": 1.0,
                    "lens_light_flux_scale": 1.0,
                }
                if parameter == "source_magnitude_offset":
                    settings[parameter] = value
                elif parameter == "theta_e_scale":
                    settings[parameter] = value
                else:
                    settings[parameter] = value
                rows.append(
                    make_row(
                        output_root=output_root,
                        experiment=experiment,
                        condition_value=value,
                        condition_unit=unit,
                        replicate=replicate,
                        seed=seed,
                        **settings,
                    )
                )

    for replicate in range(negative_controls):
        rows.append(
            make_row(
                output_root=output_root,
                experiment="negative_control",
                condition_value=1.0,
                condition_unit="control",
                replicate=replicate,
                seed=base_seed + replicate,
                negative=True,
            )
        )

    fieldnames = list(rows[0])
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lens_rows = [row for row in rows if row["label"] == 1]
    assert all(row["source1_lensed_flux"] > 0 and row["source2_lensed_flux"] > 0 for row in lens_rows)
    print(f"Generated {len(rows)} images and manifest: {output_csv}")
    print(f"Lenses: {len(lens_rows)}; negative controls: {len(rows) - len(lens_rows)}")
    print("Each positive contains nonzero flux from both configured source planes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dspl_experiment/simulation_manifest.csv"),
        help="CSV manifest path; images are saved beside it under images/.",
    )
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument(
        "--negative-controls",
        type=int,
        default=None,
        help="Number of negative controls; defaults to --replicates.",
    )
    args = parser.parse_args()
    if args.replicates < 2:
        parser.error("--replicates must be at least 2 for condition comparisons")
    negative_controls = args.replicates if args.negative_controls is None else args.negative_controls
    if negative_controls < 1:
        parser.error("--negative-controls must be at least 1")
    generate(args.output, args.replicates, negative_controls)