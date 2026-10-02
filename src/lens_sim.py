"""Single (galaxy-galaxy) strong lens simulation with lenstronomy.

Physics: SIE lens galaxy + Sersic source light. The Einstein radius is derived
from a sampled velocity dispersion and the lens/source redshifts via the
singular-isothermal-sphere relation, matching eq. (1) of Wilde et al. 2022:

    theta_E = 4*pi*(sigma_v/c)^2 * D_ls / D_s
"""
from __future__ import annotations

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy import constants as const
from lenstronomy.SimulationAPI.sim_api import SimAPI


def einstein_radius_arcsec(sigma_v_kms, z_lens, z_source, cosmo):
    """SIS/SIE Einstein radius in arcsec from velocity dispersion (eq. 1, Wilde+22)."""
    sigma_v = sigma_v_kms * u.km / u.s
    d_ls = cosmo.angular_diameter_distance(z_lens, z_source)
    d_s = cosmo.angular_diameter_distance(z_source)
    theta_rad = 4 * np.pi * (sigma_v / const.c) ** 2 * (d_ls / d_s)
    return theta_rad.decompose().value * (180.0 / np.pi) * 3600.0


class LensSimulator:
    """Builds single-lens and matched non-lens images from a config dict."""

    def __init__(self, config: dict):
        self.cfg = config
        self.cosmo = FlatLambdaCDM(H0=config["cosmology"]["H0"], Om0=config["cosmology"]["Om0"])

        img = config["image"]
        self.num_pix = img["num_pix"]
        self.kwargs_band = {
            "read_noise": img["read_noise"],
            "pixel_scale": img["pixel_scale"],
            "ccd_gain": img["ccd_gain"],
            "exposure_time": img["exposure_time"],
            "magnitude_zero_point": img["magnitude_zero_point"],
            "num_exposures": 1,
            "sky_brightness": img["sky_brightness"],
            "seeing": img["psf_fwhm"],
            "psf_type": "GAUSSIAN",
        }
        self.kwargs_model = {
            "lens_model_list": ["SIE", "SHEAR"],
            "source_light_model_list": ["SERSIC_ELLIPSE"],
            "lens_light_model_list": ["SERSIC_ELLIPSE"],
        }
        self.sim_api = SimAPI(num_pix=self.num_pix, kwargs_single_band=self.kwargs_band,
                               kwargs_model=self.kwargs_model)
        self.image_model = self.sim_api.image_model_class({"supersampling_factor": 1})

    def _sample_common(self, rng: np.random.Generator):
        lg, sg = self.cfg["lens_galaxy"], self.cfg["source_galaxy"]

        z_lens = rng.uniform(*lg["z_range"])
        z_source = rng.uniform(z_lens + 0.2, sg["z_range"][1])
        z_source = max(z_source, sg["z_range"][0])
        sigma_v = rng.uniform(*lg["velocity_dispersion_range"])
        theta_e = einstein_radius_arcsec(sigma_v, z_lens, z_source, self.cosmo)

        e_lens = rng.uniform(*lg["ellipticity_range"])
        pa_lens = rng.uniform(0, np.pi)
        e1_lens, e2_lens = e_lens * np.cos(2 * pa_lens), e_lens * np.sin(2 * pa_lens)
        gamma_ext = rng.uniform(*lg["external_shear_range"])
        pa_shear = rng.uniform(0, np.pi)
        gamma1, gamma2 = gamma_ext * np.cos(2 * pa_shear), gamma_ext * np.sin(2 * pa_shear)

        kwargs_lens = [
            {"theta_E": theta_e, "e1": e1_lens, "e2": e2_lens, "center_x": 0.0, "center_y": 0.0},
            {"gamma1": gamma1, "gamma2": gamma2, "ra_0": 0.0, "dec_0": 0.0},
        ]

        kwargs_lens_light_mag = [{
            "magnitude": rng.uniform(*lg["light_magnitude_range"]),
            "R_sersic": rng.uniform(*lg["light_r_eff_range"]),
            "n_sersic": rng.uniform(*lg["light_sersic_n_range"]),
            "e1": e1_lens, "e2": e2_lens, "center_x": 0.0, "center_y": 0.0,
        }]

        source_x = rng.uniform(*sg["offset_range"])
        source_y = rng.uniform(*sg["offset_range"])
        e_src = rng.uniform(*sg["ellipticity_range"])
        pa_src = rng.uniform(0, np.pi)
        e1_src, e2_src = e_src * np.cos(2 * pa_src), e_src * np.sin(2 * pa_src)
        kwargs_source_mag = [{
            "magnitude": rng.uniform(*sg["magnitude_range"]),
            "R_sersic": rng.uniform(*sg["r_eff_range"]),
            "n_sersic": rng.uniform(*sg["sersic_n_range"]),
            "e1": e1_src, "e2": e2_src, "center_x": source_x, "center_y": source_y,
        }]

        meta = dict(z_lens=z_lens, z_source=z_source, sigma_v=sigma_v, theta_E=theta_e,
                    source_x=source_x, source_y=source_y)
        return kwargs_lens, kwargs_lens_light_mag, kwargs_source_mag, meta

    def _render(self, kwargs_lens, kwargs_lens_light_mag, kwargs_source_mag, lensed: bool):
        kwargs_lens_light, kwargs_source, _ = self.sim_api.magnitude2amplitude(
            kwargs_lens_light_mag, kwargs_source_mag)
        if lensed:
            image = self.image_model.image(kwargs_lens, kwargs_source, kwargs_lens_light)
        else:
            # Non-lens control: same lens-light galaxy, source painted directly at its
            # unlensed position/flux (no deflection) -- i.e. no SIE/SHEAR applied.
            zero_lens = [{"theta_E": 0.0, "e1": 0.0, "e2": 0.0, "center_x": 0.0, "center_y": 0.0},
                         {"gamma1": 0.0, "gamma2": 0.0, "ra_0": 0.0, "dec_0": 0.0}]
            image = self.image_model.image(zero_lens, kwargs_source, kwargs_lens_light)
        noise = self.sim_api.noise_for_model(model=image)
        return image + noise

    def sample_lens(self, rng: np.random.Generator):
        kwargs_lens, kwargs_lens_light_mag, kwargs_source_mag, meta = self._sample_common(rng)
        image = self._render(kwargs_lens, kwargs_lens_light_mag, kwargs_source_mag, lensed=True)
        return image, meta

    def sample_nonlens(self, rng: np.random.Generator):
        kwargs_lens, kwargs_lens_light_mag, kwargs_source_mag, meta = self._sample_common(rng)
        image = self._render(kwargs_lens, kwargs_lens_light_mag, kwargs_source_mag, lensed=False)
        return image, meta


def fixed_linear_scale(image: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linear scaling to [0, 1] using a FIXED (lo, hi) shared across the whole
    dataset, so pixel values stay directly comparable/measurable across images
    and batches -- unlike a per-image percentile stretch, which would rescale
    every image to its own min/max and erase real brightness differences."""
    out = (image - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def calibrate_global_scale(config: dict, n_calib: int = 300, calib_seed: int = 999,
                            lo_pct: float = 0.5, hi_pct: float = 99.9):
    """Draw a calibration sample of raw (unscaled) lens + non-lens images under
    this config and return a fixed (lo, hi) intensity range spanning the whole
    population's pixel distribution. Run once per config; the result should be
    stored back into the config (image.scale_low/scale_high) so every dataset
    generated from it -- across separate batches/runs -- shares the same scale."""
    rng = np.random.default_rng(calib_seed)
    sim = LensSimulator(config)
    pixels = []
    for _ in range(n_calib // 2):
        img, _ = sim.sample_lens(rng)
        pixels.append(img.ravel())
    for _ in range(n_calib // 2):
        img, _ = sim.sample_nonlens(rng)
        pixels.append(img.ravel())
    all_pixels = np.concatenate(pixels)
    lo, hi = np.percentile(all_pixels, [lo_pct, hi_pct])
    return float(lo), float(hi)
