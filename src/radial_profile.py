"""Radial brightness profile analysis: detect whether a verified single lens
might actually be a compound (double-ring) lens, without needing any trained
model or labeled compound-lens data.

Method: compute the azimuthally-averaged flux vs. radius from the lens
center. A single lens shows one bump (the Einstein ring) on top of the
smoothly-declining host-galaxy light. A compound lens should show a SECOND
distinct bump at a different radius. We detrend the smooth decline (a
monotonic lower envelope) and run peak-finding on the residual.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d


def radial_profile(image: np.ndarray, center: tuple[float, float] | None = None,
                    n_bins: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthally-averaged flux vs. radius (pixels). Returns (radii, flux)."""
    h, w = image.shape
    cy, cx = center if center is not None else (h / 2, w / 2)
    y, x = np.indices(image.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    r_max = min(h, w) / 2
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    profile = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        profile[i] = image[mask].mean() if mask.any() else 0.0
    return bin_centers, profile


def detrend(profile: np.ndarray, smooth_window: int = 7) -> np.ndarray:
    """Residual = profile minus its own heavy smoothing -- isolates local
    bumps (rings) from the smooth monotonic galaxy-light decline."""
    smoothed = uniform_filter1d(profile, size=smooth_window, mode="nearest")
    return profile - smoothed


def angular_profile(image: np.ndarray, center: tuple[float, float], radius: float,
                     annulus_half_width: float = 2.0, n_angle_bins: int = 72):
    """Flux vs. angle (degrees, 0-360) within a thin ring at the given radius.
    This is the piece that lets us tell a ring apart from a companion object:
    a ring's flux is spread across a wide angular range at roughly one
    radius, a companion galaxy's flux is concentrated at one angle only."""
    h, w = image.shape
    cy, cx = center
    y, x = np.indices(image.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    theta = (np.degrees(np.arctan2(y - cy, x - cx)) + 360) % 360

    in_annulus = (r >= radius - annulus_half_width) & (r < radius + annulus_half_width)
    angle_edges = np.linspace(0, 360, n_angle_bins + 1)
    angle_centers = (angle_edges[:-1] + angle_edges[1:]) / 2
    flux = np.zeros(n_angle_bins)
    for i in range(n_angle_bins):
        mask = in_annulus & (theta >= angle_edges[i]) & (theta < angle_edges[i + 1])
        flux[i] = image[mask].mean() if mask.any() else 0.0
    return angle_centers, flux


def angular_coverage(image: np.ndarray, center: tuple[float, float], radius: float,
                      annulus_half_width: float = 2.0, n_angle_bins: int = 72) -> float:
    """Fraction (0-1) of the full circle at this radius where flux exceeds
    half the annulus's own peak-above-background value. A full/broad ring
    scores high; a single compact companion object scores low (its flux is
    confined to a narrow angular wedge, however bright that wedge is)."""
    _, flux = angular_profile(image, center, radius, annulus_half_width, n_angle_bins)
    background = np.median(flux)
    half_max = background + 0.5 * (flux.max() - background)
    if flux.max() <= background:
        return 0.0
    return float((flux >= half_max).sum()) / n_angle_bins


def classify_peak(image: np.ndarray, center: tuple[float, float], radius: float,
                   ring_coverage_threshold: float = 0.02, edge_fraction: float = 0.85) -> dict:
    """Labels one detected radial peak as 'ring-like' (likely a genuine
    second Einstein ring) or 'point-like' (likely an unrelated companion
    galaxy or saturation artifact), using its angular coverage, plus an
    independent edge-proximity flag.

    Threshold calibrated empirically on 72 angle bins, re-checked on a
    30-image sample (29 genuine-arc peaks) after the double-ring sensitivity
    test exposed that the original 36-bin/0.05 calibration was too close to
    its own companion floor (single-bin signature 1/36=2.8% vs. observed
    genuine-arc minimum ~3%, almost no margin -- borderline real arcs were
    getting misclassified). At 72 bins: single-bin companion floor = 1/72 =
    1.4%, genuine-arc coverage ranges 2.8-26% (median ~7%). 0.02 sits
    cleanly in the gap between those two populations, with real margin on
    both sides -- but a real remaining edge case exists: a peak right at
    this boundary (~0.02-0.03) that also sits very near the image edge is
    genuinely ambiguous by angular coverage alone (checked empirically:
    one real lens has a 0.028-coverage peak at r=44.4px in a 50px-max
    frame). A genuine second ring from the SAME lens system should sit well
    within the cutout; near_edge=True is a second, independent, physically-
    motivated reason for caution, reported rather than silently resolved --
    deliberately not auto-excluding on it, since one example is too little
    to safely calibrate a hard cutoff against."""
    coverage = angular_coverage(image, center, radius)
    r_max = min(image.shape) / 2
    near_edge = radius >= edge_fraction * r_max
    return {
        "radius": radius,
        "angular_coverage": coverage,
        "near_edge": near_edge,
        "classification": "ring-like" if coverage >= ring_coverage_threshold else "point-like (likely companion)",
    }


def count_rings(image: np.ndarray, center: tuple[float, float] | None = None,
                 n_bins: int = 40, prominence_sigma: float = 2.0, min_radius_px: float = 3.0,
                 min_peak_separation_px: float = 9.0):
    """Returns (n_peaks_found, radii_of_peaks, radii, raw_profile, residual).

    prominence_sigma: a peak must stick up at least this many standard
    deviations of the residual's OWN noise floor to count -- relative, not a
    fixed absolute number, since brightness/contrast varies a lot image to
    image (checked empirically: residual std ranges ~0.001 to ~0.025 across
    our real lens sample).

    min_peak_separation_px: a SINGLE asymmetric/clumpy arc (off-center
    source -> non-circular arc) smears into two close sub-peaks in a purely
    azimuthally-averaged profile even though it's genuinely one ring --
    checked empirically on 12 known-single simulated lenses, spurious gaps
    cluster at 5-7.5px. This floor merges those; a real second ring should be
    separated by considerably more than one ring's own clumpiness."""
    radii, profile = radial_profile(image, center, n_bins)
    residual = detrend(profile)
    # Ignore the innermost few bins -- that's the galaxy core, not a ring.
    valid = radii >= min_radius_px
    # Robust (MAD-based) noise floor, not plain std -- a real ring peak would
    # otherwise inflate its own detection threshold, since std is sensitive
    # to the exact outlier a peak represents while MAD mostly isn't.
    resid_valid = residual[valid]
    mad = np.median(np.abs(resid_valid - np.median(resid_valid)))
    noise_floor = 1.4826 * mad
    prominence = prominence_sigma * noise_floor
    bin_width = radii[1] - radii[0]
    distance_bins = max(1, int(round(min_peak_separation_px / bin_width)))
    peak_idx, props = find_peaks(residual[valid], prominence=prominence, distance=distance_bins)
    peak_radii = radii[valid][peak_idx]

    center_xy = center if center is not None else (image.shape[0] / 2, image.shape[1] / 2)
    classifications = [classify_peak(image, center_xy, r) for r in peak_radii]
    n_ring_like = sum(1 for c in classifications if c["classification"] == "ring-like")

    return n_ring_like, classifications, radii, profile, residual
