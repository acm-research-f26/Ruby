"""The validation the radial-profile/angular-coverage method was missing:
does it actually detect a second ring when one genuinely exists?

Builds synthetic double-ring images by combining two independent lensed-arc
renders (at controlled, different radii) onto one shared lens-galaxy +
noise background, then sweeps radius separation and brightness ratio to map
out exactly where detection succeeds vs. fails -- the same kind of sweep we
ran earlier for the Zoobot ring-size sensitivity work, now for this method.

This is a proxy for a true multi-plane (two-deflector) compound lens, not a
full physical simulation of one -- two independently-rendered single-plane
arcs at different radii, combined. It tests the detection PIPELINE directly
(does it correctly count 2 angularly-broad bumps when 2 exist), without the
added complexity/time cost of multi-plane lens modeling.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from dataset import load_config
from lens_sim import LensSimulator, fixed_linear_scale
from radial_profile import count_rings

ROOT = Path(__file__).resolve().parent.parent

INNER_THETA_E = 1.0  # arcsec -- a representative, well-detected single-ring radius
SEPARATION_RATIOS = [1.3, 1.5, 2.0, 3.0, 4.0]   # outer/inner theta_E ratio
BRIGHTNESS_RATIOS = [0.4, 0.7, 1.0, 1.5]         # outer arc peak / inner arc peak
N_REPEATS = 15

SCALE_LOW = -0.01850
SCALE_HIGH = 0.87001


def build_double_ring(sim: LensSimulator, rng: np.random.Generator,
                       inner_theta_e: float, outer_theta_e: float, brightness_ratio: float):
    background, _ = sim.sample_background_only(rng)
    inner_arc, _ = sim.sample_lensed_arc_only(rng, theta_e=inner_theta_e)
    outer_arc, _ = sim.sample_lensed_arc_only(rng, theta_e=outer_theta_e)

    inner_peak = inner_arc.max()
    outer_peak = outer_arc.max()
    if outer_peak > 0:
        outer_arc = outer_arc * (brightness_ratio * inner_peak / outer_peak)

    raw = background + inner_arc + outer_arc
    return fixed_linear_scale(raw, SCALE_LOW, SCALE_HIGH).astype(np.float32)


def main():
    cfg = load_config(ROOT / "configs" / "sim_config_large.yaml")
    sim = LensSimulator(cfg)
    rng = np.random.default_rng(2026)

    results = {}
    print(f"Sweeping {len(SEPARATION_RATIOS)} separations x {len(BRIGHTNESS_RATIOS)} "
          f"brightness ratios x {N_REPEATS} repeats = "
          f"{len(SEPARATION_RATIOS)*len(BRIGHTNESS_RATIOS)*N_REPEATS} synthetic double rings...\n")

    print(f"{'sep ratio':>10} {'bright ratio':>13} {'detection rate':>16}")
    for sep in SEPARATION_RATIOS:
        outer_theta_e = INNER_THETA_E * sep
        for bright in BRIGHTNESS_RATIOS:
            n_detected = 0
            for _ in range(N_REPEATS):
                img = build_double_ring(sim, rng, INNER_THETA_E, outer_theta_e, bright)
                n_ring_like, classifications, *_ = count_rings(img)
                if n_ring_like >= 2:
                    n_detected += 1
            rate = n_detected / N_REPEATS
            results[(sep, bright)] = rate
            print(f"{sep:>10.1f} {bright:>13.1f} {rate:>15.0%}")

    out_path = ROOT / "outputs" / "double_ring_sensitivity.npz"
    seps = np.array(SEPARATION_RATIOS)
    brights = np.array(BRIGHTNESS_RATIOS)
    rate_grid = np.array([[results[(s, b)] for b in BRIGHTNESS_RATIOS] for s in SEPARATION_RATIOS])
    np.savez(out_path, separation_ratios=seps, brightness_ratios=brights, detection_rate=rate_grid)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
