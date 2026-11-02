# measure_blur.py: how blurred are our cutouts? Measured on point-like objects in the images themselves.
#
# Usage (from ~/ruby, with lens_env311):
#   python sim/measure_blur.py
#
# Finds bright, compact, isolated objects in the training and validation cutouts, keeps the most compact
# 15%, and fits one or two Gaussians to their brightness profile. It then measures a simulated point the
# same way. Simulate.py's blur settings are the ones that make the two agree.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate as sim
from simulate import ROOT

HALF = 12
yy, xx = np.mgrid[-HALF:HALF + 1, -HALF:HALF + 1]


def stamp_of(y, r, c):
    """(size, peak, distance from the centre of light, pixels) of the object at row r, column c."""
    s = y[r - HALF:r + HALF + 1, c - HALF:c + HALF + 1]
    core = np.clip(s, 0, None) * (np.hypot(xx, yy) < 5)
    cy, cx = (core * yy).sum() / core.sum(), (core * xx).sum() / core.sum()
    rr = np.hypot(xx - cx, yy - cy)
    return np.sqrt((core * rr ** 2).sum() / core.sum()), s.max(), rr, s


def gauss(r, f):
    return np.exp(-r ** 2 / (2 * (f / 2.355) ** 2))


def double(r, f1, f2, w):
    return (1 - w) * gauss(r, f1) + w * gauss(r, f2)


def fit(stamps):
    rs = np.concatenate([t[2].ravel() for t in stamps])
    vs = np.concatenate([(t[3] / t[1]).ravel() for t in stamps])
    one, _ = curve_fit(gauss, rs, vs, p0=[2.0])
    two, _ = curve_fit(double, rs, vs, p0=[1.6, 5.0, 0.1], bounds=([0.5, 2, 0], [4, 15, 0.6]))
    share = two[2] * two[1] ** 2 / ((1 - two[2]) * two[0] ** 2 + two[2] * two[1] ** 2)
    print(f"  one Gaussian: FWHM {one[0] * sim.PIX:.3f}\"")
    print(f"  two Gaussians: core FWHM {two[0] * sim.PIX:.3f}\", wing FWHM {two[1] * sim.PIX:.3f}\", wing height {two[2]:.3f}, wing's share of the light {share:.3f}")


split = pd.read_csv(ROOT / "data/split.csv", dtype={"id": str})
stamps = []
for path in split[split.split.isin(["train", "val"])].path:
    x = np.nan_to_num(np.load(ROOT / path).astype(np.float64))
    background, noise = sim.noise_level(x)
    y = (x - background) / noise
    for r, c in zip(*np.nonzero((y == maximum_filter(y, 7)) & (y > 60))):
        if HALF + 1 <= r < sim.N - HALF - 1 and HALF + 1 <= c < sim.N - HALF - 1:
            t = stamp_of(y, r, c)
            if np.percentile(t[3][t[2] > 8], 99) <= 0.05 * t[1]:     # nothing else bright nearby
                stamps.append(t)
sizes = np.array([t[0] for t in stamps])
compact = [t for t in stamps if t[0] <= np.percentile(sizes, 15)]
print(f"Real point-like objects: {len(compact)} of {len(stamps)} candidates")
fit(compact)

rng = np.random.default_rng(0)
simulated = []
for _ in range(len(compact)):
    x0, y0 = rng.uniform(-0.05, 0.05, 2)
    light = np.exp(-((sim.FX - x0) ** 2 + (sim.FY - y0) ** 2) / (2 * 0.003 ** 2))
    image = sim.observe(light)
    simulated.append(stamp_of(image, 50, 50))
print("Simulated point, same measurement:")
fit(simulated)
