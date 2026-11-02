# simulate.py: realistic simulated lenses on real Euclid galaxies, with one or two background sources.
#
# Usage (from ~/ruby, with lens_env311):
#   python sim/simulate.py --out sim/out/v2 --train 3000 --val 600 --test 600
#
# What makes these more realistic than the painted rings in polar/:
#   the rings come from real lensing   light is ray-traced through a mass model, so arcs, quads and
#                                      uneven rings appear on their own
#   real lens shapes                   size, oval shape and shear are drawn together from one of
#                                      Euclid's published lens models (sim/lens_models.csv)
#   real brightness                    inner ring brightness is drawn from what real lenses show
#   lumpy sources                      a smooth galaxy body plus bright clumps, not a perfect band
#                                      (--real-shapes uses real galaxy shapes instead; they scored worse
#                                      in sim/realism_check.py, so they aren't the default)
#   two source planes                  the outer source is bent by the main lens more strongly, and
#                                      also by the mass of the inner source
#   real pixels                        Euclid's 0.1" pixels and the blur measured in our own cutouts,
#                                      added to real non-lens galaxies, so noise and neighbours are real
#   hosts that look like lens hosts    spirals and clumpy galaxies are left out, and the rest are
#                                      picked to match real lens galaxies in brightness, concentration,
#                                      roundness and lopsidedness; the mass lines up with their light
#
# The lens equation for two source planes (Schneider et al. 1992), angles in arcsec:
#   inner source:  b1 = t - a(t)
#   outer source:  b2 = t - d * a(t) - a_s(b1)
# a is the main lens's deflection, a_s the inner source's own, and d = outer ring radius / inner
# ring radius, which depends only on the distances. sim/check_sim.py checks this against
# lenstronomy's full multi-plane code.
#
# Pairs: every "double" has a "single" made from the same galaxy, same mass, same inner source.
# So the only difference between the two is the outer source.
#
# Not modelled: extra Poisson noise from the arcs (small next to the sky noise for faint arcs).
#
# Realism: sim/realism_check.py can still tell these from real lenses (retrained Zoobot AUC 0.90-0.91,
# against a pass mark of below 0.90, and 0.996 for painted rings). See sim/out/realism/.
#
# Galaxies stay in their split, and lens models come from training lenses only, so nothing about
# the validation or test lenses (including the 4 compound lenses) reaches the simulator.
# Nothing is sent anywhere.
#
# Output (in --out):
#   <split>.npz          single and double, raw pixel values, (pairs, 100, 100)
#   <split>_params.csv   every simulated value for each pair

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Util import param_util
from scipy.ndimage import gaussian_filter, label, map_coordinates
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path.home() / "ruby"
HERE = Path(__file__).resolve().parent
PIX, N = 0.1, 100              # Euclid VIS pixels (arcsec), cutout size
PSF_CORE, PSF_WING = 0.15, 0.36     # blur FWHMs (arcsec), set so a simulated point measures like real ones in
PSF_WING_SHARE = 0.50          # sim/measure_blur.py. This is the share of the light in the wide part
SS = 4                         # each pixel is worked out on a 4x4 finer grid, then averaged
SOURCE_SIZE = 0.05             # median half-light radius of a source (arcsec), set by check_sim.py's check 4
PLACE = 0.3                    # sources land within this fraction of the ring radius from the lens axis
MAX_ZERO = 0.005               # cutouts with more empty pixels than this have blocks missing
MAX_STRUCTURE = 0.05           # galaxies with more fine structure than this are spirals or clumpy, not lens hosts
CLUMPS = 6                     # up to this many bright clumps in a source
CLUMP_AMP = (1, 6)             # clump brightness, relative to the source's half-light brightness
MIN_SOURCE_STRUCTURE = 0.03    # galaxies used as source shapes must have at least this much fine structure
BANK = None                    # real galaxy shapes for the split being simulated (source_bank)
HOST_VOTES = 5                 # how many similar galaxies each real lens galaxy votes for
HOST_BY_ZOOBOT = False         # pick galaxies by sim/score_hosts.py's odds instead of by votes
NOISE_FLOOR = 2.72             # ring_level of pure noise
INNER_SCALE = 0.8              # peak_level reads brighter than ring_level (set by check_sim.py's check 3)

fine = ((np.arange(N * SS) + 0.5) / SS - N / 2) * PIX
FX, FY = np.meshgrid(fine, fine)                      # arcsec from the image centre (pixel 49.5)
yy, xx = np.mgrid[:N, :N] - 49.5
R = np.hypot(xx, yy)
MAIN = LensModel(["SIE", "SHEAR"])


def noise_level(x):
    background = np.median(x)
    return background, 1.4826 * np.median(np.abs(x - background)) + 1e-12


def ring_level(x, radius_px):
    """How bright the ring at this radius is, in noise units: the 98th percentile of light above the
    usual level at each radius (the lower quarter, so a full ring doesn't subtract itself)."""
    background, noise = noise_level(x)
    y = (x - background) / noise
    idx = R.astype(int)
    low = np.array([np.percentile(y[idx == i], 25) for i in range(idx.max() + 1)])
    return np.percentile((y - low[idx])[np.abs(R - radius_px) < 2], 98)


def host_angle(x):
    """Direction of a galaxy's long axis, from the second moments of its bright pixels."""
    background, noise = noise_level(x)
    y = np.clip((x - background) / noise, 0, None) * (R < 15) * ((x - background) / noise > 3)
    if y.sum() == 0:
        return 0.0
    mx, my = (y * xx).sum() / y.sum(), (y * yy).sum() / y.sum()
    cxx = (y * (xx - mx) ** 2).sum()
    cyy = (y * (yy - my) ** 2).sum()
    cxy = (y * (xx - mx) * (yy - my)).sum()
    return 0.5 * np.arctan2(2 * cxy, cxx - cyy)


def source_bank(split, part):
    """Real galaxy shapes to use as background sources: the bright, structured galaxies of one split, cut out of
    their cutouts. (The spirals and clumpy galaxies left out as lens hosts look most like distant sources.)"""
    bank = []
    for path in split[(split.split == part) & (split.label == 0)].path:
        x = np.nan_to_num(np.load(ROOT / path).astype(np.float32))
        if (x == 0).mean() >= MAX_ZERO or host_structure(x) < MIN_SOURCE_STRUCTURE:
            continue
        background, noise = noise_level(x)
        y = gaussian_filter((x - background) / noise, 0.5)
        labels, _ = label(y > 2.5)
        central = labels[N // 2, N // 2]
        mask = labels == central
        if central == 0 or y.max() < 30 or not 60 <= mask.sum() <= 2500 or mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
            continue
        light = y * mask
        total = light.sum()
        cy, cx = (light * (yy + 49.5)).sum() / total, (light * (xx + 49.5)).sum() / total
        order = np.argsort(np.hypot(yy + 49.5 - cy, xx + 49.5 - cx), axis=None)
        half = np.hypot(yy + 49.5 - cy, xx + 49.5 - cx).ravel()[order][np.searchsorted(np.cumsum(light.ravel()[order]), total / 2)]
        if half >= 2:                                  # smaller than this is a star or barely resolved
            bank.append(dict(pixels=light / light.max(), cy=cy, cx=cx, half=half))
    return bank


def draw_source(rng, x, y):
    """A background galaxy at (x, y): a real galaxy's shape if there is a bank of them, or else a smooth Sersic
    body plus a few bright clumps."""
    re = np.clip(rng.lognormal(np.log(SOURCE_SIZE), 0.4), 0.03, 0.4)
    if BANK:
        real = BANK[rng.integers(len(BANK))]
        return dict(x=x, y=y, re=re, phi=rng.uniform(0, 2 * np.pi), flip=rng.choice([-1, 1]), real=real)
    clumps = [(x + re * rng.normal(), y + re * rng.normal(), rng.uniform(0.02, 0.06), rng.uniform(*CLUMP_AMP))
              for _ in range(rng.integers(0, CLUMPS + 1))]
    return dict(x=x, y=y, re=re, n=rng.uniform(0.7, 2.5), q=rng.uniform(0.4, 1), phi=rng.uniform(0, np.pi), clumps=clumps)


def source_light(s, bx, by):
    c, si = np.cos(s["phi"]), np.sin(s["phi"])
    u, v = c * (bx - s["x"]) + si * (by - s["y"]), -si * (bx - s["x"]) + c * (by - s["y"])
    if "real" in s:
        real, scale = s["real"], s["real"]["half"] / s["re"]   # stamp pixels per arcsec
        return map_coordinates(real["pixels"], [real["cy"] + scale * v, real["cx"] + s["flip"] * scale * u], order=1, cval=0.0)
    r = np.maximum(np.hypot(u * np.sqrt(s["q"]), v / np.sqrt(s["q"])), 1e-3)
    bn = 1.9992 * s["n"] - 0.3271
    light = np.exp(-bn * ((r / s["re"]) ** (1 / s["n"]) - 1))
    for cx, cy, width, amp in s["clumps"]:
        light = light + amp * np.exp(-((bx - cx) ** 2 + (by - cy) ** 2) / (2 * width ** 2))
    return light


def observe(light):
    """Fine-grid light -> what Euclid records: blurred, then averaged into 0.1" pixels."""
    sigma = lambda fwhm: fwhm / 2.355 / PIX * SS
    blurred = (1 - PSF_WING_SHARE) * gaussian_filter(light, sigma(PSF_CORE), truncate=4) \
        + PSF_WING_SHARE * gaussian_filter(light, sigma(PSF_WING), truncate=4)
    return blurred.reshape(N, SS, N, SS).mean((1, 3))


def peak_level(image):
    """Brightness of the bright part of a lensed source (near the peak, ignoring the single top pixel)."""
    return np.percentile(image[image > 0.3 * image.max()], 90)


def trace(kwargs_main, d, inter, source1):
    """The lens equation above: where each fine-grid point lands in each source plane."""
    ax, ay = MAIN.alpha(FX, FY, kwargs_main)
    b1x, b1y = FX - ax, FY - ay
    dx, dy = b1x - source1["x"], b1y - source1["y"]
    r = np.maximum(np.hypot(dx, dy), 1e-4)
    return b1x, b1y, FX - d * ax - inter * dx / r, FY - d * ay - inter * dy / r


def draw_system(rng, models, angle=0.0, theta_e=None, q=None, d=None, b=None, inner=None, companion=None):
    """Draw every value for one lens. Anything passed in is fixed, for building detection maps."""
    row = models.iloc[rng.integers(len(models))]
    theta_e = row.theta_e if theta_e is None else theta_e
    q = row.q if q is None else q
    phi = angle + rng.normal(0, np.radians(20))        # mass follows the galaxy's light, roughly
    e1, e2 = param_util.phi_q2_ellipticity(phi, q)
    shear = rng.uniform(0, np.pi)
    kwargs_main = [dict(theta_E=theta_e, e1=e1, e2=e2, center_x=row.offset_x, center_y=row.offset_y),
                   dict(gamma1=row.gamma * np.cos(2 * shear), gamma2=row.gamma * np.sin(2 * shear), ra_0=0, dec_0=0)]

    def place(scale):                                  # uniform over the area that makes strong images
        r, t = scale * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        return row.offset_x + r * np.cos(t), row.offset_y + r * np.sin(t)

    source1 = draw_source(rng, *place(PLACE * theta_e))
    d = rng.uniform(1.1, 1.8) if d is None else d
    source2 = draw_source(rng, *place(PLACE * d * theta_e))
    companion = rng.random() < 0.25 if companion is None else companion
    extra = []
    if companion:                                      # a neighbour at the inner source's distance: extra
        t = rng.uniform(0, 2 * np.pi)                  # arcs at the SAME radius, so extra light alone
        gap = rng.uniform(0.3, 1.0)                    # doesn't mean a second ring
        extra = [draw_source(rng, source1["x"] + gap * np.cos(t), source1["y"] + gap * np.sin(t))]
    return dict(kwargs_main=kwargs_main, theta_e=theta_e, q=q, d=d, source1=source1, source2=source2, extra=extra,
                inter=rng.uniform(0.02, 0.2),          # inner source's own mass (Einstein radius, arcsec)
                inner=max(1.0, INNER_SCALE * np.sqrt(max(row.ring_level ** 2 - NOISE_FLOOR ** 2, 0))) if inner is None else inner,
                b=np.exp(rng.uniform(np.log(0.5), np.log(6))) if b is None else b)


def lensed_light(system):
    """Unit-brightness images of the inner sources and of the outer source, both in Euclid pixels."""
    b1x, b1y, b2x, b2y = trace(system["kwargs_main"], system["d"], system["inter"], system["source1"])
    inner = sum(source_light(s, b1x, b1y) for s in [system["source1"]] + system["extra"])
    return observe(inner), observe(source_light(system["source2"], b2x, b2y))


def make_pair(raw, rng, models, **fixed):
    """A real galaxy -> (single, double, the values used). Single and double differ only by the outer source."""
    x = np.nan_to_num(raw.astype(np.float32))
    _, noise = noise_level(x)
    system = draw_system(rng, models, angle=host_angle(x), **fixed)
    inner, outer = lensed_light(system)
    single = x + (system["inner"] * noise / peak_level(inner)) * inner
    double = single + (system["b"] * noise / peak_level(outer)) * outer
    return single.astype(np.float32), double.astype(np.float32), system


def arc_shape(x, theta_e):
    """Width (arcsec) and radius (in theta_e) of the three brightest 30-degree stretches of arc."""
    background, noise = noise_level(x)
    y = (x - background) / noise
    idx = R.astype(int)
    low = np.array([np.percentile(y[idx == i], 25) for i in range(idx.max() + 1)])
    res = y - low[idx]
    angles = np.radians(np.arange(0, 360, 3))
    radii = np.arange(0.5 * theta_e / PIX, 1.6 * theta_e / PIX, 0.25)
    strip = map_coordinates(res, [49.5 + np.outer(radii, np.sin(angles)), 49.5 + np.outer(radii, np.cos(angles))], order=1)
    sectors = strip.reshape(len(radii), 12, 10).mean(2)
    found = []
    for k in np.argsort(sectors.max(0))[-3:]:
        c = sectors[:, k]
        peak = c.argmax()
        if c[peak] < 6 or peak in (0, len(c) - 1):
            continue
        lo, hi = peak, peak
        while lo > 0 and c[lo - 1] > c[peak] / 2:
            lo -= 1
        while hi < len(c) - 1 and c[hi + 1] > c[peak] / 2:
            hi += 1
        if lo > 0 and hi < len(c) - 1:                 # the half-height point must be in view on both sides
            found.append(((hi - lo + 1) * 0.25 * PIX, radii[peak] * PIX / theta_e))
    return found


def host_features(x):
    """Brightness, concentration, roundness and lopsidedness of a galaxy, and the image's noise."""
    background, noise = noise_level(x)
    y = (x - background) / noise
    flux = max(y[R < 15].sum(), 1)
    core = np.clip(y, 0, None) * (R < 6)               # the middle only, inside most rings
    cxx, cyy, cxy = (core * xx ** 2).sum(), (core * yy ** 2).sum(), (core * xx * yy).sum()
    spread = np.sqrt((cxx - cyy) ** 2 + 4 * cxy ** 2)
    return [np.log10(flux), y[R < 5].sum() / flux, (cxx + cyy - spread) / max(cxx + cyy + spread, 1),
            np.abs(core - core[::-1, ::-1]).sum() / max(core.sum(), 1), np.log10(noise)]


def host_structure(x):
    """Share of pixels outside the core with fine structure: high for spiral arms and clumpy discs."""
    background, noise = noise_level(x)
    y = (x - background) / noise
    return ((gaussian_filter(y, 1) - gaussian_filter(y, 4))[(R > 4) & (R < 25)] > 1.5).mean()


def host_weights(split, hosts):
    """Chance of picking each galaxy. Broken cutouts, spirals and clumpy galaxies are out. Then every real training lens galaxy
    votes for the 5 remaining galaxies most like it, so the picks match real lens galaxies on all the measures
    at once. (Real lens galaxies can't be screened for structure themselves, because their arcs count as structure.)"""
    def features(frame):
        return np.array([host_features(np.nan_to_num(np.load(ROOT / p).astype(np.float32))) for p in frame.path])
    real, candidates = features(split[(split.split == "train") & (split.label == 1)]), features(hosts)
    def usable(path):
        x = np.nan_to_num(np.load(ROOT / path).astype(np.float32))
        return (x == 0).mean() < MAX_ZERO and host_structure(x) < MAX_STRUCTURE
    smooth = np.flatnonzero([usable(p) for p in hosts.path])
    if HOST_BY_ZOOBOT:
        odds = pd.read_csv(HERE / "host_scores.csv", dtype={"id": str}).set_index("id").odds.loc[hosts.id].values
        weight = np.zeros(len(hosts))
        weight[smooth] = np.minimum(odds[smooth], np.percentile(odds[smooth], 99))
        return weight / weight.sum()
    scale = StandardScaler().fit(candidates[smooth])
    near = NearestNeighbors(n_neighbors=HOST_VOTES).fit(scale.transform(candidates[smooth]))
    votes = np.bincount(smooth[near.kneighbors(scale.transform(real))[1].ravel()], minlength=len(hosts)).astype(float)
    return votes / votes.sum()


def load_inputs():
    split = pd.read_csv(ROOT / "data/split.csv", dtype={"id": str})
    models = pd.read_csv(HERE / "lens_models.csv")
    return split, models


def flat_params(system, host_id):
    s1, s2 = system["source1"], system["source2"]
    return dict(host=host_id, theta_e=system["theta_e"], q=system["q"], d=system["d"], inner=system["inner"], b=system["b"],
                inter=system["inter"], companion=len(system["extra"]) > 0, re1=s1["re"], re2=s2["re"],
                offset1=np.hypot(s1["x"], s1["y"]) / system["theta_e"],
                offset2=np.hypot(s2["x"], s2["y"]) / (system["d"] * system["theta_e"]))


def build(split_name, n_pairs, split, models, rng, out, real_shapes=False):
    global BANK
    BANK = source_bank(split, split_name) if real_shapes else None
    hosts = split[(split.split == split_name) & (split.label == 0)].reset_index(drop=True)
    weight = host_weights(split, hosts)
    print(f"{split_name}: {n_pairs} pairs from {len(hosts)} galaxies "
          f"(the picks act like {1 / (weight ** 2).sum():.0f} equally used galaxies), {len(BANK or [])} source shapes", flush=True)
    singles, doubles, params = [], [], []
    start = time.time()
    for k, i in enumerate(rng.choice(len(hosts), n_pairs, p=weight)):
        single, double, system = make_pair(np.load(ROOT / hosts.path[i]), rng, models)
        singles.append(single), doubles.append(double), params.append(flat_params(system, hosts.id[i]))
        if (k + 1) % 500 == 0:
            print(f"  {k + 1} pairs, {time.time() - start:.0f}s", flush=True)
    np.savez(out / f"{split_name}.npz", single=np.stack(singles), double=np.stack(doubles))
    pd.DataFrame(params).to_csv(out / f"{split_name}_params.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sim/out/v2")
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--val", type=int, default=600)
    ap.add_argument("--test", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--real-shapes", action="store_true")    # real galaxy shapes as sources (v3)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    split, models = load_inputs()
    rng = np.random.default_rng(args.seed)
    for name, n in [("train", args.train), ("val", args.val), ("test", args.test)]:
        build(name, n, split, models, rng, out, real_shapes=args.real_shapes)
    print("Saved to", args.out)
