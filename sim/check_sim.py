# check_sim.py: does the simulator do what it says, and do its lenses resemble real ones?
#
# Usage (from ~/ruby, with lens_env311):
#   python sim/check_sim.py
#
# Checks:
#   1. Lens equation   our two-plane equation lands light where lenstronomy's multi-plane code does
#   2. Ring radius     a centred source gives rings at theta_e and d * theta_e
#   3. Ring brightness simulated rings measure like real ones, using the same measurement
#   4. Ring thickness  the brightest arcs have the same radial width as real lenses', and sit at the
#                      same radius
#   5. Host galaxies   the picked galaxies measure like real lens galaxies
#   6. Picture         the 4 real compound lenses next to simulated ones (sim/out/check.png)

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Util import param_util

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate as sim
from simulate import FX, FY, HERE, PIX, R, ROOT

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
split, models = sim.load_inputs()
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def stretch(x):
    background, noise = sim.noise_level(x)
    y = np.arcsinh((x - background) / noise)
    lo, hi = np.percentile(y, [1, 99.5])
    return np.clip((y - lo) / (hi - lo + 1e-12), 0, 1)


# ---- 1. our equation vs lenstronomy's full multi-plane code --------------------------------------
print("1. Lens equation, compared with lenstronomy multi-plane")
worst = 0
for zl, zs1, zs2 in [(0.3, 1.0, 2.0), (0.5, 1.2, 3.0), (0.8, 1.5, 2.2), (0.4, 0.9, 4.0)]:
    ratio = lambda z: cosmo.angular_diameter_distance_z1z2(zl, z).value / cosmo.angular_diameter_distance(z).value
    d = ratio(zs2) / ratio(zs1)
    theta_e, inter = 0.9, 0.12
    e1, e2 = param_util.phi_q2_ellipticity(0.4, 0.7)
    kwargs_main = [dict(theta_E=theta_e, e1=e1, e2=e2, center_x=0.03, center_y=-0.02), dict(gamma1=0.03, gamma2=-0.02, ra_0=0, dec_0=0)]
    source1 = dict(x=0.1, y=0.05)
    _, _, b2x, b2y = sim.trace(kwargs_main, d, inter, source1)
    full = LensModel(["SIE", "SHEAR", "SIS"], lens_redshift_list=[zl, zl, zs1], z_source=zs2, multi_plane=True, cosmo=cosmo)
    # lenstronomy's lens strengths are stated for the farthest source, so the main lens gets d * theta_e
    kw = [dict(kwargs_main[0], theta_E=d * theta_e), dict(gamma1=d * 0.03, gamma2=d * -0.02, ra_0=0, dec_0=0),
          dict(theta_E=inter, center_x=0.1, center_y=0.05)]
    fx, fy = full.ray_shooting(FX[::8, ::8].ravel(), FY[::8, ::8].ravel(), kw)
    gap = np.hypot(fx - b2x[::8, ::8].ravel(), fy - b2y[::8, ::8].ravel())
    worst = max(worst, np.percentile(gap, 99))
    wrong = np.hypot(*(np.array(sim.trace(kwargs_main, d + 0.05, inter, source1)[2:]) - np.array([b2x, b2y])))
    print(f"   z = {zl}, {zs1}, {zs2}  d = {d:.2f}  99th percentile gap {np.percentile(gap, 99):.5f}\"  (d off by 0.05 would give {np.median(wrong):.3f}\")")
print(f"   {'PASS' if worst < 0.02 else 'FAIL'} (gap must stay under 0.02\", a fifth of a pixel)")

print("\n   What d do real distances give? (random z_lens 0.3-0.9, z1 up to 3, z2 up to 6)")
ds = []
for _ in range(2000):
    zl = rng.uniform(0.3, 0.9)
    zs1 = rng.uniform(zl + 0.4, 3)
    zs2 = rng.uniform(zs1 + 0.2, 6)
    ratio = lambda z: cosmo.angular_diameter_distance_z1z2(zl, z).value / cosmo.angular_diameter_distance(z).value
    ds.append(ratio(zs2) / ratio(zs1))
print("   d percentiles 5/25/50/75/95:", np.round(np.percentile(ds, [5, 25, 50, 75, 95]), 2))

# ---- 2. ring radius -------------------------------------------------------------------------------
print("\n2. Ring radius, round mass and centred smooth sources")
ok = True
for theta_e, d in [(0.6, 1.2), (0.84, 1.3), (1.2, 1.5), (0.9, 1.8)]:
    kwargs_main = [dict(theta_E=theta_e, e1=0, e2=0, center_x=0, center_y=0), dict(gamma1=0, gamma2=0, ra_0=0, dec_0=0)]
    s1 = dict(x=0, y=0, re=0.05, n=1, q=1, phi=0, clumps=[])
    b1x, b1y, b2x, b2y = sim.trace(kwargs_main, d, 0.0, s1)
    r1 = sim.observe(sim.source_light(s1, b1x, b1y))
    r2 = sim.observe(sim.source_light(s1, b2x, b2y))
    bins = np.arange(0, 45, 1)
    profile = lambda im: np.array([im[(R >= a) & (R < a + 1)].mean() for a in bins]) + 0
    p1, p2 = (bins[np.argmax(profile(im))] + 0.5 for im in (r1, r2))
    good = abs(p1 - theta_e / PIX) <= 1 and abs(p2 - d * theta_e / PIX) <= 1
    ok &= good
    print(f"   theta_e {theta_e}\" d {d}: inner ring at {p1 * PIX:.2f}\" (want {theta_e:.2f}\"), outer at {p2 * PIX:.2f}\" (want {d * theta_e:.2f}\")")
print(f"   {'PASS' if ok else 'FAIL'} (each within 1 pixel)")

# ---- sample simulated lenses on galaxies from the training split ----------------------------------
hosts = split[(split.split == "train") & (split.label == 0)].reset_index(drop=True)
lenses = split[(split.split == "train") & (split.label == 1)]
weight = sim.host_weights(split, hosts)
print(f"\nHost galaxies: {len(hosts)}, picks act like {1 / (weight ** 2).sum():.0f} equally used galaxies")
pairs = [sim.make_pair(np.load(ROOT / hosts.path[i]), rng, models) for i in rng.choice(len(hosts), 400, p=weight)]
pct = [5, 25, 50, 75, 95]

# ---- 3. ring brightness ---------------------------------------------------------------------------
real_level = models.ring_level.values
sim_level = np.array([sim.ring_level(s, y["theta_e"] / PIX) for s, _, y in pairs])
print("\n3. Ring brightness, same measurement (noise units), percentiles 5/25/50/75/95")
print("   real lenses     ", np.round(np.percentile(real_level, pct), 1))
print("   simulated single", np.round(np.percentile(sim_level, pct), 1))

# ---- 4. ring thickness ----------------------------------------------------------------------------
def arc_table(images, radii):
    rows = [row for x, t in zip(images, radii) for row in sim.arc_shape(x, t)]
    return np.array(rows)


real_images = [np.nan_to_num(np.load(ROOT / p).astype(np.float32)) for p in lenses.set_index("id").loc[models.id].path]
real_arcs = arc_table(real_images, models.theta_e)
sim_arcs = arc_table([s for s, _, _ in pairs], [y["theta_e"] for _, _, y in pairs])
print("\n4. Arcs, percentiles 5/25/50/75/95")
print(f"   real lenses      width (\") {np.round(np.percentile(real_arcs[:, 0], pct), 2)}  radius/theta_e {np.round(np.percentile(real_arcs[:, 1], pct), 2)}  ({len(real_arcs)} arcs)")
print(f"   simulated single width (\") {np.round(np.percentile(sim_arcs[:, 0], pct), 2)}  radius/theta_e {np.round(np.percentile(sim_arcs[:, 1], pct), 2)}  ({len(sim_arcs)} arcs)")

# ---- 5. host galaxies -----------------------------------------------------------------------------
names = ["log flux", "concentration", "roundness", "lopsidedness", "log noise"]
all_hosts = np.array([sim.host_features(np.nan_to_num(np.load(ROOT / p).astype(np.float32))) for p in hosts.path])
real_hosts = np.array([sim.host_features(np.nan_to_num(np.load(ROOT / p).astype(np.float32))) for p in lenses.path])
print("\n5. Host galaxies, percentiles 25/50/75: every non-lens, picked with the weights, real lens galaxies")
picked = all_hosts[rng.choice(len(hosts), 5000, p=weight)]
for k, name in enumerate(names):
    q = [25, 50, 75]
    print(f"   {name:14s} {np.round(np.percentile(all_hosts[:, k], q), 2)}  {np.round(np.percentile(picked[:, k], q), 2)}  {np.round(np.percentile(real_hosts[:, k], q), 2)}")

# ---- 6. picture -----------------------------------------------------------------------------------
compound = split[split.notes == "dspl"]
fig, ax = plt.subplots(5, 5, figsize=(11.5, 12))
picks = rng.choice(len(pairs), 15, replace=False)


def show(a, x, title):
    a.imshow(stretch(x), cmap="gray", origin="lower")
    a.set_title(title, fontsize=8)


for j, p in enumerate(compound.path):
    show(ax[0, j], np.nan_to_num(np.load(ROOT / p).astype(np.float32)), "real compound")
ax[0, 4].axis("off")
for j, p in enumerate(lenses.path.sample(5, random_state=3)):
    show(ax[1, j], np.nan_to_num(np.load(ROOT / p).astype(np.float32)), "real ordinary lens")
for j, k in enumerate(picks[:5]):
    single, double, y = pairs[k]
    show(ax[2, j], double, f"sim double: r {y['theta_e']:.2f}\" d {y['d']:.2f} b {y['b']:.1f}")
    show(ax[3, j], single, f"same, single: inner {y['inner']:.0f}")
for j, k in enumerate(picks[5:10]):
    single, double, y = pairs[k]
    show(ax[4, j], double, f"sim double: r {y['theta_e']:.2f}\" d {y['d']:.2f} b {y['b']:.1f}")
for a in ax.ravel():
    a.axis("off")
fig.tight_layout()
fig.savefig(OUT / "check.png", dpi=110)
print("\nSaved sim/out/check.png")
