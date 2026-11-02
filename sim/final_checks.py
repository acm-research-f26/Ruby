# final_checks.py: follow-up checks on Step 4's real compound lens result (mean rank 3.9 of 79). Chosen after seeing
# the result, so these explain it and can't change the verdict in sim/out/final/results.txt.
#
# Usage (from ~/ruby, with lens_env311):
#   python -I sim/final_checks.py
#
#   1. Are the 4 compound cutouts different files? (shape, noise, zero pixels)
#   2. Do simple measures rank them as high? If so, the score may only mean "bright, obvious lens".
#      Then: where do they land once those measures are taken out of the score?
#   3. Does the score fall when the outer light goes? Everything beyond a radius R is replaced with sky and noise.
#   4. Which part of the image does the score depend on? Small patches are replaced with sky and noise, one at a time.
#
# Output (in sim/out/final/checks/): checks.txt, radius.csv, top_lenses.png, occlusion.png

import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate as sim
from simulate import ROOT

NAMES = {"107505": "Galileo's Lens", "726389": "Cosmic Ammonite", "469177": "Cosmic Dartboard", "378903": "Teapot Lens"}
SEEDS = [1, 2, 3]
device = "mps" if torch.backends.mps.is_available() else "cpu"
out_dir = ROOT / "sim/out/final/checks"
out_dir.mkdir(parents=True, exist_ok=True)

report = []
def say(line=""):
    print(line, flush=True)
    report.append(line)

def preprocess(pixels):
    x = np.nan_to_num(pixels.astype(np.float32))
    background = np.median(x)
    noise = 1.4826 * np.median(np.abs(x - background)) + 1e-12
    x = np.arcsinh((x - background) / noise)
    lo, hi = np.percentile(x, [1, 99.5])
    return np.clip((x - lo) / (hi - lo + 1e-12), 0, 1).astype(np.float32)

def to_tensor(images):
    return torch.from_numpy(np.stack([preprocess(x) for x in images]))[:, None]

def zoobot(num_classes, path):
    net = timm.create_model("convnext_nano", pretrained=False, num_classes=num_classes, in_chans=1)
    state = torch.load(path, map_location="cpu")
    net.load_state_dict({k.removeprefix("net."): v for k, v in state.items()})
    return net.to(device).eval()

@torch.no_grad()
def softmax(net, x):
    out = [torch.softmax(net(F.interpolate(x[i:i + 64].to(device), size=(224, 224), mode="bilinear", align_corners=False)), 1).cpu()
           for i in range(0, len(x), 64)]
    return torch.cat(out).numpy()

nets = [zoobot(3, ROOT / f"sim/out/models2/seed{k}/zoobot3.pt") for k in SEEDS]
def p_compound(x):
    return np.mean([softmax(net, x)[:, 2] for net in nets], axis=0)

split = pd.read_csv(ROOT / "data/split.csv", dtype={"id": str})
lenses = split[(split.split == "test") & (split.label == 1)].reset_index(drop=True)
compound = lenses.notes.eq("dspl").values
raw = [np.load(ROOT / p) for p in lenses.path]
clean = [np.nan_to_num(x.astype(np.float32)) for x in raw]
name_of = lambda i: NAMES[lenses.id[i][-6:]]
which = np.flatnonzero(compound)

# ---- 1. Are the files different? ------------------------------------------------------------
say("1. The files")
say(f"  shapes {sorted({x.shape for x in raw})}, types {sorted({str(x.dtype) for x in raw})}")
noise = np.array([sim.noise_level(x)[1] for x in clean])
zeros = np.array([(x == 0).mean() for x in clean])
nans = np.array([np.isnan(x.astype(np.float32)).mean() for x in raw])
say(f"  noise, all 79, 10/50/90%: {np.percentile(noise, [10, 50, 90]).round(4)}; zero pixels above 0: {(zeros > 0).sum()}; missing pixels above 0: {(nans > 0).sum()}")
for i in which:
    say(f"  {name_of(i)}: {lenses.path[i]}, grade {lenses.grade[i]}, noise {noise[i]:.4f}, zero pixels {zeros[i]:.3f}")
say(f"  folders: {lenses.path.map(lambda p: str(Path(p).parent)).value_counts().to_dict()}")
say(f"  grades, ordinary lenses: {lenses.grade[~compound].value_counts().to_dict()}")

# ---- 2. Simple measures ----------------------------------------------------------------------
scores = pd.read_csv(ROOT / "sim/out/final/ranks.csv", dtype={"id": str})
scores = scores[scores.model == "zoobot3"].groupby("id").score.mean().reindex(lenses.id).values

def measures(x):
    background, n = sim.noise_level(x)
    y = (x - background) / n
    yy, xx = np.mgrid[:100, :100]
    r = np.hypot(yy - 49.5, xx - 49.5)
    bright = y > 3
    ring_zone = (r > 5) & (r < 35)
    return dict(pixels_above_3=bright.sum(), light_outside_centre=np.clip(y[ring_zone], 0, None).sum(),
                peak=y.max(), extent=np.percentile(r[bright], 90) if bright.any() else 0,
                far_light=np.clip(y[(r > 15) & (r < 35)], 0, None).sum(), brightness_only=preprocess(x).mean())
simple = pd.DataFrame([measures(x) for x in clean])
baseline = np.mean([softmax(zoobot(2, ROOT / f"runs/zoobot_lr0.0001_seed{k}/best.pt"), to_tensor(clean))[:, 1] for k in SEEDS], axis=0)
simple["baseline_p_lens"] = baseline

say("\n2. Simple measures (rank 1 = highest). Correlation is with the mean compound score, on the 75 ordinary lenses")
say(f"  compound score: ranks {[int(k) for k in pd.Series(-scores).rank()[compound]]}, mean {pd.Series(-scores).rank()[compound].mean():.1f}")
for column in simple:
    rank = simple[column].rank(ascending=False)
    rho = spearmanr(simple[column][~compound], scores[~compound]).statistic
    say(f"  {column:22s} ranks {[int(k) for k in rank[compound]]}, mean {rank[compound].mean():5.1f}; correlation {rho:+.2f}")

logit = np.log(np.clip(scores, 1e-4, 1 - 1e-4) / (1 - np.clip(scores, 1e-4, 1 - 1e-4)))
inputs = np.log1p(simple[["pixels_above_3", "light_outside_centre", "peak", "extent", "far_light"]].values)
inputs = np.c_[inputs, simple.brightness_only.values, simple.baseline_p_lens.values]
fit = LinearRegression().fit(inputs[~compound], logit[~compound])
left = logit - fit.predict(inputs)
rank = pd.Series(-left).rank()
say(f"  simple measures explain {fit.score(inputs[~compound], logit[~compound]):.0%} of the score on ordinary lenses")
say(f"  ranks with them taken out: {[int(k) for k in rank[compound]]}, mean {rank[compound].mean():.1f}")

count = sum(1 for c in combinations(range(1, 26), 4) if sum(c) / 4 <= 3.9)
say(f"  chance of 4 random lenses of 79 averaging 3.9 or better: {count} in {sum(1 for _ in combinations(range(1, 80), 4))}")

# ---- A look at the top of the list -------------------------------------------------------------
order = np.argsort(-scores)[:24]
fig, axes = plt.subplots(4, 6, figsize=(12, 8.4))
for ax, i in zip(axes.ravel(), order):
    ax.imshow(preprocess(clean[i]), cmap="gray", origin="lower")
    ax.set_title(f"{scores[i]:.2f} " + (name_of(i) if compound[i] else "ordinary"), fontsize=8, fontweight="bold" if compound[i] else "normal")
    ax.axis("off")
fig.tight_layout()
fig.savefig(out_dir / "top_lenses.png", dpi=130)
plt.close(fig)

# ---- 3. Take the outer light away ---------------------------------------------------------------
yy, xx = np.mgrid[:100, :100]
R = np.hypot(yy - 49.5, xx - 49.5)
RADII = np.arange(0.5, 5.01, 0.25)                    # arcsec

def with_sky(x, mask, rng):
    background, n = sim.noise_level(x)
    y = x.copy()
    y[mask] = rng.normal(background, n, mask.sum())
    return y

ordinary_high = np.argsort(-np.where(compound, -1, scores))[:12]
ordinary_rest = np.flatnonzero(~compound)
rows = []
for i in list(which) + list(ordinary_high) + list(ordinary_rest):
    rng = np.random.default_rng(int(i))
    images = [with_sky(clean[i], R > radius / 0.1, rng) for radius in RADII]
    for radius, p in zip(RADII, p_compound(to_tensor(images))):
        rows.append(dict(i=int(i), kind="compound" if compound[i] else "ordinary", name=name_of(i) if compound[i] else "", radius=radius, p=p, full=scores[i]))
curves = pd.DataFrame(rows).drop_duplicates(["i", "radius"])
curves.to_csv(out_dir / "radius.csv", index=False)
say("\n3. Compound score with everything beyond a radius replaced by sky and noise (radius in arcsec)")
for i in which:
    c = curves[curves.i == i]
    say(f"  {name_of(i):17s} full {scores[i]:.2f}: " + " ".join(f"{r:.2g}:{p:.2f}" for r, p in zip(c.radius, c.p)))
top = curves[curves.i.isin(ordinary_high)].groupby("radius").p.mean()
rest = curves[curves.kind == "ordinary"].groupby("radius").p.mean()
say(f"  top 12 ordinary, mean, full {scores[ordinary_high].mean():.2f}: " + " ".join(f"{r:.2g}:{p:.2f}" for r, p in top.items()))
say(f"  all 75 ordinary, mean, full {scores[~compound].mean():.2f}: " + " ".join(f"{r:.2g}:{p:.2f}" for r, p in rest.items()))

# ---- 4. Which part of the image matters? ----------------------------------------------------------
SIZE, STEP = 7, 3
def occlusion(x, rng):
    corners = [(a, c) for a in range(0, 100 - SIZE + 1, STEP) for c in range(0, 100 - SIZE + 1, STEP)]
    images = []
    for a, c in corners:
        mask = np.zeros_like(x, dtype=bool)
        mask[a:a + SIZE, c:c + SIZE] = True
        images.append(with_sky(x, mask, rng))
    p = p_compound(to_tensor(images))
    drop, hits = np.zeros_like(x), np.zeros_like(x)
    for (a, c), v in zip(corners, p):
        drop[a:a + SIZE, c:c + SIZE] += v
        hits[a:a + SIZE, c:c + SIZE] += 1
    return drop / hits

show = list(which) + list(ordinary_high[:2])
fig, axes = plt.subplots(2, len(show), figsize=(2.6 * len(show), 5.4))
say("\n4. Largest fall in the score when one 0.7\" patch is hidden")
for col, i in enumerate(show):
    full = p_compound(to_tensor([clean[i]]))[0]
    change = full - occlusion(clean[i], np.random.default_rng(int(i)))
    peak = np.unravel_index(change.argmax(), change.shape)
    distance = np.hypot(peak[0] - 49.5, peak[1] - 49.5) * 0.1
    label = name_of(i) if compound[i] else "ordinary"
    say(f"  {label:17s} full {full:.2f}, largest fall {change.max():.2f}, {distance:.1f}\" from the centre")
    axes[0, col].imshow(preprocess(clean[i]), cmap="gray", origin="lower")
    axes[0, col].set_title(label, fontsize=9)
    limit = max(abs(change).max(), 1e-3)
    axes[1, col].imshow(change, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="lower")
    axes[1, col].set_title(f"fall in score (max {change.max():.2f})", fontsize=8)
    for row in (0, 1):
        axes[row, col].axis("off")
fig.tight_layout()
fig.savefig(out_dir / "occlusion.png", dpi=130)
plt.close(fig)

(out_dir / "checks.txt").write_text("\n".join(report) + "\n")
print("Done.", flush=True)
