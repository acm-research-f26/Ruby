# probe_filters.py: name filters by testing them on controlled images
#
# Usage (from ~/ruby, with lens_env311):
#   python nesy/probe_filters.py
#
# Looking at a filter's strongest images (name_filters.py) is hard here: every lens looks
# like a fuzzy blob, and the filter's 7x7 grid is too coarse to show a ring. So instead we
# ask each filter direct questions. Take 200 real non-lens galaxies (Zoobot says "not a
# lens"), paint ONE thing onto a copy of each, and see which filters switch on:
#   ring        a full Einstein ring (radius 0.84", the typical Euclid lens)
#   faint ring  the same ring at half the brightness
#   arcs        two opposite arcs on that ring (the more common look of a lens)
#   compound    the ring plus a fainter second ring at 1.5x the radius
#   outer only  just that fainter outer ring, no inner ring (control for compound)
#   bright ring one inner ring carrying as much light as both compound rings together
#               (control: is it two rings, or just more ring light?)
#   companion   a small galaxy 2.5" away (a lookalike that is NOT a lens)
#   brighter    a brighter, more compact centre (tests "it just likes bright centres")
# A filter that switches on for ring and arcs but not for companion or brighter is
# detecting lens arcs. A filter that is ON for compound but not for ring, outer only or
# bright ring is responding to the two rings together, not just to more ring light.
#
# The painted shapes are simple (Gaussian profiles at Euclid's resolution), not physical
# lens models, so read the results as "what kind of shape does this filter respond to".
#
# Output (in nesy/out/<run>_<measure>/probes/):
#   examples.png        one galaxy with each painted version, to see what was added
#   probe_results.csv   per filter (all 640): % of the 200 galaxies where it is ON, before
#                       and after each probe

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent   # ~/ruby

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="nesy/out/zoobot_lr0.0001_seed1_norm")   # written by extract.py
parser.add_argument("--n", type=int, default=200)       # how many non-lens galaxies to paint on
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

IMG = 224
device = "mps" if torch.backends.mps.is_available() else "cpu"
data_dir = ROOT / args.dir
run_name, measure = data_dir.name.rsplit("_", 1)
out_dir = data_dir / "probes"
os.makedirs(out_dir, exist_ok=True)
rng = np.random.default_rng(args.seed)

# ---- 1. Same preprocessing as train.py and extract.py ------------------------------
def preprocess(pixels):
    """Raw VIS pixels -> 0-1 image with faint arcs brought out (arcsinh stretch)."""
    x = np.nan_to_num(pixels.astype(np.float32))
    background = np.median(x)
    noise = 1.4826 * np.median(np.abs(x - background)) + 1e-12
    x = np.arcsinh((x - background) / noise)
    lo, hi = np.percentile(x, [1, 99.5])
    return np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)

def to_model_input(batch):
    """100x100 -> 224x224 (what the pretrained model expects)."""
    return F.interpolate(batch.to(device), size=(IMG, IMG), mode="bilinear", align_corners=False)

# ---- 2. The shapes we paint (pixels are 0.1", cutouts are centred on the galaxy) ------
yy, xx = np.mgrid[:100, :100] - 49.5
rr, angle = np.hypot(xx, yy), np.arctan2(yy, xx)
R = 8.4          # Einstein radius in pixels (0.84")
WIDTH = 1.0      # ring thickness in pixels, about Euclid's blur

def ring(radius):
    return np.exp(-(rr - radius) ** 2 / (2 * WIDTH ** 2))

def near(a, centre, half_width):
    return np.abs(np.angle(np.exp(1j * (a - centre)))) < half_width

def paint(raw, probe):
    """Add one shape to a raw image. Brightness is set in units of that image's noise."""
    x = np.nan_to_num(raw.astype(np.float32)).copy()
    noise = 1.4826 * np.median(np.abs(x - np.median(x)))
    if probe == "ring":
        x += 6 * noise * ring(R)
    elif probe == "faint ring":
        x += 3 * noise * ring(R)
    elif probe == "arcs":
        t = rng.uniform(0, np.pi)
        x += 6 * noise * ring(R) * (near(angle, t, np.radians(50)) | near(angle, t + np.pi, np.radians(50)))
    elif probe == "compound":
        x += 6 * noise * ring(R) + 3 * noise * ring(1.5 * R)
    elif probe == "outer only":
        x += 3 * noise * ring(1.5 * R)
    elif probe == "bright ring":
        x += 10.5 * noise * ring(R)    # 6 + 3 x 1.5 (the outer ring is 1.5x longer)
    elif probe == "companion":
        t = rng.uniform(0, 2 * np.pi)
        x += 6 * noise * np.exp(-((xx - 25 * np.cos(t)) ** 2 + (yy - 25 * np.sin(t)) ** 2) / (2 * 1.5 ** 2))
    elif probe == "brighter":
        peak = x[45:55, 45:55].max() - np.median(x)
        x += 0.5 * peak * np.exp(-rr ** 2 / (2 * 1.2 ** 2))
    return x

PROBES = ["original", "ring", "faint ring", "arcs", "compound", "outer only", "bright ring",
          "companion", "brighter"]

# ---- 3. Galaxies to paint on, the trained Zoobot, and the ON/OFF thresholds ---------
images = pd.read_csv(data_dir / "images.csv")
base = images[(images["split"] == "train") & (images["label"] == 0) & (images["zoobot_score"] < 0.5)]
base = base.iloc[rng.choice(len(base), size=min(args.n, len(base)), replace=False)]
raws = [np.load(ROOT / p) for p in base["path"]]
# example for the picture: a compact galaxy (so a painted ring is visible outside its glow)
# whose centre is the brightest thing in the cutout (no bright star nearby)
def example_score(r):
    r = np.nan_to_num(r.astype(np.float32))
    bg = np.median(r)
    noise = 1.4826 * np.median(np.abs(r - bg)) + 1e-12
    centre = (r[45:55, 45:55].max() - bg) / noise
    at_ring = (np.median(r[np.abs(rr - R) < 1]) - bg) / noise
    clean = r[45:55, 45:55].max() >= r.max()
    return centre if (clean and at_ring < 1.5) else -1
example = int(np.argmax([example_score(r) for r in raws]))
threshold = np.load(data_dir / "strengths.npz", allow_pickle=True)["threshold"]

in_rules = {}
for f in sorted(data_dir.glob("rules_*.json")):
    for line in json.loads(f.read_text())["asp"]:
        for k in re.findall(r"\b(k\d+)\(", line):
            in_rules.setdefault(k, set()).add(f.stem[len("rules_"):])
rule_filters = sorted(in_rules, key=lambda k: (-len(in_rules[k]), int(k[1:])))

model = timm.create_model("convnext_nano", pretrained=False, num_classes=2, in_chans=1)
model.load_state_dict(torch.load(ROOT / "runs" / run_name / "best.pt", map_location="cpu"))
model.to(device).eval()

def measure_all(raw_images):
    """Strength of every filter (same measure as extract.py) and Zoobot's lens score."""
    strengths, scores = [], []
    with torch.no_grad():
        for i in range(0, len(raw_images), 64):
            x = np.stack([preprocess(r) for r in raw_images[i:i + 64]])
            fmap = model.forward_features(to_model_input(torch.from_numpy(x).unsqueeze(1)))
            flat = fmap.flatten(2)
            strengths.append((flat.norm(dim=2) if measure == "norm" else flat.clamp(min=0).mean(dim=2)).cpu())
            scores.append(torch.softmax(model.forward_head(fmap), dim=1)[:, 1].cpu())
    return torch.cat(strengths).numpy(), torch.cat(scores).numpy()

# ---- 4. Run every probe ------------------------------------------------------------
on_rate, zoobot = {}, {}
painted_example = {}
for probe in PROBES:
    painted = raws if probe == "original" else [paint(r, probe) for r in raws]
    painted_example[probe] = painted[example]
    strength, score = measure_all(painted)
    on_rate[probe] = (strength > threshold).mean(0)          # per filter, over the galaxies
    zoobot[probe] = score
    print(f"{probe:>10s}: Zoobot's average lens score {score.mean():.2f}, "
          f"calls {np.mean(score > 0.5):4.0%} of them lenses")

results = pd.DataFrame({p: on_rate[p] for p in PROBES})
results.insert(0, "filter", [f"k{k}" for k in range(len(threshold))])
results["in_rules"] = [", ".join(sorted(in_rules.get(f"k{k}", []))) for k in range(len(threshold))]
results.to_csv(out_dir / "probe_results.csv", index=False)

# ---- 5. Report ---------------------------------------------------------------------
print(f"\n% of {len(raws)} non-lens galaxies where each rule filter is ON, before and after painting:")
print(f"{'filter':>7s} " + "".join(f"{p:>12s}" for p in PROBES))
for k in rule_filters:
    row = results.loc[results["filter"] == k, PROBES].iloc[0]
    print(f"{k:>7s} " + "".join(f"{v:12.0%}" for v in row))

# "two rings" score: ON for compound, minus the most it is ON for any single ring
extra = results.assign(two_rings=results["compound"]
                       - results[["ring", "outer only", "bright ring"]].max(axis=1))
print("\nFilters (of all 640) most specific to TWO rings (compound vs. either ring alone):")
for _, r in extra.sort_values("two_rings", ascending=False).head(8).iterrows():
    print(f"  {r['filter']:>5s}  ring {r['ring']:4.0%} | outer only {r['outer only']:4.0%} | "
          f"bright ring {r['bright ring']:4.0%} | "
          f"compound {r['compound']:4.0%}" + (f"   (in rules: {r['in_rules']})" if r["in_rules"] else ""))

fig, axes = plt.subplots(1, len(PROBES), figsize=(1.9 * len(PROBES), 2.3))
for ax, p in zip(axes, PROBES):
    ax.imshow(preprocess(painted_example[p]), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"{p}\nZoobot {zoobot[p][example]:.2f}", fontsize=8)
    ax.axis("off")
fig.suptitle("What each probe adds to a real non-lens galaxy", fontsize=10)
fig.tight_layout()
fig.savefig(out_dir / "examples.png", dpi=130)
print(f"\nSaved {out_dir.relative_to(ROOT)}/probe_results.csv and examples.png")
