# name_filters.py: show what each rule filter responds to, so you can name it
#
# Usage (from ~/ruby, with lens_env311):
#   python nesy/name_filters.py                        # every filter used in the saved rules
#   python nesy/name_filters.py --filters k445 k185    # just these
#
# NeSyFOLD step 5 (semantic labelling). The paper names filters automatically by
# overlapping where they fire with hand-drawn object masks. Real Euclid images have no
# masks, so this is the manual version the paper falls back on: look at the images that
# switch a filter on most strongly, see WHERE on the image it fires, and write down what
# it seems to detect.
#
# Output (in nesy/out/<run>_<measure>/filters/):
#   <filter>.png   row 1: the 10 training images where the filter fires most strongly
#                  row 2: the same images with a heat map of where it fires
#                  row 3: where it fires on average for all training lenses and all
#                         non-lenses, and on the 4 compound lenses
#   naming.csv     one row per filter with an empty "label" column for your names
#                  (re-running keeps the names you already typed in)
#
# Resolution: the filter's grid is 7x7 over a 10" image, so each cell is about 1.4".
# A typical Einstein ring has a radius of about 0.8", so the heat maps can tell
# "centre" from "surroundings", but not "ring" from "lens galaxy".

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
parser.add_argument("--filters", nargs="*")          # default: every filter in rules_*.json
parser.add_argument("--n-images", type=int, default=10)
args = parser.parse_args()

IMG = 224
device = "mps" if torch.backends.mps.is_available() else "cpu"
data_dir = ROOT / args.dir
run_name, measure = data_dir.name.rsplit("_", 1)     # e.g. "zoobot_lr0.0001_seed1", "norm"
run_dir = ROOT / "runs" / run_name
out_dir = data_dir / "filters"
os.makedirs(out_dir, exist_ok=True)

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

# ---- 2. What extract.py and train_rules.py saved -----------------------------------
images = pd.read_csv(data_dir / "images.csv")
images["notes"] = images["notes"].fillna("")
saved = np.load(data_dir / "strengths.npz", allow_pickle=True)
assert (saved["ids"].astype(str) == images["id"].astype(str)).all(), "strengths.npz and images.csv disagree"
strength, threshold = saved[measure], saved["threshold"]       # (images, 640), (640,)
stats = pd.read_csv(data_dir / "filters.csv").set_index("filter")

in_rules = {}     # filter -> the rule sets it appears in, e.g. {"truth", "zoobot_top100"}
for f in sorted(data_dir.glob("rules_*.json")):
    for line in json.loads(f.read_text())["asp"]:
        for k in re.findall(r"\b(k\d+)\(", line):
            in_rules.setdefault(k, set()).add(f.stem[len("rules_"):])
names = args.filters or sorted(in_rules, key=lambda k: (-len(in_rules[k]), int(k[1:])))
if not names:
    raise SystemExit("No rules_*.json found. Run train_rules.py first, or pass --filters k445 ...")
idx = [int(k[1:]) for k in names]

# ---- 3. Where each filter fires: its 7x7 grid for every image ------------------------
model = timm.create_model("convnext_nano", pretrained=False, num_classes=2, in_chans=1)
model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
model.to(device).eval()

maps = []
with torch.no_grad():
    for i in range(0, len(images), 64):
        rows = images.iloc[i:i + 64]
        x = np.stack([preprocess(np.load(ROOT / p)) for p in rows["path"]])
        maps.append(model.forward_features(to_model_input(torch.from_numpy(x).unsqueeze(1)))[:, idx].cpu())
maps = torch.cat(maps).numpy()                       # (images, filters, 7, 7)
# the heat map shows what the strength number counts: the size of each cell for "norm",
# only the positive part for "posmean"
heat = np.abs(maps) if measure == "norm" else np.clip(maps, 0, None)
check = np.sqrt((maps ** 2).sum((2, 3))) if measure == "norm" else np.clip(maps, 0, None).mean((2, 3))
print(f"{len(names)} filters | check vs strengths.npz: max difference "
      f"{np.abs(check - strength[:, idx]).max():.1e} (should be tiny)")

train = (images["split"] == "train").to_numpy()
lens = (images["label"] == 1).to_numpy()
compound = np.where(images["notes"] == "dspl")[0]

def tag(i, f):
    r = images.iloc[i]
    kind = "compound" if r["notes"] == "dspl" else (f"lens {r['grade']}" if r["label"] == 1 else "non-lens")
    on = "ON" if strength[i, idx[f]] > threshold[idx[f]] else "off"
    return f"{kind}, {on}\nZoobot {r['zoobot_score']:.2f}"

def overlay(ax, i, f, vmax, title):
    img = preprocess(np.load(ROOT / images.iloc[i]["path"]))
    big = F.interpolate(torch.tensor(heat[i, f])[None, None], size=img.shape,
                        mode="bilinear", align_corners=False)[0, 0].numpy()
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.imshow(big, cmap="inferno", alpha=0.55, vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=7)

# ---- 4. One picture per filter -------------------------------------------------------
naming_path = data_dir / "naming.csv"
old_labels = (pd.read_csv(naming_path).fillna("").set_index("filter")["label"].to_dict()
              if naming_path.exists() else {})
table = []
n = max(args.n_images, 7)
for f, k in enumerate(names):
    order = np.argsort(-strength[:, idx[f]])
    top = [i for i in order if train[i]][:args.n_images]
    vmax = heat[top, f].max()                        # one colour scale for rows 2-3
    avg_lens, avg_non = heat[train & lens, f].mean(0), heat[train & ~lens, f].mean(0)
    centre_share = avg_lens[2:5, 2:5].sum() / avg_lens.sum()   # central 3x3 cells; 18% if spread evenly

    fig, axes = plt.subplots(3, n, figsize=(1.35 * n, 5.6))
    for ax in axes.ravel():
        ax.axis("off")
    for c, i in enumerate(top):
        axes[0, c].imshow(preprocess(np.load(ROOT / images.iloc[i]["path"])), cmap="gray", vmin=0, vmax=1)
        axes[0, c].set_title(tag(i, f), fontsize=7)
        overlay(axes[1, c], i, f, vmax, f"#{c + 1} strongest")
    avg_max = max(avg_lens.max(), avg_non.max())
    axes[2, 0].imshow(avg_lens, cmap="inferno", vmin=0, vmax=avg_max)
    axes[2, 0].set_title(f"average, all\nlenses ({centre_share:.0%} centre)", fontsize=7)
    axes[2, 1].imshow(avg_non, cmap="inferno", vmin=0, vmax=avg_max)
    axes[2, 1].set_title("average, all\nnon-lenses", fontsize=7)
    for c, i in enumerate(compound[:n - 3]):
        overlay(axes[2, 3 + c], i, f, vmax, tag(i, f))
    s = stats.loc[k]
    used = ", ".join(sorted(in_rules.get(k, []))) or "none"
    fig.suptitle(f"Filter {k}: ON for {s['on_rate_lenses']:.0%} of training lenses, "
                 f"{s['on_rate_nonlenses']:.0%} of non-lenses  |  in rules: {used}", fontsize=10)
    fig.tight_layout(h_pad=1.5)
    fig.savefig(out_dir / f"{k}.png", dpi=120)
    plt.close(fig)

    table.append({"filter": k, "on_rate_lenses": round(s["on_rate_lenses"], 3),
                  "on_rate_nonlenses": round(s["on_rate_nonlenses"], 3), "in_rules": used,
                  "lens_centre_share": round(float(centre_share), 3),
                  "compound_on": int(sum(strength[i, idx[f]] > threshold[idx[f]] for i in compound)),
                  "label": old_labels.get(k, "")})
    print(f"  {k:>5s}  lenses {s['on_rate_lenses']:4.0%} / non-lenses {s['on_rate_nonlenses']:4.0%}"
          f"  | centre share {centre_share:4.0%} | ON for {table[-1]['compound_on']}/4 compound lenses")

pd.DataFrame(table).to_csv(naming_path, index=False)
print(f"\nSaved {len(names)} pictures to {out_dir.relative_to(ROOT)}/ and {naming_path.relative_to(ROOT)}")
print("Open each picture, decide what the filter detects, and type it into the 'label' column of naming.csv")
