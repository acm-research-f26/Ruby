# candidate_sheet.py: the ordinary lenses the 3-answer Zoobot scores most like compound lenses, for an expert to look at.
# These are NOT confirmed compound lenses. Training lenses are left out, because the model has seen them.
#
# Usage (from ~/ruby, with lens_env311):
#   python -I sim/candidate_sheet.py
#
# Output (in sim/out/final/candidates/): candidates.csv (all 153 scored lenses, 4 of them known compound), sheet.png,
# and one image per lens in top/

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F

ROOT = Path.home() / "ruby"
SEEDS = [1, 2, 3]
TOP = 20
NAMES = {"107505": "Galileo's Lens", "726389": "Cosmic Ammonite", "469177": "Cosmic Dartboard", "378903": "Teapot Lens"}
device = "mps" if torch.backends.mps.is_available() else "cpu"
out_dir = ROOT / "sim/out/final/candidates"
(out_dir / "top").mkdir(parents=True, exist_ok=True)

def preprocess(pixels):
    x = np.nan_to_num(pixels.astype(np.float32))
    background = np.median(x)
    noise = 1.4826 * np.median(np.abs(x - background)) + 1e-12
    x = np.arcsinh((x - background) / noise)
    lo, hi = np.percentile(x, [1, 99.5])
    return np.clip((x - lo) / (hi - lo + 1e-12), 0, 1).astype(np.float32)

@torch.no_grad()
def p_compound(seed, x):
    net = timm.create_model("convnext_nano", pretrained=False, num_classes=3, in_chans=1)
    state = torch.load(ROOT / f"sim/out/models2/seed{seed}/zoobot3.pt", map_location="cpu")
    net.load_state_dict({k.removeprefix("net."): v for k, v in state.items()})
    net = net.to(device).eval()
    out = [torch.softmax(net(F.interpolate(x[i:i + 64].to(device), size=(224, 224), mode="bilinear", align_corners=False)), 1).cpu()
           for i in range(0, len(x), 64)]
    return torch.cat(out).numpy()[:, 2]

split = pd.read_csv(ROOT / "data/split.csv", dtype={"id": str})
rows = split[split.split.isin(["val", "test"]) & (split.label == 1)].reset_index(drop=True)
images = [preprocess(np.load(ROOT / p)) for p in rows.path]
x = torch.from_numpy(np.stack(images))[:, None]
for seed in SEEDS:
    rows[f"score_seed{seed}"] = p_compound(seed, x)
columns = [f"score_seed{k}" for k in SEEDS]
rows["score"] = rows[columns].mean(axis=1)
rows["lowest_seed"] = rows[columns].min(axis=1)
rows["known_compound"] = rows.notes.eq("dspl")
rows["name"] = rows.id.str[-6:].map(NAMES).where(rows.known_compound, "")
rows = rows.sort_values("score", ascending=False).reset_index(drop=True)
rows["rank"] = rows.index + 1
rows[["rank", "id", "ra", "dec", "grade", "split", "known_compound", "name", "score", "lowest_seed"] + columns].round(4).to_csv(out_dir / "candidates.csv", index=False)

ordinary = rows[~rows.known_compound].head(TOP)
print(f"{len(rows)} lenses scored, {rows.known_compound.sum()} known compound")
print("Known compound, rank of", len(rows), rows[rows.known_compound][["name", "rank", "score", "lowest_seed"]].round(2).to_string(index=False))
print(ordinary[["rank", "id", "ra", "dec", "grade", "split", "score", "lowest_seed"]].round(3).to_string(index=False))

show = lambda ax, image: ax.imshow(image[20:80, 20:80], cmap="gray", origin="lower", extent=(-3, 3, -3, 3))
def picture(row):
    return preprocess(np.load(ROOT / row.path))

known = rows[rows.known_compound]
fig, axes = plt.subplots(5, 5, figsize=(11, 12.2))
for ax in axes.ravel():
    ax.axis("off")
for ax, (_, row) in zip(axes[0], known.iterrows()):
    show(ax, picture(row))
    ax.set_title(f"KNOWN: {row['name']}\nscore {row.score:.2f}", fontsize=9, fontweight="bold")
axes[0, 4].text(0.5, 0.5, "Top row: the 4 known\ncompound lenses.\nBelow: highest-scoring\nother lenses, unconfirmed.\nEach panel is 6\" across.", ha="center", va="center", fontsize=9, transform=axes[0, 4].transAxes)
for ax, (_, row) in zip(axes[1:].ravel(), ordinary.iterrows()):
    image = picture(row)
    show(ax, image)
    ax.set_title(f"#{row['rank']}  score {row.score:.2f}, grade {row.grade}\n{row.ra:.4f}, {row.dec:.4f}", fontsize=8)
    plt.imsave(out_dir / "top" / f"rank{row['rank']:03d}.png", np.kron(image[20:80, 20:80], np.ones((5, 5))), cmap="gray", origin="lower")
fig.tight_layout()
fig.savefig(out_dir / "sheet.png", dpi=140)
