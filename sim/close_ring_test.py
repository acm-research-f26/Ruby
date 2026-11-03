# close_ring_test.py: how close can an outer source be before it stops counting as anything more than extra light on the inner ring?
# Run after Step 4, on frozen models, so this explains the result and can't change the verdict in sim/out/final/results.txt.
#
# Usage (from ~/ruby, with lens_env311):
#   python -I sim/close_ring_test.py
#
# Lenses: the 84 ordinary real lenses with a Euclid radius, from the validation and test sets, as in sim/real_lens_map.py.
# Distance 1.1, 1.2, 1.3, 1.4, 1.6, 1.8 by brightness 2, 3, 4, 6, with 3 sources per lens. The same source is used in every cell.
# Brightness 1 and below is left out, because the model barely notices those at any distance.
# At each distance, three versions add the same total light:
#   outer   the source lensed to that distance
#   same    the same source lensed to the inner ring's radius: extra light, no second ring
#   thick   the same source spread evenly from the inner ring's radius out to that distance: one thick band, with no gap
# The thick band isn't a real kind of lens. It is what a close double would look like if the model couldn't see the gap.
#
# Models: the 3-answer Zoobot (rise in P(compound)) and frozen Zoobot (rise in its double-vs-single score, the "before").
#
# How to read it, set before scoring. Rise = score with the light minus score without, averaged per lens, then over lenses.
# The interval is a 95% bootstrap over lenses.
#   Close limit, the main reading. A distance passes if, on all 3 seeds, outer rises at least 1.5x as much as same, and the
#   interval of outer minus same is above 0. The limit is the smallest distance that passes with every larger distance passing too.
#   Gap reading, the stricter one. A distance passes if, on all 3 seeds, the interval of outer minus thick is above 0.
#   The limit is found the same way. Only this one is evidence that the model tells a second ring from one thick ring.
#   Training expands the limit if the 3-answer Zoobot's limit is smaller than frozen's.
# 1.5x was chosen knowing that sim/second_ring_test.py got 1.88-2.44x at 1.3-1.8x.
#
# A close limit below the distance where the gap equals the 0.2" blur (about 1.23x at the median radius) can't come from
# seeing a gap, so that case is read as "responds to light farther out", not as separating the rings.
# The outer sources are simulated, so this doesn't test real compound lenses.
#
# Trust check: for the 3-answer Zoobot, the share of sources that raise a lens's own score must match sim/real_lens_map.py
# (new sources, same lenses and cells), with a mean difference of at most 0.05.
#
# Output (in sim/out/final/close_ring/): results.txt, cells.csv, rises.csv, close_ring.png, examples.png

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate as sim
from simulate import ROOT

DISTANCES = [1.1, 1.2, 1.3, 1.4, 1.6, 1.8]
BRIGHTNESS = [2, 3, 4, 6]
VERSIONS = 3
SEEDS = [1, 2, 3]
ARMS = ["outer", "same", "thick"]
GRID = np.round(np.arange(1.0, 1.81, 0.05), 2)
RATIO = 1.5
BLUR = 0.2
BOOTSTRAPS = 5000
device = "mps" if torch.backends.mps.is_available() else "cpu"
out_dir = ROOT / "sim/out/final/close_ring"
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

class Zoobot(nn.Module):
    def __init__(self, num_classes, seed):
        super().__init__()
        self.net = timm.create_model("convnext_nano", pretrained=False, num_classes=num_classes, in_chans=1)
        state = torch.load(ROOT / f"runs/zoobot_lr0.0001_seed{seed}/best.pt", map_location="cpu")
        if num_classes != 2:
            state = {k: v for k, v in state.items() if not k.startswith("head.fc")}
        self.net.load_state_dict(state, strict=False)
    def forward(self, img):
        return self.net(F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False))

@torch.no_grad()
def run(model, x):
    model.eval()
    return torch.cat([model(x[i:i + 64].to(device)).cpu() for i in range(0, len(x), 64)]).numpy()

# ---- Images, made once --------------------------------------------------------------------
split, lens_models = sim.load_inputs()
table = pd.read_csv(ROOT / "data/zenodo_15003116/modeling_lens_mass.csv")
e = np.hypot(table.ell_comps_0_median_pdf.astype(float), table.ell_comps_1_median_pdf.astype(float))
table = pd.DataFrame(dict(id=table.id_str, theta_e=table.einstein_radius_effective_median_pdf.astype(float), q=(1 - e) / (1 + e)))
lenses = split[split.split.isin(["val", "test"]) & (split.label == 1) & ~split.notes.eq("dspl")].merge(table, on="id")
lenses = lenses[lenses.theta_e.between(0.2, 3)].reset_index(drop=True)
say(f"{len(lenses)} ordinary real lenses with a Euclid radius, {VERSIONS} sources each, brightness {BRIGHTNESS}")
say(f"Median radius {lenses.theta_e.median():.2f}\". Gap at that radius, by distance: " + ", ".join(f"{d}x {lenses.theta_e.median() * (d - 1):.2f}\"" for d in DISTANCES) + f"; blur about {BLUR}\"")

rows, plain, images, pictures = [], [], {arm: [] for arm in ARMS}, []
for lens, row in lenses.iterrows():
    raw = np.nan_to_num(np.load(ROOT / row.path).astype(np.float32))
    _, noise = sim.noise_level(raw)
    angle = sim.host_angle(raw)
    plain.append(preprocess(raw))
    for version in range(VERSIONS):
        units = {}
        for d in GRID:
            system = sim.draw_system(np.random.default_rng([2033, lens, version]), lens_models, angle=angle, theta_e=row.theta_e, q=row.q, d=float(d), b=1.0, companion=False)
            units[float(d)] = sim.lensed_light(system)[1]
        for d in DISTANCES:
            spread = [units[float(g)] / units[float(g)].sum() for g in GRID if g <= d + 1e-9]
            thick_unit = np.mean(spread, axis=0)
            for b in BRIGHTNESS:
                outer = (b * noise / sim.peak_level(units[d])) * units[d]
                total = outer.sum()
                same = units[1.0] * total / units[1.0].sum()
                thick = thick_unit * total
                images["outer"].append(preprocess(raw + outer))
                images["same"].append(preprocess(raw + same))
                images["thick"].append(preprocess(raw + thick))
                rows.append(dict(lens=lens, version=version, d=d, b=b))
                if lens == 0 and version == 0 and b == 4:
                    pictures.append((d, preprocess(raw), preprocess(raw + outer), preprocess(raw + thick), preprocess(raw + same)))
frame = pd.DataFrame(rows)
x_plain = torch.from_numpy(np.stack(plain))[:, None]
x_arms = {arm: torch.from_numpy(np.stack(v))[:, None] for arm, v in images.items()}
del images, plain

# ---- Scores ---------------------------------------------------------------------------------
tall, aucs = [], []
for seed in SEEDS:
    folder = ROOT / f"sim/out/models2/seed{seed}"
    trained = Zoobot(3, seed)
    trained.load_state_dict(torch.load(folder / "zoobot3.pt", map_location="cpu"))
    trained.to(device)
    encoder = Zoobot(0, seed).to(device)
    with open(folder / "frozen.pkl", "rb") as f:
        frozen = pickle.load(f)
    score = {"zoobot3": lambda x: torch.softmax(torch.from_numpy(run(trained, x)), 1).numpy()[:, 2],
             "frozen": lambda x: frozen.decision_function(run(encoder, x))}
    for name, f in score.items():
        base = f(x_plain)[frame.lens.values]
        values = {arm: f(x_arms[arm]) for arm in ARMS}
        for arm in ARMS:
            tall.append(frame.assign(model=name, seed=seed, arm=arm, rise=values[arm] - base, beats_own=values[arm] > base))
        for d in DISTANCES:
            keep = frame.d.values == d
            aucs.append(dict(model=name, seed=seed, d=d, auc=roc_auc_score(np.r_[np.ones(keep.sum()), np.zeros(keep.sum())], np.r_[values["outer"][keep], values["thick"][keep]])))
tall = pd.concat(tall, ignore_index=True)
tall.to_csv(out_dir / "rises.csv", index=False)
aucs = pd.DataFrame(aucs)

# ---- Reading --------------------------------------------------------------------------------
rng = np.random.default_rng(2034)
cells = []
for (name, seed, d), group in tall.groupby(["model", "seed", "d"]):
    per_lens = group.groupby(["lens", "arm"]).rise.mean().unstack("arm")
    picks = rng.integers(len(per_lens), size=(BOOTSTRAPS, len(per_lens)))
    low = lambda other: np.percentile((per_lens.outer.values - per_lens[other].values)[picks].mean(1), 2.5)
    outer, same, thick = per_lens.outer.mean(), per_lens.same.mean(), per_lens.thick.mean()
    cells.append(dict(model=name, seed=seed, d=d, outer=outer, same=same, thick=thick, ratio=outer / same if same > 0 else np.inf,
                      low_same=low("same"), low_thick=low("thick"), pass_same=outer >= RATIO * same and low("same") > 0 and outer > 0, pass_thick=low("thick") > 0))
cells = pd.DataFrame(cells)
cells.to_csv(out_dir / "cells.csv", index=False)

def limit(passes):
    """The smallest distance that passes on all seeds, with every larger distance passing too."""
    found = None
    for d in reversed(DISTANCES):
        if not passes[d]:
            break
        found = d
    return found

limits = {}
for name in ["frozen", "zoobot3"]:
    part = cells[cells.model == name]
    say(f"\n{name}: mean rise over 3 seeds")
    mean = part.groupby("d")[["outer", "same", "thick"]].mean()
    mean["outer / same"], mean["outer / thick"] = mean.outer / mean.same, mean.outer / mean.thick
    mean["double vs. thick AUC"] = aucs[aucs.model == name].groupby("d").auc.mean()
    say(mean.round(3).to_string())
    say("  outer / same per seed:")
    say(part.pivot(index="d", columns="seed", values="ratio").round(2).to_string())
    say("  low end of the interval, outer minus same, per seed:")
    say(part.pivot(index="d", columns="seed", values="low_same").round(3).to_string())
    say("  low end of the interval, outer minus thick, per seed:")
    say(part.pivot(index="d", columns="seed", values="low_thick").round(3).to_string())
    same_pass = part.groupby("d").pass_same.all()
    thick_pass = part.groupby("d").pass_thick.all()
    limits[name] = (limit(same_pass), limit(thick_pass))
    say(f"  Close limit: {limits[name][0]}. Passing distances: {[d for d in DISTANCES if same_pass[d]]}")
    say(f"  Gap reading: {limits[name][1]}. Passing distances: {[d for d in DISTANCES if thick_pass[d]]}")

known = pd.read_csv(ROOT / "sim/out/final/real_map/summary.csv")
known = known[(known.model == "zoobot3") & known.b.isin(BRIGHTNESS) & known.d.isin(DISTANCES)].set_index(["b", "d"]).beats_own
now = tall[(tall.model == "zoobot3") & (tall.arm == "outer")].groupby(["b", "d"]).beats_own.mean()
difference = (now - known).abs().mean()
say(f"\nTrust check: share raising its own lens vs. the real-lens map, mean difference {difference:.3f} (limit 0.05): {'ok' if difference <= 0.05 else 'FAILED'}")

median_gap = lambda d: lenses.theta_e.median() * (d - 1)
for name, (close, gap) in limits.items():
    note = ""
    if close is not None and median_gap(close) < BLUR:
        note = "; below the blur, so this means it responds to light farther out, not that it separates the rings"
    say(f"{name}: close limit {close}{note}; gap reading {gap}")
rank = lambda value: np.inf if value is None else value
say(f"Training expands the close limit: {'yes' if rank(limits['zoobot3'][0]) < rank(limits['frozen'][0]) else 'no'}; expands the gap reading: {'yes' if rank(limits['zoobot3'][1]) < rank(limits['frozen'][1]) else 'no'}")

# ---- Figures --------------------------------------------------------------------------------
SURFACE, INK, SECONDARY, GRIDLINE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e4df"
COLOURS = {"outer": "#2a78d6", "thick": "#e34948", "same": "#52514e"}
LABELS = {"outer": "Outer source", "thick": "One thick band", "same": "Same light at inner radius"}
plt.rcParams["font.family"] = ["Arial", "Helvetica", "DejaVu Sans"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
for ax, name, title, unit in zip(axes, ["zoobot3", "frozen"], ["3-answer Zoobot", "Frozen Zoobot"], ["Rise in P(compound)", "Rise in double-vs-single score"]):
    mean = cells[cells.model == name].groupby("d")[ARMS].mean()
    for arm in ["outer", "thick", "same"]:
        ax.plot(mean.index, mean[arm], color=COLOURS[arm], lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label=LABELS[arm])
    ax.axvline(1 + BLUR / lenses.theta_e.median(), color=SECONDARY, lw=1, ls=(0, (3, 3)))
    ax.text(1 + BLUR / lenses.theta_e.median() + 0.01, ax.get_ylim()[0], "gap = blur", color=SECONDARY, fontsize=9, va="bottom")
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=13, color=INK, fontweight="bold", loc="left")
    ax.set_xlabel("Outer ring distance (× inner ring radius)", fontsize=10.5, color=SECONDARY)
    ax.set_ylabel(unit, fontsize=10.5, color=SECONDARY)
    ax.set_xticks(DISTANCES, [f"{d:g}×" for d in DISTANCES])
    ax.tick_params(length=0, labelsize=10, labelcolor=SECONDARY)
    ax.grid(axis="y", color=GRIDLINE, lw=1)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRIDLINE)
axes[0].legend(frameon=False, fontsize=10, labelcolor=SECONDARY)
fig.text(0.01, 0.01, f"{len(lenses)} real lenses with a simulated source, brightness 2–6× the noise, mean of 3 seeds. Same total light in all three versions. After-the-fact check.", fontsize=9.5, color=SECONDARY)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(out_dir / "close_ring.png", dpi=170, facecolor=SURFACE)

fig, axes = plt.subplots(len(pictures), 4, figsize=(8, 2 * len(pictures)))
for line, (d, *panels) in zip(axes, pictures):
    for ax, panel, title in zip(line, panels, ["real lens", f"+ outer source at {d}x", "+ one thick band", "+ same light, inner radius"]):
        ax.imshow(panel, cmap="gray", origin="lower")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
fig.tight_layout()
fig.savefig(out_dir / "examples.png", dpi=130)

(out_dir / "results.txt").write_text("\n".join(report) + "\n")
print("Done.", flush=True)
