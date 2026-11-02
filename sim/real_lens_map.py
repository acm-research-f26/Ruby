# real_lens_map.py: the detection map again, on real lenses. Each cell compares real lenses with a simulated outer source
# against the same real lenses without one. Run after Step 4, so this explains the result and can't change the verdict.
#
# Usage (from ~/ruby, with lens_env311):
#   python -I sim/real_lens_map.py
#
# Lenses: the ordinary real lenses with a Euclid radius, from the validation and test sets. The 4 compound lenses are left out.
# Cells: distance 1.1, 1.2, 1.3, 1.4, 1.6, 1.8 by brightness 0.5, 1, 1.5, 2, 3, 4, 6, as in Step 4. Distance 2.2 and 3 and
# brightness 9 and 14 are added to cover the Cosmic Dartboard, and are beyond the training range.
# Each lens gets 3 outer sources, and the same sources are used in every cell.
#
# How to read it, set before scoring. Take the 24 cells with brightness 2 or more inside the Step 4 map, and compare the
# trained model's AUC here with its AUC on the simulated map in sim/out/final/map.csv:
#   mean difference +0.05 or more     the simulated map was too harsh
#   mean difference within +/-0.02    the simulated map was about right
#   anything else                     in between
#
# Output (in sim/out/final/real_map/): results.txt, cells.csv (one row per model, seed and cell), summary.csv

import pickle
import sys
import time
from pathlib import Path

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

DISTANCES = [1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.2, 3.0]
BRIGHTNESS = [0.5, 1, 1.5, 2, 3, 4, 6, 9, 14]
VERSIONS = 3
SEEDS = [1, 2, 3]
BOOTSTRAPS = 200
device = "mps" if torch.backends.mps.is_available() else "cpu"
out_dir = ROOT / "sim/out/final/real_map"
out_dir.mkdir(parents=True, exist_ok=True)

report = []
def say(line=""):
    print(line, flush=True)
    report.append(line)

start = time.time()
def stage(name):
    print(f"[{name}, {(time.time() - start) / 60:.0f} min so far]", flush=True)

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

def auc(yes, no):
    return roc_auc_score(np.r_[np.ones(len(yes)), np.zeros(len(no))], np.r_[yes, no])

# ---- Images, made once --------------------------------------------------------------------
stage("Making images")
split, lens_models = sim.load_inputs()
table = pd.read_csv(ROOT / "data/zenodo_15003116/modeling_lens_mass.csv")
e = np.hypot(table.ell_comps_0_median_pdf.astype(float), table.ell_comps_1_median_pdf.astype(float))
table = pd.DataFrame(dict(id=table.id_str, theta_e=table.einstein_radius_effective_median_pdf.astype(float), q=(1 - e) / (1 + e)))
lenses = split[split.split.isin(["val", "test"]) & (split.label == 1) & ~split.notes.eq("dspl")].merge(table, on="id")
lenses = lenses[lenses.theta_e.between(0.2, 3)].reset_index(drop=True)
say(f"{len(lenses)} ordinary real lenses with a Euclid radius ({lenses.split.value_counts().to_dict()}), {VERSIONS} outer sources each")

plain, cells = [], {(b, d): [] for b in BRIGHTNESS for d in DISTANCES}
for lens, row in lenses.iterrows():
    raw = np.nan_to_num(np.load(ROOT / row.path).astype(np.float32))
    _, noise = sim.noise_level(raw)
    angle = sim.host_angle(raw)
    plain.append(preprocess(raw))
    for version in range(VERSIONS):
        for d in DISTANCES:
            system = sim.draw_system(np.random.default_rng([2030, lens, version]), lens_models, angle=angle, theta_e=row.theta_e, q=row.q, d=d, b=1.0, companion=False)
            _, unit = sim.lensed_light(system)
            unit = (noise / sim.peak_level(unit)) * unit
            for b in BRIGHTNESS:
                cells[b, d].append(preprocess(raw + b * unit))
x_plain = torch.from_numpy(np.stack(plain))[:, None]
x_cells = {key: torch.from_numpy(np.stack(v))[:, None] for key, v in cells.items()}
del cells, plain

# ---- Scores ---------------------------------------------------------------------------------
rows = []
keep = {"zoobot3": {}, "frozen": {}}
keep_plain = {"zoobot3": [], "frozen": []}
for seed in SEEDS:
    stage(f"Scoring seed {seed}")
    folder = ROOT / f"sim/out/models2/seed{seed}"
    zoobot3 = Zoobot(3, seed)
    zoobot3.load_state_dict(torch.load(folder / "zoobot3.pt", map_location="cpu"))
    zoobot3.to(device)
    encoder = Zoobot(0, seed).to(device)
    with open(folder / "frozen.pkl", "rb") as f:
        frozen = pickle.load(f)
    score = {"zoobot3": lambda x: torch.softmax(torch.from_numpy(run(zoobot3, x)), 1).numpy()[:, 2],
             "frozen": lambda x: frozen.decision_function(run(encoder, x))}
    for name, f in score.items():
        base = f(x_plain)
        keep_plain[name].append(base)
        for key, images in x_cells.items():
            values = f(images)
            keep[name].setdefault(key, []).append(values)
            b, d = key
            rows.append(dict(model=name, seed=seed, b=b, d=d, auc=auc(values, base),
                             beats_own=(values.reshape(len(lenses), VERSIONS) > base[:, None]).mean(), rise=(values.reshape(len(lenses), VERSIONS) - base[:, None]).mean()))
frame = pd.DataFrame(rows)
frame.to_csv(out_dir / "cells.csv", index=False)

# ---- Summary --------------------------------------------------------------------------------
stage("Summarizing")
rng = np.random.default_rng(2031)
summary = frame.groupby(["model", "b", "d"]).agg(auc=("auc", "mean"), low_seed=("auc", "min"), high_seed=("auc", "max"), beats_own=("beats_own", "mean")).reset_index()
intervals = []
for _, r in summary.iterrows():
    # A scale that means the same across seeds: each seed's score as its rank among that seed's plain lenses
    values = np.stack(keep[r.model][r.b, r.d]).mean(0).reshape(len(lenses), VERSIONS)
    base = np.stack(keep_plain[r.model]).mean(0)
    runs = []
    for _ in range(BOOTSTRAPS):
        pick = rng.integers(len(lenses), size=len(lenses))
        runs.append(auc(values[pick].ravel(), base[pick]))
    intervals.append(np.percentile(runs, [2.5, 97.5]))
summary[["boot_low", "boot_high"]] = np.array(intervals)
summary.to_csv(out_dir / "summary.csv", index=False)

simulated = pd.read_csv(ROOT / "sim/out/final/map.csv")
simulated = simulated[simulated.model.isin(["zoobot3", "frozen"])].groupby(["model", "b", "d"]).auc.mean().rename("simulated").reset_index()
both = summary.merge(simulated, on=["model", "b", "d"])
both["difference"] = both.auc - both.simulated

for name in ["frozen", "zoobot3"]:
    say(f"\n{name}, AUC on real lenses, mean of 3 seeds (rows = brightness, columns = distance):")
    say(summary[summary.model == name].pivot(index="b", columns="d", values="auc").iloc[::-1].round(2).to_string())
say("\nTraining's gain on real lenses (zoobot3 minus frozen):")
gain = summary[summary.model == "zoobot3"].pivot(index="b", columns="d", values="auc") - summary[summary.model == "frozen"].pivot(index="b", columns="d", values="auc")
say(gain.iloc[::-1].round(2).to_string())
say("\nzoobot3, a source beats its own lens without one, share of versions:")
say(summary[summary.model == "zoobot3"].pivot(index="b", columns="d", values="beats_own").iloc[::-1].round(2).to_string())
say("\nzoobot3, real-lens map minus simulated map:")
say(both[both.model == "zoobot3"].pivot(index="b", columns="d", values="difference").iloc[::-1].round(2).to_string())
say(f"\nTypical 95% interval width for one cell, zoobot3: {(summary[summary.model == 'zoobot3'].boot_high - summary[summary.model == 'zoobot3'].boot_low).median():.3f}")

inside = frame[frame.b.isin(BRIGHTNESS[:7]) & frame.d.isin(DISTANCES[:6])]
counts = inside[inside.auc >= 0.8].groupby(["model", "seed"]).size().unstack(fill_value=0).reindex(index=["frozen", "zoobot3"], columns=SEEDS, fill_value=0)
say("\nCells of the 42 in the Step 4 map at AUC 0.8 or more, seeds 1, 2, 3 (not a pass mark here):")
say(counts.to_string())

clear = both[(both.model == "zoobot3") & (both.b >= 2)]
mean_difference = clear.difference.mean()
reading = "the simulated map was too harsh" if mean_difference >= 0.05 else "the simulated map was about right" if abs(mean_difference) <= 0.02 else "in between"
say(f"\nThe 24 cells at brightness 2+: real-lens map {clear.auc.mean():.3f}, simulated map {clear.simulated.mean():.3f}, mean difference {mean_difference:+.3f}. Reading set beforehand: {reading}")

placement = pd.read_csv(ROOT / "sim/out/final/placement/placement.csv")
say("\nThe 4 real compound lenses at their nearest cell on this map (real-lens AUC, trained):")
trained = summary[summary.model == "zoobot3"].pivot(index="b", columns="d", values="auc")
for _, r in placement.iterrows():
    row, column = trained.index[np.abs(np.log(trained.index) - np.log(r.b)).argmin()], trained.columns[np.abs(trained.columns - r.d).argmin()]
    say(f"  {r.lens:17s} b={row:g}, d={column:g}: {trained.loc[row, column]:.2f}; real mean rank {r['rank']:.1f}")

(out_dir / "results.txt").write_text("\n".join(report) + "\n")
print(f"Done ({(time.time() - start) / 60:.0f} min).", flush=True)
