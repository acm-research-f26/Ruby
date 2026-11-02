# tune_realism.py: which source settings bring the simulations closest to real lenses?
#
# Usage (from ~/ruby, with lens_env311):
#   python sim/tune_realism.py sources    (blur and source settings)
#   python sim/tune_realism.py            (dimmer rings, placement and how hosts are picked)
#   python sim/tune_realism.py brightness (brighter rings)
#   python sim/tune_realism.py shapes     (real galaxy shapes as sources)
#
# For each setting: simulate singles, then see how well pretrained Zoobot's frozen features + a simple
# classifier tell them from real lenses (5 folds, a galaxy never in two folds). It takes about a minute
# per setting, so it can try many, and realism_check.py then judges the winner with the full test.
# Lower is better, and 0.5 would mean they can't be told apart.
#
# Only training and validation lenses are used. Nothing is sent anywhere.
#
# Output: sim/out/tune_realism_<sources, hosts, brightness or shapes>.csv

import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"     # use the copy of Zoobot already on this Mac

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate as sim
from simulate import ROOT

N_TRAIN, N_VAL = 1000, 300
device = "mps" if torch.backends.mps.is_available() else "cpu"
split, models = sim.load_inputs()
split = pd.read_csv(ROOT / "data/split.csv", dtype={"id": str})
encoder = timm.create_model("hf_hub:mwalmsley/zoobot-encoder-greyscale-convnext_nano", pretrained=True, num_classes=0, in_chans=1).to(device).eval()


def preprocess(pixels):
    x = np.nan_to_num(pixels.astype(np.float32))
    background = np.median(x)
    noise = 1.4826 * np.median(np.abs(x - background)) + 1e-12
    x = np.arcsinh((x - background) / noise)
    lo, hi = np.percentile(x, [1, 99.5])
    return np.clip((x - lo) / (hi - lo + 1e-12), 0, 1).astype(np.float32)


@torch.no_grad()
def features(images):
    x = torch.from_numpy(np.stack(images))[:, None]
    return torch.cat([encoder(F.interpolate(x[i:i + 64].to(device), size=(224, 224), mode="bilinear", align_corners=False)).cpu()
                      for i in range(0, len(x), 64)]).numpy()


lenses = split[(split.label == 1) & split.split.isin(["train", "val"])]
assert not lenses.notes.eq("dspl").any()
real_raw = {i: np.load(ROOT / p) for i, p in zip(lenses.id, lenses.path)}
real = features([preprocess(x) for x in real_raw.values()])
real_ids = np.array(list(real_raw))
theta_of = dict(zip(models.id, models.theta_e))
real_arcs = np.array([row for i, x in real_raw.items() if i in theta_of for row in sim.arc_shape(np.nan_to_num(x.astype(np.float32)), theta_of[i])])
print(f"Real lenses: {len(real)}. Arc width percentiles 5/25/50/75/95: {np.round(np.percentile(real_arcs[:, 0], [5, 25, 50, 75, 95]), 2)}", flush=True)


use_bank = False


def trial(label, **settings):
    for key, value in settings.items():
        setattr(sim, key, value)
    rng = np.random.default_rng(1)
    images, ids, arcs = [], [], []
    for part, n in [("train", N_TRAIN), ("val", N_VAL)]:
        sim.BANK = sim.source_bank(split, part) if use_bank else None
        hosts = split[(split.split == part) & (split.label == 0)].reset_index(drop=True)
        weight = sim.host_weights(split, hosts)
        for i in rng.choice(len(hosts), n, p=weight):
            single, _, system = sim.make_pair(np.load(ROOT / hosts.path[i]), rng, models)
            images.append(preprocess(single))
            ids.append(hosts.id[i])
            arcs += sim.arc_shape(single, system["theta_e"])
    other = features(images)
    x = np.vstack([real, other])
    y = np.r_[np.ones(len(real)), np.zeros(len(other))]
    groups = pd.factorize(np.r_[real_ids, np.array(ids)])[0]
    aucs = []
    for fit, check in GroupKFold(5).split(x, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced")).fit(x[fit], y[fit])
        aucs.append(roc_auc_score(y[check], clf.decision_function(x[check])))
    width = np.percentile(np.array(arcs)[:, 0], [5, 25, 50, 75, 95])
    print(f"{label:34s} AUC {np.mean(aucs):.3f} (folds vary by {np.std(aucs):.3f})  arc width {np.round(width, 2)}", flush=True)
    return dict(label=label, auc=np.mean(aucs), spread=np.std(aucs), width_median=width[2], **settings)


soft = dict(CLUMPS=4, CLUMP_AMP=(0.5, 2))
none = dict(CLUMPS=0, CLUMP_AMP=(1, 1))
lumpy = dict(CLUMPS=6, CLUMP_AMP=(1, 6))
if len(sys.argv) > 1 and sys.argv[1] == "sources":
    rows = [trial("sharp blur, as in v1", PSF_CORE=0.16, PSF_WING=0.36, PSF_WING_SHARE=0.0, SOURCE_SIZE=0.08, **lumpy)]
    sim.PSF_CORE, sim.PSF_WING, sim.PSF_WING_SHARE = 0.15, 0.36, 0.50
    for size in [0.05, 0.08, 0.12, 0.18, 0.25]:
        rows.append(trial(f"measured blur, size {size}, lumpy", SOURCE_SIZE=size, **lumpy))
    for size in [0.08, 0.12, 0.18, 0.25]:
        rows.append(trial(f"measured blur, size {size}, soft clumps", SOURCE_SIZE=size, **soft))
    for size in [0.12, 0.18]:
        rows.append(trial(f"measured blur, size {size}, no clumps", SOURCE_SIZE=size, **none))
elif len(sys.argv) > 1 and sys.argv[1] == "shapes":
    use_bank = True
    rows = [trial(f"real shapes, size {size}", SOURCE_SIZE=size) for size in [0.05, 0.08, 0.12, 0.18, 0.25]]
    sim.SOURCE_SIZE = 0.08
    rows.append(trial("real shapes, size 0.08, placement 0.6", PLACE=0.6))
elif len(sys.argv) > 1 and sys.argv[1] == "brightness":
    rows = [trial(f"inner brightness x{inner}", INNER_SCALE=inner) for inner in [1.0, 1.3, 1.7]]
else:
    sim.SOURCE_SIZE = 0.05
    rows = []
    for inner in [0.8, 0.6, 0.4, 0.25]:
        rows.append(trial(f"inner brightness x{inner}", INNER_SCALE=inner))
    sim.INNER_SCALE = 0.8
    rows.append(trial("source placement 0.6", PLACE=0.6))
    sim.PLACE = 0.3
    rows.append(trial("15 votes per lens galaxy", HOST_VOTES=15))
    sim.HOST_VOTES = 5
    rows.append(trial("hosts by Zoobot odds", HOST_BY_ZOOBOT=True))
    for inner in [0.6, 0.4]:
        rows.append(trial(f"hosts by Zoobot odds, brightness x{inner}", INNER_SCALE=inner))
pd.DataFrame(rows).to_csv(sim.HERE / "out" / f"tune_realism_{sys.argv[1] if len(sys.argv) > 1 else 'hosts'}.csv", index=False)
