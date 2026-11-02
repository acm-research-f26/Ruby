# score_hosts.py: how much does each non-lens galaxy look like a real lens galaxy, as Zoobot sees it?
#
# Usage (from ~/ruby, with lens_env311):
#   python sim/score_hosts.py
#
# Trains pretrained Zoobot's frozen features + a simple classifier on real training lenses vs. training
# non-lenses, with the ring zone (0.5-1.6 Einstein radii) blanked in both, so arcs can't decide it.
# Non-lenses get a radius drawn from the real lens table. Only training lenses are used.
# Nothing is sent anywhere.
#
# Output: sim/host_scores.csv, each non-lens galaxy's odds of being a lens galaxy

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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate as sim
from simulate import PIX, R, ROOT

device = "mps" if torch.backends.mps.is_available() else "cpu"
split = pd.read_csv(ROOT / "data/split.csv", dtype={"id": str})
models = pd.read_csv(sim.HERE / "lens_models.csv")
rng = np.random.default_rng(0)
encoder = timm.create_model("hf_hub:mwalmsley/zoobot-encoder-greyscale-convnext_nano", pretrained=True, num_classes=0, in_chans=1).to(device).eval()


def blanked(path, radius):
    x = np.nan_to_num(np.load(ROOT / path).astype(np.float32))
    background = np.median(x)
    noise = 1.4826 * np.median(np.abs(x - background)) + 1e-12
    x = np.arcsinh((x - background) / noise)
    lo, hi = np.percentile(x, [1, 99.5])
    x = np.clip((x - lo) / (hi - lo + 1e-12), 0, 1).astype(np.float32)
    ratio = R * PIX / radius
    x[(ratio >= 0.5) & (ratio <= 1.6)] = np.median(x)
    return x


@torch.no_grad()
def features(images):
    x = torch.from_numpy(np.stack(images))[:, None]
    return torch.cat([encoder(F.interpolate(x[i:i + 64].to(device), size=(224, 224), mode="bilinear", align_corners=False)).cpu()
                      for i in range(0, len(x), 64)]).numpy()


lens_paths = split.set_index("id").loc[models.id].path
real = features([blanked(p, t) for p, t in zip(lens_paths, models.theta_e)])
train_hosts = split[(split.split == "train") & (split.label == 0)]
others = features([blanked(p, t) for p, t in zip(train_hosts.path, rng.choice(models.theta_e, len(train_hosts)))])
x = np.vstack([real, others])
y = np.r_[np.ones(len(real)), np.zeros(len(others))]
clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced")).fit(x, y)

hosts = split[split.label == 0]
total = np.zeros(len(hosts))
for _ in range(3):                                     # average over a few radii
    total += clf.decision_function(features([blanked(p, t) for p, t in zip(hosts.path, rng.choice(models.theta_e, len(hosts)))]))
out = pd.DataFrame(dict(id=hosts.id, split=hosts.split, odds=np.exp(total / 3)))
out.to_csv(sim.HERE / "host_scores.csv", index=False)
print(out.groupby("split").odds.describe(percentiles=[0.1, 0.5, 0.9]).round(2).to_string())
