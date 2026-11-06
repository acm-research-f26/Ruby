# testing_data_metrics.py: the team's shared metrics on the Testing-Data branch's 3,000 simulated images.
# The models are the frozen ones from sim/out/models2. Nothing is trained or tuned on this data.
#
# Usage (from ~/ruby, with lens_env311):
#   python -I sim/testing_data_metrics.py data/testing_data
#
# Data: 1,000 compound lenses, 1,000 single lenses, 500 lenses with a nearby galaxy, 500 non-lenses, all simulated
# (labels from test/labels.csv, is_compound).
#
# Models, 3 seeds each:
#   zoobot3    our 3-answer Zoobot, score = P(compound)
#   baseline   untouched Zoobot with a simple compound classifier on its features, score = its double-vs-single score
#   lens only  the 0.99 Zoobot, score = P(lens). It has no compound answer, so it is a reference only
#
# Metrics, as the team defined them:
#   AUROC              compound vs. all non-compound images, plus each non-compound class on its own
#   FPR at 80% recall  each model's cutoff catches 800 of the 1,000 compounds; FPR = share of non-compound images at or
#                      above that cutoff, overall and per class
#   Mean ranking       the 4 real compound lenses among the 79 real test lenses, from sim/out/final/ranks.csv
#
# Trust check: on our own simulated test pairs, double vs. single AUROC must match sim/out/final/results.txt within 0.01.
#
# Output (in sim/out/testing_data/): results.txt, metrics.csv, scores.csv

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path.home() / "ruby"
DATA = ROOT / sys.argv[1]
SEEDS = [1, 2, 3]
RECALL = 0.80
BOOTSTRAPS = 1000
KNOWN = {"zoobot3": [0.603, 0.622, 0.627], "baseline": [0.575, 0.573, 0.582]}
CLASSES = ["single", "nearby", "non_lens"]
device = "mps" if torch.backends.mps.is_available() else "cpu"
out_dir = ROOT / "sim/out/testing_data"
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

labels = pd.read_csv(DATA / "test/labels.csv")
x_test = to_tensor([np.load(DATA / "test" / f"{i}.npy", allow_pickle=False) for i in labels.id])
is_compound = labels.is_compound.values == 1
say(f"Testing-Data: {len(labels)} images, {labels.class_name.value_counts().to_dict()}")
ours = np.load(ROOT / "sim/out/v2/test.npz")
x_single, x_double = to_tensor(ours["single"]), to_tensor(ours["double"])

rows, scores, checks = [], [], []
rng = np.random.default_rng(2035)
for seed in SEEDS:
    folder = ROOT / f"sim/out/models2/seed{seed}"
    trained = Zoobot(3, seed)
    trained.load_state_dict(torch.load(folder / "zoobot3.pt", map_location="cpu"))
    trained.to(device)
    encoder = Zoobot(0, seed).to(device)
    with open(folder / "frozen.pkl", "rb") as f:
        frozen = pickle.load(f)
    lens_only = Zoobot(2, seed).to(device)
    score = {"zoobot3": lambda x: torch.softmax(torch.from_numpy(run(trained, x)), 1).numpy()[:, 2],
             "baseline": lambda x: frozen.decision_function(run(encoder, x)),
             "lens only": lambda x: torch.softmax(torch.from_numpy(run(lens_only, x)), 1).numpy()[:, 1]}
    for name, f in score.items():
        s = f(x_test)
        scores.append(pd.DataFrame(dict(model=name, seed=seed, id=labels.id, class_name=labels.class_name, score=s)))
        yes, no = s[is_compound], s[~is_compound]
        cutoff = np.sort(yes)[int(np.floor((1 - RECALL) * len(yes)))]
        row = dict(model=name, seed=seed, auroc=auc(yes, no), recall=(yes >= cutoff).mean(), fpr=(no >= cutoff).mean())
        picks = rng.integers(len(s), size=(BOOTSTRAPS, len(s)))
        boot = [roc_auc_score(is_compound[p], s[p]) for p in picks]
        row["auroc_low"], row["auroc_high"] = np.percentile(boot, [2.5, 97.5])
        for cls in CLASSES:
            other = s[labels.class_name.values == cls]
            row[f"auroc_vs_{cls}"] = auc(yes, other)
            row[f"fpr_{cls}"] = (other >= cutoff).mean()
        rows.append(row)
        if name in KNOWN:
            checks.append(dict(model=name, seed=seed, now=auc(f(x_double), f(x_single)), before=KNOWN[name][seed - 1]))
    print(f"seed {seed} done", flush=True)

frame = pd.DataFrame(rows)
frame.to_csv(out_dir / "metrics.csv", index=False)
pd.concat(scores).to_csv(out_dir / "scores.csv", index=False)
checks = pd.DataFrame(checks)
checks["difference"] = checks.now - checks.before
say(f"\nTrust check, our simulated test pairs, double vs. single AUROC (limit 0.01):")
say(checks.round(3).to_string(index=False))
say(f"  {'ok' if checks.difference.abs().max() <= 0.01 else 'FAILED'}")

for name in ["zoobot3", "baseline", "lens only"]:
    part = frame[frame.model == name]
    say(f"\n{name}")
    say(f"  AUROC, compound vs. all others: {', '.join(f'{v:.3f}' for v in part.auroc)} (mean {part.auroc.mean():.3f}); "
        f"95% intervals " + ", ".join(f"{a:.3f}-{b:.3f}" for a, b in zip(part.auroc_low, part.auroc_high)))
    for cls in CLASSES:
        say(f"    vs. {cls:9s} {', '.join(f'{v:.3f}' for v in part[f'auroc_vs_{cls}'])} (mean {part[f'auroc_vs_{cls}'].mean():.3f})")
    say(f"  FPR at 80% recall: {', '.join(f'{v:.1%}' for v in part.fpr)} (mean {part.fpr.mean():.1%}); recall reached {', '.join(f'{v:.1%}' for v in part.recall)}")
    for cls in CLASSES:
        say(f"    {cls:9s} {', '.join(f'{v:.1%}' for v in part[f'fpr_{cls}'])} (mean {part[f'fpr_{cls}'].mean():.1%})")

ranks = pd.read_csv(ROOT / "sim/out/final/ranks.csv", dtype={"id": str})
mean_rank = ranks[ranks.compound].groupby(["model", "seed"])["rank"].mean().unstack()
say("\nMean ranking of the 4 real compound lenses among 79 real test lenses (from the final test):")
say(f"  zoobot3 {', '.join(f'{v:.2f}' for v in mean_rank.loc['zoobot3'])} (mean {mean_rank.loc['zoobot3'].mean():.1f}); "
    f"baseline {', '.join(f'{v:.2f}' for v in mean_rank.loc['frozen'])} (mean {mean_rank.loc['frozen'].mean():.1f})")

(out_dir / "results.txt").write_text("\n".join(report) + "\n")
print("Done.", flush=True)
