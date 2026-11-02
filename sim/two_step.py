# two_step.py: the baseline Zoobot finds lenses, then the 3-answer Zoobot checks only those for a second ring.
# Run after Step 4, so this explains the result and can't change the verdict in sim/out/final/results.txt.
#
# Usage (from ~/ruby, with lens_env311):
#   python -I sim/two_step.py
#
# Step one keeps an image if the baseline's P(lens) clears a cutoff. The cutoff is set on validation lenses, separately for
# each seed, so that it keeps 95% of them (90% and 99% are shown for reference). Step two flags a kept image as compound if
# the same seed's 3-answer Zoobot gives P(compound) > 0.5.
# Lens finding is the baseline's by construction, because step one is the baseline.
#
# Worth using if, at the 95% cutoff on every seed: all 4 compound lenses pass step one and get flagged, and fewer
# non-lenses get flagged than with the 3-answer Zoobot alone. Set before scoring.
#
# Output (in sim/out/final/two_step/): results.txt, scores.csv

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path.home() / "ruby"
SEEDS = [1, 2, 3]
KEEP = [0.95, 0.90, 0.99]
device = "mps" if torch.backends.mps.is_available() else "cpu"
out_dir = ROOT / "sim/out/final/two_step"
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

def load(rows):
    return torch.from_numpy(np.stack([preprocess(np.load(ROOT / p)) for p in rows.path]))[:, None]

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

split = pd.read_csv(ROOT / "data/split.csv", dtype={"id": str})
val_lenses = split[(split.split == "val") & (split.label == 1)].reset_index(drop=True)
test = split[split.split == "test"].reset_index(drop=True)
x_val, x_test = load(val_lenses), load(test)
is_lens = test.label.eq(1).values
is_compound = test.notes.eq("dspl").values
is_ordinary = is_lens & ~is_compound
is_non = ~is_lens
say(f"Test images: {is_compound.sum()} compound lenses, {is_ordinary.sum()} ordinary lenses, {is_non.sum()} non-lenses")

rows, table, passes = [], [], []
for seed in SEEDS:
    baseline = zoobot(2, ROOT / f"runs/zoobot_lr0.0001_seed{seed}/best.pt")
    trained = zoobot(3, ROOT / f"sim/out/models2/seed{seed}/zoobot3.pt")
    val_lens = softmax(baseline, x_val)[:, 1]
    p_lens = softmax(baseline, x_test)[:, 1]
    answers = softmax(trained, x_test)
    p_compound = answers[:, 2]
    rows.append(pd.DataFrame(dict(seed=seed, id=test.id, label=test.label, compound=is_compound, p_lens=p_lens, p_compound=p_compound)))

    flagged = p_compound > 0.5
    rank_all = pd.Series(-p_compound).rank().values
    table.append(dict(seed=seed, method="3-answer alone", keep="", lens_auc=roc_auc_score(is_lens, 1 - answers[:, 0]),
                      lenses_kept="", non_kept="", compound_kept="", compound_flagged=flagged[is_compound].sum(),
                      ordinary_flagged=flagged[is_ordinary].sum(), non_flagged=flagged[is_non].sum(), compound_mean_rank=rank_all[is_compound].mean()))
    for keep in KEEP:
        cutoff = np.percentile(val_lens, 100 * (1 - keep))
        kept = p_lens >= cutoff
        both = kept & flagged
        rank_kept = pd.Series(-np.where(kept, p_compound, -1)).rank().values
        table.append(dict(seed=seed, method="two steps", keep=f"{keep:.0%}", lens_auc=roc_auc_score(is_lens, p_lens),
                          lenses_kept=kept[is_lens].sum(), non_kept=kept[is_non].sum(), compound_kept=kept[is_compound].sum(),
                          compound_flagged=both[is_compound].sum(), ordinary_flagged=both[is_ordinary].sum(), non_flagged=both[is_non].sum(),
                          compound_mean_rank=rank_kept[is_compound].mean()))
        if keep == 0.95:
            passes.append(both[is_compound].all() and both[is_non].sum() < flagged[is_non].sum())
            say(f"  seed {seed}: baseline P(lens) cutoff {cutoff:.3f}; compound lens scores at step one {np.round(p_lens[is_compound], 3)}")

frame = pd.DataFrame(table)
say("\n" + frame.round(4).to_string(index=False))
pd.concat(rows).to_csv(out_dir / "scores.csv", index=False)

main = frame[frame.keep == "95%"]
say(f"\nAt the 95% cutoff, non-lenses flagged on {3 * is_non.sum()} seed-images: {main.non_flagged.sum()}; compound flagged {main.compound_flagged.sum()} of {3 * is_compound.sum()}; ordinary flagged {main.ordinary_flagged.sum()} of {3 * is_ordinary.sum()}")
say(f"Worth using by the rule set beforehand: {'yes' if all(passes) else 'no'} {passes}")
(out_dir / "results.txt").write_text("\n".join(report) + "\n")
