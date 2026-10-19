# extract.py: turn the trained Zoobot into on/off "filter switches" for NeSyFOLD
#
# Usage (from ~/ruby):
#   python nesy/extract.py
#   python nesy/extract.py --measure posmean                  # try the other strength measure
#   python nesy/extract.py --run runs/zoobot_lr0.0001_seed2   # a different trained model
#
# Steps 1-2 of NeSyFOLD (Padalkar, Wang & Gupta, AAAI 2024):
#   1. Run every image through the trained Zoobot (frozen, nothing is retrained) and read
#      its last convolutional layer: 640 filters, each giving a 7x7 grid. Squash each grid
#      into one "strength" number.
#   2. A filter is ON for an image if its strength > alpha * mean + gamma * std, where the
#      mean and std come from TRAINING images only (paper: alpha 0.6, gamma 0.7).
#
# Nothing here is sent anywhere. The FOLD-SE step (which uses Gupta's server) is separate.
#
# Output (in nesy/out/<run>_<measure>/):
#   train_bin.csv, val_bin.csv, test_bin.csv   filter columns (0/1) + label (true answer)  -> FOLD-SE
#   train_bin_zoobot.csv, ...                  same, but label = Zoobot's own answer       -> FOLD-SE
#   images.csv      one row per image, in the same order as the *_bin files
#   filters.csv     per filter: threshold, and how often it is ON for lenses vs non-lenses
#   strengths.npz   raw strength numbers (both measures), for the filter-naming step later

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent   # ~/ruby

parser = argparse.ArgumentParser()
parser.add_argument("--run", default="runs/zoobot_lr0.0001_seed1")    # best validation AUC
parser.add_argument("--measure", choices=["norm", "posmean"], default="norm")
parser.add_argument("--alpha", type=float, default=0.6)
parser.add_argument("--gamma", type=float, default=0.7)
args = parser.parse_args()

IMG = 224
device = "mps" if torch.backends.mps.is_available() else "cpu"
run_dir = ROOT / args.run
out_dir = ROOT / "nesy" / "out" / f"{run_dir.name}_{args.measure}"
os.makedirs(out_dir, exist_ok=True)

# ---- 1. Same images, split and preprocessing as train.py --------------------------
# (copied, not imported: importing train.py would start a training run)
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

m = pd.read_csv(ROOT / "data" / "split.csv")
m["notes"] = m["notes"].fillna("")
order = {"train": 0, "val": 1, "test": 2}
m = m.sort_values("split", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)

# ---- 2. The trained Zoobot ---------------------------------------------------------
# Plain ConvNeXt-Nano with the weights train.py saved (lens vs non-lens, real Euclid images)
model = timm.create_model("convnext_nano", pretrained=False, num_classes=2, in_chans=1)
model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
model.to(device).eval()

# ---- 3. Read the last convolutional layer for every image --------------------------
norms, posmeans, scores = [], [], []
with torch.no_grad():
    for i in range(0, len(m), 64):
        rows = m.iloc[i:i + 64]
        images = np.stack([preprocess(np.load(ROOT / p)) for p in rows["path"]])
        x = to_model_input(torch.from_numpy(images).unsqueeze(1))
        fmap = model.forward_features(x)                    # (batch, 640, 7, 7)
        flat = fmap.flatten(2)                              # (batch, 640, 49)
        norms.append(flat.norm(dim=2).cpu())                # paper's measure: size of the grid
        posmeans.append(flat.clamp(min=0).mean(dim=2).cpu())   # only the positive part
        scores.append(torch.softmax(model.forward_head(fmap), dim=1)[:, 1].cpu())
norms = torch.cat(norms).numpy()
posmeans = torch.cat(posmeans).numpy()
m["zoobot_score"] = torch.cat(scores).numpy()
m["zoobot_pred"] = (m["zoobot_score"] > 0.5).astype(int)
n_filters = norms.shape[1]
print(f"{len(m)} images | last conv layer: {n_filters} filters, 7x7 each | device {device}")

# Sanity check: these scores should match the ones train.py saved for the test images
saved = pd.read_csv(run_dir / "test_scores.csv")[["id", "score"]]
check = m.merge(saved, on="id")
diff = (check["zoobot_score"] - check["score"]).abs().max()
print(f"check vs {args.run}/test_scores.csv: {len(check)} images, max score difference {diff:.1e}"
      + ("  (OK)" if diff < 1e-3 else "  (WARNING: preprocessing or weights differ)"))

# ---- 4. ON/OFF switches: thresholds from training images only ----------------------
strength = norms if args.measure == "norm" else posmeans
is_train = (m["split"] == "train").to_numpy()
theta = args.alpha * strength[is_train].mean(0) + args.gamma * strength[is_train].std(0)  # population std, as in the paper
on = (strength > theta).astype(int)

# drop filters that are always ON or always OFF in training: they can't appear in a rule
train_rate = on[is_train].mean(0)
keep = np.where((train_rate > 0) & (train_rate < 1))[0]
cols = [f"k{k}" for k in keep]          # names keep the original filter number
print(f"filters kept: {len(keep)} of {n_filters} "
      f"({n_filters - len(keep)} never change in training)")

# ---- 5. Save -----------------------------------------------------------------------
label_name = lambda y: np.where(y == 1, "lens", "nonlens")
for split in ["train", "val", "test"]:
    idx = (m["split"] == split).to_numpy()
    table = pd.DataFrame(on[idx][:, keep], columns=cols)
    table["label"] = label_name(m.loc[idx, "label"].to_numpy())
    table.to_csv(out_dir / f"{split}_bin.csv", index=False)
    table["label"] = label_name(m.loc[idx, "zoobot_pred"].to_numpy())
    table.to_csv(out_dir / f"{split}_bin_zoobot.csv", index=False)

m[["id", "split", "label", "grade", "notes", "zoobot_score", "zoobot_pred", "path"]].to_csv(
    out_dir / "images.csv", index=False)

lens_rows = is_train & (m["label"] == 1).to_numpy()
nonlens_rows = is_train & (m["label"] == 0).to_numpy()
filters = pd.DataFrame({
    "filter": [f"k{k}" for k in range(n_filters)],
    "threshold": theta,
    "on_rate_lenses": on[lens_rows].mean(0),        # training images only
    "on_rate_nonlenses": on[nonlens_rows].mean(0),
    "kept": np.isin(np.arange(n_filters), keep),
})
filters["difference"] = filters["on_rate_lenses"] - filters["on_rate_nonlenses"]
filters.to_csv(out_dir / "filters.csv", index=False)
np.savez_compressed(out_dir / "strengths.npz", ids=m["id"].to_numpy(),
                    norm=norms, posmean=posmeans, threshold=theta)

# ---- 6. Quick look before FOLD-SE ----------------------------------------------------
test = (m["split"] == "test").to_numpy()
acc = (m.loc[test, "zoobot_pred"] == m.loc[test, "label"]).mean()
print(f"Zoobot test accuracy at score > 0.5: {acc:.3f}  (the number the rules will be compared to)")
print(f"filters ON per training image: lenses {on[lens_rows].sum(1).mean():.1f}, "
      f"non-lenses {on[nonlens_rows].sum(1).mean():.1f}")
show = filters[filters["kept"]].sort_values("difference")
fmt = lambda r: f"  {r.filter:>5s}  ON for {r.on_rate_lenses:4.0%} of lenses, {r.on_rate_nonlenses:4.0%} of non-lenses"
print("filters most typical of LENSES:")
print("\n".join(fmt(r) for r in show.tail(5).iloc[::-1].itertuples()))
print("filters most typical of NON-LENSES:")
print("\n".join(fmt(r) for r in show.head(5).itertuples()))
print(f"\nSaved to {out_dir.relative_to(ROOT)}/")
