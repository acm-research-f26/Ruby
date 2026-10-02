# train.py: fine-tune Zoobot or a Vision Transformer on the Euclid lens dataset
#
# Usage:
#   python train.py --model zoobot --seed 1
#   python train.py --model vit    --seed 1
#
# Both models get IDENTICAL data, splits, preprocessing, augmentation and training
# rules. The only difference is the model, so any difference in results comes
# from the model.
#
# Output (in runs/<model>_lr<lr>_seed<seed>/):
#   test_scores.csv   lens score (0-1) for every test image  -> used by evaluate.py
#   log.json          settings + validation score after every epoch
#   best.pt           weights of the best epoch (not uploaded to GitHub)

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

MODELS = {
    # galaxy-pretrained CNN (Galaxy Zoo), 15M parameters, greyscale version
    "zoobot": "hf_hub:mwalmsley/zoobot-encoder-greyscale-convnext_nano",
    # everyday-photo-pretrained transformer (ImageNet-21k), 86M parameters
    "vit": "vit_base_patch16_224.augreg_in21k",
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=MODELS, required=True)
parser.add_argument("--seed", type=int, default=1)          # changes training randomness only, NOT the split
parser.add_argument("--lr", type=float, default=1e-4)       # Zoobot's default fine-tuning learning rate
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--max-epochs", type=int, default=30)
parser.add_argument("--patience", type=int, default=3)      # stop after 3 epochs without improvement (DES rule)
args = parser.parse_args()

SPLIT_SEED = 0   # fixed forever, so every model and every seed sees the exact same split
IMG = 224        # both pretrained models expect 224x224 inputs
device = "mps" if torch.backends.mps.is_available() else "cpu"
out_dir = f"runs/{args.model}_lr{args.lr:g}_seed{args.seed}"
os.makedirs(out_dir, exist_ok=True)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# ---- 1. The split: train / validation / test (made once, then reused) ----------
split_path = "data/split.csv"
if not os.path.exists(split_path):
    m = pd.read_csv("data/manifest.csv")
    m["notes"] = m["notes"].fillna("")
    rng = np.random.default_rng(SPLIT_SEED)
    m["split"] = ""
    # the compound lenses ALWAYS go to test: the models must never train on them
    m.loc[m["notes"] == "dspl", "split"] = "test"
    # everything else: 70 / 15 / 15, done separately for lenses and non-lenses
    for label in (0, 1):
        idx = m.index[(m["label"] == label) & (m["split"] == "")].to_numpy()
        rng.shuffle(idx)
        n_train, n_val = int(0.70 * len(idx)), int(0.15 * len(idx))
        m.loc[idx[:n_train], "split"] = "train"
        m.loc[idx[n_train:n_train + n_val], "split"] = "val"
        m.loc[idx[n_train + n_val:], "split"] = "test"
    m.to_csv(split_path, index=False)
m = pd.read_csv(split_path)
m["notes"] = m["notes"].fillna("")

# ---- 2. Preprocessing: the same for every image and both models ---------------
def preprocess(pixels):
    """Raw VIS pixels -> 0-1 image with faint arcs brought out (arcsinh stretch)."""
    x = np.nan_to_num(pixels.astype(np.float32))
    background = np.median(x)
    noise = 1.4826 * np.median(np.abs(x - background)) + 1e-12   # robust noise estimate
    x = np.arcsinh((x - background) / noise)                    # like the Euclid paper's arcsinh scaling
    lo, hi = np.percentile(x, [1, 99.5])
    return np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)

def load(split):
    rows = m[m["split"] == split]
    images = np.stack([preprocess(np.load(p)) for p in rows["path"]])
    return torch.from_numpy(images).unsqueeze(1), torch.tensor(rows["label"].to_numpy()), rows

x_train, y_train, _ = load("train")
x_val, y_val, _ = load("val")
x_test, y_test, test_rows = load("test")
print(f"train {len(y_train)} ({int(y_train.sum())} lenses) | val {len(y_val)} | test {len(y_test)} "
      f"(incl. {(test_rows['notes'] == 'dspl').sum()} compound lenses)")

def to_model_input(batch):
    """100x100 -> 224x224 (what the pretrained models expect)."""
    return F.interpolate(batch.to(device), size=(IMG, IMG), mode="bilinear", align_corners=False)

def augment(batch):
    """Random flips and 90-degree rotations: a lens is still a lens when turned."""
    if random.random() < 0.5:
        batch = batch.flip(-1)
    if random.random() < 0.5:
        batch = batch.flip(-2)
    return torch.rot90(batch, k=random.randint(0, 3), dims=(-2, -1))

# ---- 3. The model ----------------------------------------------------------------
model = timm.create_model(MODELS[args.model], pretrained=True, num_classes=2, in_chans=1).to(device)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"{args.model}: {n_params:.1f}M parameters on {device}")

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
# ~4x more non-lenses than lenses, so weight the classes to keep lenses from being ignored
n_pos = int(y_train.sum())
class_weights = torch.tensor([1.0, (len(y_train) - n_pos) / n_pos], device=device)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

@torch.no_grad()
def lens_scores(x):
    """Probability of 'lens' for every image."""
    model.eval()
    scores = []
    for i in range(0, len(x), 64):
        logits = model(to_model_input(x[i:i + 64]))
        scores.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    return torch.cat(scores).numpy()

# ---- 4. Train, keeping the epoch with the best validation score ------------------
log = {"args": vars(args), "params_M": round(n_params, 1), "epochs": []}
best_auc, bad_epochs = -1.0, 0
for epoch in range(1, args.max_epochs + 1):
    model.train()
    start, total_loss = time.time(), 0.0
    order = torch.randperm(len(y_train))
    for i in range(0, len(order), args.batch):
        idx = order[i:i + args.batch]
        inputs = to_model_input(augment(x_train[idx]))
        loss = loss_fn(model(inputs), y_train[idx].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(idx)

    val_auc = roc_auc_score(y_val.numpy(), lens_scores(x_val))
    minutes = (time.time() - start) / 60
    print(f"epoch {epoch}: train loss {total_loss / len(y_train):.3f} | val AUC {val_auc:.4f} | {minutes:.1f} min")
    log["epochs"].append({"epoch": epoch, "loss": total_loss / len(y_train), "val_auc": val_auc, "minutes": minutes})

    if val_auc > best_auc:
        best_auc, bad_epochs = val_auc, 0
        torch.save(model.state_dict(), f"{out_dir}/best.pt")
    else:
        bad_epochs += 1
        if bad_epochs >= args.patience:
            print(f"no improvement for {args.patience} epochs, stopping")
            break

# ---- 5. Score the test set with the best epoch's weights -------------------------
model.load_state_dict(torch.load(f"{out_dir}/best.pt", map_location=device))
out = test_rows[["id", "label", "grade", "notes"]].copy()
out["score"] = lens_scores(x_test)
out.to_csv(f"{out_dir}/test_scores.csv", index=False)
log["best_val_auc"] = best_auc
json.dump(log, open(f"{out_dir}/log.json", "w"), indent=2)
print(f"best val AUC {best_auc:.4f}. Test scores saved to {out_dir}/test_scores.csv")
