# realism_check.py: can a model tell our simulated lenses from real ones?
#
# Usage (from ~/ruby, with lens_env311):
#   python sim/realism_check.py --data sim/out/v1 --name v1
#
# The main test: real ordinary lens vs. simulated single. A simulated double isn't the main test,
# because almost every real lens here is single, so a double differs for a legitimate reason.
#
# Only training and validation lenses are used. Test lenses, including the 4 real compound lenses,
# never appear here, because this check is used to tune the simulator.
#
# Each model starts from the pretrained Zoobot encoder, which has never seen our lenses:
#   frozen      Zoobot's 640 features + a simple classifier, 5 folds, a galaxy never in two folds
#   retrained   Zoobot trained end to end (train.py's settings), train split -> validation split.
#               It keeps its best epoch on the validation split, so its AUC leans high, which makes
#               the pass mark harder to meet, not easier.
#
# The comparisons, real lens vs.:
#   simulated single      the main test
#   simulated double      for information
#   painted single        the old painted rings on any galaxy: the reference, this should score higher
#   nothing added         the same galaxies with no lens: the other reference, this is lens finding
#   real, shuffled        real lenses split in two at random: must be near 0.5
#   ring zone hidden      everything from 0.5 to 1.6 Einstein radii blanked: tests the galaxies and pixels
#   only the ring zone    everything else blanked: tests the arcs
#   lenses with a radius  the full image, using only real lenses with a Euclid model, because the two
#                         masked comparisons need a radius. They should be compared with this one.
#
# Pass mark, set before any run:
#   The simulations pass if retrained Zoobot's AUC on real vs. simulated single is below 0.90 on all
#   3 seeds. The test is trusted only if real vs. shuffled stays within 0.4-0.6 and painted single
#   scores higher than simulated single.
#   If they fail: fix the biggest cue the masks point to and run again, at most 2 rounds of fixes.
#
# Changed after the first run (v1), which failed the pass mark and scored 0.622 on real vs. shuffled:
#   Keeping the best epoch on 74 validation lenses inflates AUC, and one shuffled split is noisy. The pass
#   mark itself is unchanged. Shuffled now runs on all 3 seeds and the trust check uses their average, and
#   every run also reports "later epochs": the mean AUC from epoch 3 on, with no best epoch picked.
#
# Nothing is sent anywhere.
#
# Output (in sim/out/realism/<name>/):
#   results.csv       AUC for every comparison and model
#   summary.txt       the printed report
#   extremes.png      the images retrained Zoobot finds most and least real

import argparse
import os
import random
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"     # use the copy of Zoobot already on this Mac

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate as sim
from simulate import PIX, R, ROOT, xx, yy

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="sim/out/v1")
parser.add_argument("--name", default="v1")
parser.add_argument("--seeds", default="1,2,3")
parser.add_argument("--n-train", type=int, default=1000)      # simulated images per comparison
parser.add_argument("--n-val", type=int, default=300)
parser.add_argument("--max-epochs", type=int, default=15)
parser.add_argument("--limit", type=int, default=None)        # fewer images, to check the script runs
args = parser.parse_args()

SEEDS = [int(s) for s in args.seeds.split(",")]
ZOOBOT = "hf_hub:mwalmsley/zoobot-encoder-greyscale-convnext_nano"
IMG = 224
device = "mps" if torch.backends.mps.is_available() else "cpu"
out_dir = ROOT / "sim" / "out" / "realism" / args.name
out_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

report = []
def say(line=""):
    print(line, flush=True)
    report.append(line)

start = time.time()
def stage(name):
    say(f"\n{name} ({(time.time() - start) / 60:.0f} min so far)")

# ---- 1. Same preprocessing as train.py ---------------------------------------------------
def preprocess(pixels):
    x = np.nan_to_num(pixels.astype(np.float32))
    background = np.median(x)
    noise = 1.4826 * np.median(np.abs(x - background)) + 1e-12
    x = np.arcsinh((x - background) / noise)
    lo, hi = np.percentile(x, [1, 99.5])
    return np.clip((x - lo) / (hi - lo + 1e-12), 0, 1).astype(np.float32)

def to_model_input(batch):
    return F.interpolate(batch, size=(IMG, IMG), mode="bilinear", align_corners=False)

def augment(batch):
    if random.random() < 0.5:
        batch = batch.flip(-1)
    if random.random() < 0.5:
        batch = batch.flip(-2)
    return torch.rot90(batch, k=random.randint(0, 3), dims=(-2, -1))

def hide(images, radii, keep_ring):
    """Blank the ring zone (0.5-1.6 Einstein radii), or everything except it, with each image's own median."""
    out = images.copy()
    for k, (image, radius) in enumerate(zip(images, radii)):
        ratio = R * PIX / radius
        zone = (ratio >= 0.5) & (ratio <= 1.6)
        out[k][~zone if keep_ring else zone] = np.median(image)
    return out

def painted_single(raw):
    """A single ring or pair of arcs as painted in polar/quick_test.py, with the same ranges."""
    x = np.nan_to_num(raw.astype(np.float32))
    _, noise = sim.noise_level(x)
    radius, q, phi = rng.uniform(6, 12), rng.uniform(0.75, 1), rng.uniform(0, np.pi)
    cx, cy, width = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.68, 1)
    u = np.cos(phi) * (xx - cx) + np.sin(phi) * (yy - cy)
    v = -np.sin(phi) * (xx - cx) + np.cos(phi) * (yy - cy)
    ring = np.exp(-(np.hypot(u * np.sqrt(q), v / np.sqrt(q)) - radius) ** 2 / (2 * width ** 2))
    if rng.random() < 0.5:
        angle, c, half = np.arctan2(yy - cy, xx - cx), rng.uniform(0, np.pi), np.radians(rng.uniform(30, 70))
        near = lambda a: np.abs(np.angle(np.exp(1j * (angle - a)))) < half
        ring = ring * (near(c) | near(c + np.pi))
    return x + rng.uniform(4, 12) * noise * ring

# ---- 2. Images ---------------------------------------------------------------------------
split = pd.read_csv(ROOT / "data/split.csv", dtype={"id": str})
models = pd.read_csv(sim.HERE / "lens_models.csv")
radii_table = pd.read_csv(ROOT / "data/zenodo_15003116/modeling_lens_mass.csv", usecols=["id_str", "einstein_radius_effective_median_pdf"])
radii_table = radii_table[radii_table.einstein_radius_effective_median_pdf.between(0.2, 3)]
radius_of = dict(zip(radii_table.id_str, radii_table.einstein_radius_effective_median_pdf))
path_of = dict(zip(split.id, split.path))
limit_real = args.limit if args.limit else None

stage("Loading images")
real, simulated = {}, {}
for name, n in [("train", args.n_train), ("val", args.n_val)]:
    rows = split[(split.split == name) & (split.label == 1)]
    assert not rows.notes.eq("dspl").any()             # the 4 real compound lenses are test lenses
    rows = rows.iloc[:limit_real]
    real[name] = dict(id=rows.id.values, x=np.stack([preprocess(np.load(ROOT / p)) for p in rows.path]),
                      radius=np.array([radius_of.get(i, np.nan) for i in rows.id]))
    n = min(n, args.limit) if args.limit else n
    data = np.load(ROOT / args.data / f"{name}.npz")
    params = pd.read_csv(ROOT / args.data / f"{name}_params.csv", dtype={"host": str}).iloc[:n]
    any_host = split[(split.split == name) & (split.label == 0)]
    painted_from = rng.choice(any_host.path.values, n)
    simulated[name] = dict(id=params.host.values, params=params,
                           single=np.stack([preprocess(x) for x in data["single"][:n]]),
                           double=np.stack([preprocess(x) for x in data["double"][:n]]),
                           plain=np.stack([preprocess(np.load(ROOT / path_of[h])) for h in params.host]),
                           painted=np.stack([preprocess(painted_single(np.load(ROOT / p))) for p in painted_from]),
                           painted_id=painted_from, radius=params.theta_e.values)
    say(f"  {name}: {len(rows)} real lenses ({np.isfinite(real[name]['radius']).sum()} with a radius), {n} simulated")

def real_with_radius(name):
    return real[name]["x"][np.isfinite(real[name]["radius"])], real[name]["radius"][np.isfinite(real[name]["radius"])], \
        real[name]["id"][np.isfinite(real[name]["radius"])]

shuffle_seed = 5

def comparison(name, part):
    """(real images, real ids, other images, other ids) for one comparison and one split."""
    r, s = real[part], simulated[part]
    if name == "simulated single":
        return r["x"], r["id"], s["single"], s["id"]
    if name == "simulated double":
        return r["x"], r["id"], s["double"], s["id"]
    if name == "painted single":
        return r["x"], r["id"], s["painted"], s["painted_id"]
    if name == "nothing added":
        return r["x"], r["id"], s["plain"], s["id"]
    if name == "real, shuffled":
        order = np.random.default_rng(shuffle_seed).permutation(len(r["x"]))
        half = len(order) // 2
        return r["x"][order[:half]], r["id"][order[:half]], r["x"][order[half:]], r["id"][order[half:]]
    x, radius, ids = real_with_radius(part)
    if name == "lenses with a radius":
        return x, ids, s["single"], s["id"]
    keep = name == "only the ring zone"
    return hide(x, radius, keep), ids, hide(s["single"], s["radius"], keep), s["id"]

COMPARISONS = ["simulated single", "simulated double", "painted single", "nothing added", "real, shuffled",
               "lenses with a radius", "ring zone hidden", "only the ring zone"]

# ---- 3. Frozen Zoobot + a simple classifier --------------------------------------------
def to_tensor(images):
    return torch.from_numpy(images)[:, None]

@torch.no_grad()
def features(encoder, images):
    x = to_tensor(images)
    return torch.cat([encoder(to_model_input(x[i:i + 64].to(device))).cpu() for i in range(0, len(x), 64)]).numpy()

stage("Frozen Zoobot + a simple classifier")
encoder = timm.create_model(ZOOBOT, pretrained=True, num_classes=0, in_chans=1).to(device).eval()
results = []
for name in COMPARISONS:
    parts = [comparison(name, part) for part in ("train", "val")]
    a = features(encoder, np.concatenate([p[0] for p in parts]))
    b = features(encoder, np.concatenate([p[2] for p in parts]))
    x = np.vstack([a, b])
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    groups = pd.factorize(np.concatenate([p[1] for p in parts] + [p[3] for p in parts]))[0]
    aucs = []
    for fit, check in GroupKFold(5).split(x, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced")).fit(x[fit], y[fit])
        aucs.append(roc_auc_score(y[check], clf.decision_function(x[check])))
    results.append(dict(comparison=name, model="frozen", seed=0, auc=np.mean(aucs), spread=np.std(aucs)))
    say(f"  {name:22s} AUC {np.mean(aucs):.3f} (folds vary by {np.std(aucs):.3f})")
del encoder

# ---- 4. Retrained Zoobot ---------------------------------------------------------------
@torch.no_grad()
def scores(model, images):
    model.eval()
    x = to_tensor(images)
    return torch.cat([torch.softmax(model(to_model_input(x[i:i + 64].to(device))), 1)[:, 1].cpu() for i in range(0, len(x), 64)]).numpy()

def retrain(name, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    a, _, b, _ = comparison(name, "train")
    x = to_tensor(np.concatenate([a, b]))
    y = torch.cat([torch.ones(len(a), dtype=torch.long), torch.zeros(len(b), dtype=torch.long)])
    va, _, vb, _ = comparison(name, "val")
    x_val = np.concatenate([va, vb])
    y_val = np.r_[np.ones(len(va)), np.zeros(len(vb))]
    weight = torch.tensor([len(y) / (2.0 * len(b)), len(y) / (2.0 * len(a))], dtype=torch.float32).to(device)
    model = timm.create_model(ZOOBOT, pretrained=True, num_classes=2, in_chans=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    best, best_state, bad = -1.0, None, 0
    history = []
    for epoch in range(args.max_epochs):
        model.train()
        order = torch.randperm(len(y))
        for i in range(0, len(order), 32):
            idx = order[i:i + 32]
            loss = F.cross_entropy(model(to_model_input(augment(x[idx]).to(device))), y[idx].to(device), weight=weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        auc = roc_auc_score(y_val, scores(model, x_val))
        history.append(auc)
        print(f"    {name}, seed {seed}, epoch {epoch + 1}: {auc:.3f}", flush=True)
        if auc > best:
            best, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    model.load_state_dict(best_state)
    return model, best, np.mean(history[2:] or history)

stage("Retrained Zoobot")
main_model = None
for seed in SEEDS:
    shuffle_seed = seed + 4
    for name in COMPARISONS if seed == SEEDS[0] else [COMPARISONS[0], "real, shuffled"]:     # these two get every seed
        model, auc, later = retrain(name, seed)
        results.append(dict(comparison=name, model="retrained", seed=seed, auc=auc, later=later, spread=np.nan))
        say(f"  {name:22s} seed {seed}: AUC {auc:.3f}, later epochs {later:.3f} ({(time.time() - start) / 60:.0f} min so far)")
        if name == COMPARISONS[0] and seed == SEEDS[0]:
            main_model = model
        else:
            del model
            if device == "mps":
                torch.mps.empty_cache()

frame = pd.DataFrame(results)
frame.to_csv(out_dir / "results.csv", index=False)

# ---- 5. What gives the simulations away? -----------------------------------------------
stage("What the main model looks at")
r_val, s_val = real["val"], simulated["val"]
s_scores, r_scores = scores(main_model, s_val["single"]), scores(main_model, r_val["x"])
say(f"  Share called real at a 0.5 cutoff: real lenses {(r_scores > 0.5).mean():.0%}, simulated {(s_scores > 0.5).mean():.0%}")
say("  How realistic a simulated lens looks, against what we drew (rank correlation, + = looks more real):")
for column in ["theta_e", "q", "inner", "b", "d", "re1", "offset1", "inter"]:
    say(f"    {column:9s} {spearmanr(s_val['params'][column], s_scores)[0]:+.2f}")
with_companion = s_val["params"].companion.values.astype(bool)
say(f"    with a neighbouring source: {s_scores[with_companion].mean():.2f}, without: {s_scores[~with_companion].mean():.2f} (mean score)")

fig, ax = plt.subplots(4, 8, figsize=(13, 7))
rows = [("sim, looks real", s_val["single"], np.argsort(-s_scores)), ("sim, looks simulated", s_val["single"], np.argsort(s_scores)),
        ("real, looks real", r_val["x"], np.argsort(-r_scores)), ("real, looks simulated", r_val["x"], np.argsort(r_scores))]
all_scores = [s_scores, s_scores, r_scores, r_scores]
for i, (label, images, order) in enumerate(rows):
    for j, k in enumerate(order[:8]):
        ax[i, j].imshow(images[k], cmap="gray", origin="lower")
        ax[i, j].set_title(f"{label}\n{all_scores[i][k]:.2f}", fontsize=7)
        ax[i, j].axis("off")
fig.tight_layout()
fig.savefig(out_dir / "extremes.png", dpi=100)

# ---- 6. The verdict -----------------------------------------------------------------------
stage("Result")
table = frame.pivot_table(index="comparison", columns="model", values="auc", aggfunc="mean").reindex(COMPARISONS)
table["retrained, later epochs"] = frame[frame.model == "retrained"].groupby("comparison").later.mean()
say(table.round(3).to_string())
main = frame[(frame.comparison == COMPARISONS[0]) & (frame.model == "retrained")].auc
later = frame[(frame.comparison == COMPARISONS[0]) & (frame.model == "retrained")].later
runs = frame[(frame.comparison == "real, shuffled") & (frame.model == "retrained")]
shuffled = runs.auc.mean()
painted = frame[(frame.comparison == "painted single") & (frame.model == "retrained")].auc.iloc[0]
say(f"\nRetrained Zoobot, real vs. simulated single, per seed: {np.round(main.values, 3)}, later epochs {np.round(later.values, 3)}")
say(f"Real vs. shuffled, per seed: {np.round(runs.auc.values, 3)}, later epochs {np.round(runs.later.values, 3)}")
say(f"Passes (all below 0.90): {(main < 0.90).all()}")
say(f"Trusted: shuffled {shuffled:.3f} within 0.4-0.6: {0.4 <= shuffled <= 0.6}; painted {painted:.3f} above simulated {main.mean():.3f}: {painted > main.mean()}")
(out_dir / "summary.txt").write_text("\n".join(report) + "\n")
print(f"\nSaved to sim/out/realism/{args.name}/")
