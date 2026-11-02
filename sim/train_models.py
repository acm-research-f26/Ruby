# train_models.py: Step 3. Train the models that answer "not a lens / lens / compound" on the simulations.
#
# Usage (from ~/ruby, with lens_env311):
#   python -I sim/train_models.py --seed 1
#
# Training images (all from the training split):
#   not a lens   real non-lens galaxies
#   lens         real lenses + simulated singles (sim/out/v2)
#   compound     simulated doubles (sim/out/v2)
# Simulated singles count as "lens", so looking simulated can't mean "compound". Real lenses are repeated
# 4 times per epoch, because otherwise 9 of every 10 lens images would be simulated.
#
# Models, on the same images:
#   zoobot3   the Zoobot we fine-tuned on real lenses (the team's baseline), with a new 3-answer output,
#             retrained end to end at train.py's settings
#   polar3    the polar branch from polar/faint_test.py with a 3-answer output, trained from scratch
#   frozen    the baseline Zoobot, untouched, plus a simple double-vs-single classifier: the "before"
#             for the detection map
#
# Pass marks for Step 4, set before any of these models were trained. Each is judged on test data that
# this script never loads:
#   Real compound lenses  rank by P(compound) among the 79 real test lenses. The 4 compound lenses must
#                         average 23 or better over the 3 seeds (luck gets there about 5% of the time),
#                         and be better than guessing (40) on every seed.
#   Ordinary lens-finding AUC of real lens vs. real non-lens on the 379-image test, using
#                         1 - P(not a lens), must stay within 0.01 of the same seed's baseline Zoobot.
#   False alarms          at most 20% of the 75 ordinary real test lenses called compound (P > 0.5), every seed.
#   Detection map         double vs. single AUC on simulated test galaxies, for distance 1.1, 1.2, 1.3, 1.4,
#                         1.6, 1.8 by outer brightness 0.5, 1, 1.5, 2, 3, 4, 6. At least 5 more of the 42
#                         cells must reach AUC 0.8 than the frozen baseline, every seed.
#   Semi-real             real test lens + a simulated outer source vs. the same lens: AUC must beat the
#                         frozen baseline by at least 0.05, every seed.
# What they mean: zoobot3 passing all of them is the main result. If only polar3 passes, the polar idea
# comes back. If neither does, we report the detection map and the gap between simulated and real.
#
# This script reports on validation data only. That includes a warning check: because a model can tell
# simulated from real (sim/realism_check.py), it might learn "looks real = not compound". So real
# validation lenses also get a simulated outer source, to see whether the model notices it.
#
# Round 2, the one extra round allowed after round 1's validation results (the pass marks don't change):
#   python -I sim/train_models.py --seed 1 --min-signal 6 --patience 8 --name models2
# It trains longer before stopping, and leaves simulated doubles with a best possible signal under 6 out of
# training, because a model can't tell them from singles. Validation and test images stay the same.
#
# Nothing is sent anywhere.
#
# Output (in sim/out/models/seed<k>/, or sim/out/<name>/seed<k>/):
#   zoobot3.pt, polar3.pt, frozen.pkl   the trained models (large: don't commit them)
#   report.txt                          the printed report

import argparse
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate as sim
from simulate import ROOT

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)             # also picks which baseline Zoobot (1, 2 or 3)
parser.add_argument("--data", default="sim/out/v2")
parser.add_argument("--max-epochs", type=int, default=30)
parser.add_argument("--limit", type=int, default=None)         # fewer images, to check the script runs
parser.add_argument("--name", default="models")
parser.add_argument("--min-signal", type=float, default=0)     # leave fainter simulated doubles out of training
parser.add_argument("--patience", type=int, default=None)      # epochs without improvement before stopping
args = parser.parse_args()

REAL_REPEAT = 4
IMG = 224
N_R, N_A, R_MAX = 120, 180, 3.0                                # the polar strip, as in polar/faint_test.py
device = "mps" if torch.backends.mps.is_available() else "cpu"
out_dir = ROOT / "sim" / "out" / args.name / f"seed{args.seed}"
out_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

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

def to_tensor(images):
    return torch.from_numpy(np.stack(images))[:, None]

# ---- 2. Models ---------------------------------------------------------------------------
r = (torch.arange(N_R) + 0.5) * (R_MAX / N_R) / 0.1
t = torch.arange(N_A) * (2 * np.pi / N_A)
GRID = torch.stack([r[:, None] * torch.cos(t) * 2 / 100, r[:, None] * torch.sin(t) * 2 / 100], -1)

def unwrap(img):
    return F.grid_sample(img, GRID.to(img.device).expand(len(img), -1, -1, -1), mode="bilinear", align_corners=False)

class PolarBranch(nn.Module):
    """Keeps every radius row, adds up light around the ring, then looks along radius (2" at once)."""
    def __init__(self, ch=32):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, kernel_size=(9, 7), padding=(4, 0)), nn.BatchNorm2d(cout),
                                 nn.GELU(), nn.AvgPool2d(kernel_size=(1, 2)))
        def radial(dilation):
            return nn.Sequential(nn.Conv1d(ch, ch, 9, padding=4 * dilation, dilation=dilation), nn.BatchNorm1d(ch), nn.GELU())
        self.blocks = nn.ModuleList([block(1, ch), block(ch, ch), block(ch, ch)])
        self.radial = nn.Sequential(radial(1), radial(2), radial(4))
        self.head = nn.Linear(2 * ch, 3)
    def forward(self, img):
        x = unwrap(img)
        for b in self.blocks:
            x = b(F.pad(x, (3, 3, 0, 0), mode="circular"))    # 360 deg wraps to 0 deg
        x = self.radial(x.mean(dim=3))                        # average all the way around
        return self.head(torch.cat([x.amax(dim=2), x.mean(dim=2)], dim=1))

class Zoobot(nn.Module):
    """The baseline Zoobot, resized to 224 and with its old 2-answer layer replaced."""
    def __init__(self, num_classes):
        super().__init__()
        self.net = timm.create_model("convnext_nano", pretrained=False, num_classes=num_classes, in_chans=1)
        state = torch.load(ROOT / f"runs/zoobot_lr0.0001_seed{args.seed}/best.pt", map_location="cpu")
        if num_classes != 2:
            state = {k: v for k, v in state.items() if not k.startswith("head.fc")}
        self.net.load_state_dict(state, strict=False)
    def forward(self, img):
        return self.net(to_model_input(img))

@torch.no_grad()
def probabilities(model, x):
    model.eval()
    return torch.cat([torch.softmax(model(x[i:i + 64].to(device)), 1).cpu() for i in range(0, len(x), 64)]).numpy()

@torch.no_grad()
def features(model, x):
    model.eval()
    return torch.cat([model(x[i:i + 64].to(device)).cpu() for i in range(0, len(x), 64)]).numpy()

# ---- 3. Images. Training and validation only: the test split is never loaded -------------
split = pd.read_csv(ROOT / "data/split.csv", dtype={"id": str})
split = split[split.split.isin(["train", "val"])]
assert not split.notes.eq("dspl").any()                # the 4 real compound lenses are test lenses

def real_images(part, label):
    rows = split[(split.split == part) & (split.label == label)].iloc[:args.limit]
    return rows, to_tensor([preprocess(np.load(ROOT / p)) for p in rows.path])

def simulated(part):
    data = np.load(ROOT / args.data / f"{part}.npz")
    params = pd.read_csv(ROOT / args.data / f"{part}_params.csv", dtype={"host": str}).iloc[:args.limit]
    n = len(params)
    single, double = data["single"][:n], data["double"][:n]
    noise = np.array([sim.noise_level(np.nan_to_num(x))[1] for x in single])
    signal = np.sqrt(((double - single) ** 2).sum(axis=(1, 2))) / noise    # how far the outer source stands above the noise
    return params, to_tensor([preprocess(x) for x in single]), to_tensor([preprocess(x) for x in double]), signal

stage("Loading images")
_, train_non = real_images("train", 0)
_, train_real = real_images("train", 1)
_, train_single, train_double, train_signal = simulated("train")
train_double = train_double[torch.from_numpy(train_signal >= args.min_signal)]
rows_val_lens, val_real = real_images("val", 1)
_, val_non = real_images("val", 0)
val_params, val_single, val_double, _ = simulated("val")
x_train = torch.cat([train_non] + [train_real] * REAL_REPEAT + [train_single, train_double])
y_train = torch.cat([torch.zeros(len(train_non)), torch.ones(len(train_real) * REAL_REPEAT + len(train_single)),
                     2 * torch.ones(len(train_double))]).long()
say(f"  training: {len(train_non)} non-lenses, {len(train_real)} real lenses (x{REAL_REPEAT}), {len(train_single)} simulated singles, {len(train_double)} simulated doubles")
say(f"  validation: {len(val_non)} non-lenses, {len(val_real)} real lenses, {len(val_single)} simulated pairs")

def checks(model):
    """Real lens vs. non-lens, and double vs. single: the two jobs, on validation images."""
    lens = 1 - np.r_[probabilities(model, val_real)[:, 0], probabilities(model, val_non)[:, 0]]
    finding = roc_auc_score(np.r_[np.ones(len(val_real)), np.zeros(len(val_non))], lens)
    compound = np.r_[probabilities(model, val_double)[:, 2], probabilities(model, val_single)[:, 2]]
    second = roc_auc_score(np.r_[np.ones(len(val_double)), np.zeros(len(val_single))], compound)
    return finding, second

def train(model, params, batch, patience, name):
    """Same loop as train.py: keep the epoch with the best validation score, here the mean of the two jobs."""
    model.to(device)
    optimizer = torch.optim.AdamW(params, weight_decay=0.05)
    counts = torch.bincount(y_train, minlength=3).float()
    weight = (counts.sum() / (3 * counts)).to(device)
    best, best_state, bad = -1.0, None, 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        order = torch.randperm(len(y_train))
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            loss = F.cross_entropy(model(augment(x_train[idx]).to(device)), y_train[idx].to(device), weight=weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        finding, second = checks(model)
        print(f"    {name} epoch {epoch}: lens finding {finding:.4f}, double vs. single {second:.4f} ({(time.time() - start) / 60:.0f} min so far)", flush=True)
        if (finding + second) / 2 > best:
            best, bad = (finding + second) / 2, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model

# ---- 4. Training -------------------------------------------------------------------------
stage("Training Zoobot with 3 answers")
zoobot3 = Zoobot(3)
train(zoobot3, [dict(params=zoobot3.parameters(), lr=1e-4)], batch=32, patience=args.patience or 3, name="zoobot3")
torch.save(zoobot3.state_dict(), out_dir / "zoobot3.pt")

stage("Training the polar branch with 3 answers")
polar3 = PolarBranch()
train(polar3, [dict(params=polar3.parameters(), lr=1e-3)], batch=64, patience=args.patience or 5, name="polar3")
torch.save(polar3.state_dict(), out_dir / "polar3.pt")

stage("Fitting the frozen baseline")
baseline = Zoobot(2).to(device)
encoder = Zoobot(0).to(device)
x = np.vstack([features(encoder, train_double), features(encoder, train_single)])
y = np.r_[np.ones(len(train_double)), np.zeros(len(train_single))]
frozen = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)).fit(x, y)
with open(out_dir / "frozen.pkl", "wb") as f:
    pickle.dump(frozen, f)

# ---- 5. Report, on validation images only ------------------------------------------------
stage("Validation report")
frozen_score = lambda images: frozen.decision_function(features(encoder, images))
second = {"zoobot3": lambda images: probabilities(zoobot3, images)[:, 2],
          "polar3": lambda images: probabilities(polar3, images)[:, 2],
          "frozen": frozen_score}

base = probabilities(baseline, torch.cat([val_real, val_non]))[:, 1]
truth = np.r_[np.ones(len(val_real)), np.zeros(len(val_non))]
say(f"Lens finding, real lens vs. non-lens: baseline Zoobot {roc_auc_score(truth, base):.4f}, zoobot3 {checks(zoobot3)[0]:.4f}, polar3 {checks(polar3)[0]:.4f}")

say("\nDouble vs. single on simulated validation pairs (0.5 = guessing):")
scores_double = {k: f(val_double) for k, f in second.items()}
scores_single = {k: f(val_single) for k, f in second.items()}
pair_truth = np.r_[np.ones(len(val_double)), np.zeros(len(val_single))]
auc_on = lambda mask, k: roc_auc_score(np.r_[np.ones(mask.sum()), np.zeros(mask.sum())], np.r_[scores_double[k][mask], scores_single[k][mask]])
say("  all pairs: " + ", ".join(f"{k} {auc_on(np.ones(len(val_params), bool), k):.3f}" for k in second))
d_bins = [(1.1, 1.3), (1.3, 1.5), (1.5, 1.8)]
b_bins = [(0.5, 1), (1, 2), (2, 4), (4, 6)]
for k in second:
    rows = []
    for lo, hi in b_bins:
        row = []
        for a, c in d_bins:
            mask = (val_params.b.values >= lo) & (val_params.b.values < hi) & (val_params.d.values >= a) & (val_params.d.values < c)
            row.append(auc_on(mask, k) if mask.sum() >= 5 else np.nan)
        rows.append(row)
    table = pd.DataFrame(rows, index=[f"b {lo}-{hi}" for lo, hi in b_bins], columns=[f"d {a}-{c}" for a, c in d_bins])
    say(f"  {k}:\n{table.round(3).to_string()}")

say("\nShare called compound (P > 0.5):")
for k, model in [("zoobot3", zoobot3), ("polar3", polar3)]:
    share = lambda images: (probabilities(model, images)[:, 2] > 0.5).mean()
    say(f"  {k}: non-lenses {share(val_non):.0%}, real lenses {share(val_real):.0%}, simulated singles {share(val_single):.0%}, simulated doubles {share(val_double):.0%}")

# ---- 6. The warning check: does the model notice a second source on a REAL lens? ----------
stage("Warning check: real validation lenses + a simulated outer source")
table = pd.read_csv(ROOT / "data/zenodo_15003116/modeling_lens_mass.csv")
e = np.hypot(table.ell_comps_0_median_pdf.astype(float), table.ell_comps_1_median_pdf.astype(float))
table = pd.DataFrame(dict(id=table.id_str, theta_e=table.einstein_radius_effective_median_pdf.astype(float), q=(1 - e) / (1 + e)))
models = pd.read_csv(sim.HERE / "lens_models.csv")
known = rows_val_lens.merge(table, on="id")
known = known[known.theta_e.between(0.2, 3)]
semi, plain, semi_b = [], [], []
for _, row in known.iterrows():
    raw = np.nan_to_num(np.load(ROOT / row.path).astype(np.float32))
    _, noise = sim.noise_level(raw)
    for _ in range(5):
        b = np.exp(rng.uniform(np.log(0.5), np.log(6)))
        system = sim.draw_system(rng, models, angle=sim.host_angle(raw), theta_e=row.theta_e, q=row.q, b=b, companion=False)
        _, outer = sim.lensed_light(system)
        semi.append(preprocess(raw + (b * noise / sim.peak_level(outer)) * outer))
        plain.append(preprocess(raw))
        semi_b.append(b)
semi, plain, semi_b = to_tensor(semi), to_tensor(plain), np.array(semi_b)
say(f"  {len(known)} real lenses with a Euclid radius, 5 versions each")
for k, f in second.items():
    a, c = f(semi), f(plain)
    faint, clear = semi_b < 2, semi_b >= 2
    pair = lambda m: roc_auc_score(np.r_[np.ones(m.sum()), np.zeros(m.sum())], np.r_[a[m], c[m]])
    say(f"  {k}: with vs. without the outer source: AUC {pair(faint | clear):.3f} (faint {pair(faint):.3f}, clear {pair(clear):.3f})")
for k, model in [("zoobot3", zoobot3), ("polar3", polar3)]:
    p = lambda images: probabilities(model, images)[:, 2].mean()
    say(f"  {k} mean P(compound): simulated single {p(val_single):.2f}, simulated double {p(val_double):.2f}, real lens {p(plain):.2f}, real lens + outer source {p(semi):.2f}")

(out_dir / "report.txt").write_text("\n".join(report) + "\n")
say(f"\nDone ({(time.time() - start) / 60:.0f} min). Saved to sim/out/{args.name}/seed{args.seed}/")
