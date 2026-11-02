# faint_test.py: does flattening help with faint outer rings, where Zoobot is weakest?
#
# Usage (from ~/ruby, with lens_env311):
#   python polar/faint_test.py --seed 1
#
# Same question for every model: does this galaxy have a faint outer ring, or only the inner ring?
#   double           inner ring + faint outer ring at 1.4-1.8x the inner radius
#   plain single     the same inner ring on the same galaxy, nothing else
#   matched single   one inner ring with the SAME total light as the double (test only).
#                    A model that just adds up light gets this backwards.
# Training pairs a double with a plain single at the same inner brightness (4-12x the noise),
# so a brighter inner ring says nothing about the answer.
#
# Why this might favour flattening: a ring at 0.5x the noise is invisible pixel by pixel, but
# adding up light all the way around lifts it out of the noise, and that is what the polar
# branch's average around the ring does. Oval rings smear in that average, so they are tested
# separately from perfect circles.
#
# The models see the same images:
#   polar + zoobot   the real Polar-Zoobot: both branches joined, trained end to end
#   zoobot finetuned trained Zoobot, retrained end to end (as train.py does)
#   polar            the polar branch alone, trained from scratch
#   zoobot frozen    trained Zoobot, untouched, plus a simple classifier on its 640 features
#   profile only     no CNN: average the strip around the ring, simple classifier on the result
#   brightness only  average image brightness as the score
#
# The polar branch here sees 2" of radius at once (quick_test.py's first version failed because
# it saw 0.175"), and its strip reaches 3" instead of 2", since a 1.8x oval ring can pass 2".
#
# Pass mark, set before any run:
#   Polar-Zoobot helps if, on oval and off-centre full rings against the matched single, at
#   1.4-1.8x and 0.5-2x the noise, its AUC beats the fine-tuned Zoobot by at least 0.05 on all
#   3 seeds. The test is trusted only if brightness only stays within 0.4-0.6.
#   If polar alone scores below profile only on perfect circles, the branch is built wrong, and
#   the result says nothing about the idea.
#
# Galaxies come from the existing split, so no test galaxy is ever seen in training.
# Nothing is sent anywhere.
#
# Output (in polar/out/faint_test/seed<k>/):
#   results.csv     AUC for each model, distance, brightness, thickness, shape
#   summary.txt     the printed report
#   examples.png    what the two classes look like, as images and as strips
#   real_lenses.csv every model's score for the real test-set lenses, compound ones marked
#   *.pt            the trained models (large: don't commit them)

import argparse
import os
import random
import time
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent   # ~/ruby

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)             # also picks which trained Zoobot (1, 2 or 3)
parser.add_argument("--views", type=int, default=2)            # painted versions of each training galaxy
parser.add_argument("--n-test", type=int, default=300)         # test galaxies per grid cell
parser.add_argument("--max-epochs", type=int, default=30)
parser.add_argument("--limit", type=int, default=None)         # fewer galaxies, to check the script runs
args = parser.parse_args()

TEST_DISTANCE = [1.4, 1.6, 1.8]        # outer ring radius / inner ring radius
TEST_BRIGHTNESS = [0.5, 1, 1.5, 2, 3]  # outer ring, in units of the image's noise (3 = clear, for reference)
TEST_WIDTH = [0.68, 1.0]               # ring thickness: Euclid's blur alone, and thicker
R, INNER = 8.4, 6                      # test inner ring: 0.84" and 6x the noise, as in detection_map.py

IMG = 224
device = "mps" if torch.backends.mps.is_available() else "cpu"
out_dir = ROOT / "polar" / "out" / "faint_test" / f"seed{args.seed}"
os.makedirs(out_dir, exist_ok=True)
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

# ---- 1. Same preprocessing as train.py and extract.py ------------------------------
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
    return F.interpolate(batch, size=(IMG, IMG), mode="bilinear", align_corners=False)

def augment(batch):
    """Random flips and 90-degree rotations: a lens is still a lens when turned."""
    if random.random() < 0.5:
        batch = batch.flip(-1)
    if random.random() < 0.5:
        batch = batch.flip(-2)
    return torch.rot90(batch, k=random.randint(0, 3), dims=(-2, -1))

# ---- 2. Rings (pixels are 0.1", cutouts centred on the galaxy) ------------------------
yy, xx = np.mgrid[:100, :100] - 49.5

def elliptical_radius(q, phi, cx, cy):
    """Distance from the centre, with circles squashed into ovals of axis ratio q."""
    u = np.cos(phi) * (xx - cx) + np.sin(phi) * (yy - cy)
    v = -np.sin(phi) * (xx - cx) + np.cos(phi) * (yy - cy)
    return np.hypot(u * np.sqrt(q), v / np.sqrt(q))

def bump(rr, radius, width):
    return np.exp(-(rr - radius) ** 2 / (2 * width ** 2))

def rings(kind, s):
    """Painted light, in units of the noise. kind: double, plain or matched. s: settings."""
    rr = elliptical_radius(s["q"], s["phi"], s["cx"], s["cy"])
    if s["shape"] == "rings":
        mask, covered = 1.0, 1.0
    else:                                 # two opposite arcs
        angle = np.arctan2(yy - s["cy"], xx - s["cx"])
        near = lambda a: np.abs(np.angle(np.exp(1j * (angle - a)))) < s["arc_half"]
        mask = (near(s["arc_centre"]) | near(s["arc_centre"] + np.pi)).astype(np.float32)
        covered = 2 * s["arc_half"] / np.pi
    r1, d, w, inner, b = s["r1"], s["d"], s["width"], s["inner"], s["b"]
    if kind == "plain":
        return inner * bump(rr, r1, w)
    if kind == "matched":                 # a ring's light goes as brightness x radius
        return (inner + b * d * covered) * bump(rr, r1, w)
    return inner * bump(rr, r1, w) + mask * b * bump(rr, d * r1, w)

def add_light(raw, painted):
    x = np.nan_to_num(raw.astype(np.float32)).copy()
    noise = 1.4826 * np.median(np.abs(x - np.median(x)))
    return x + noise * painted

def random_settings(**fixed):
    """Oval and slightly off-centre on purpose: perfect circles would flatter the polar branch."""
    s = dict(q=rng.uniform(0.75, 1.0), phi=rng.uniform(0, np.pi), cx=rng.uniform(-1, 1), cy=rng.uniform(-1, 1),
             r1=rng.uniform(6, 12), d=rng.uniform(1.4, 1.8), b=rng.uniform(0.5, 2), inner=rng.uniform(4, 12),
             width=rng.uniform(0.68, 1.0), shape=rng.choice(["rings", "arcs"]),
             arc_centre=rng.uniform(0, np.pi), arc_half=np.radians(rng.uniform(30, 70)))
    s.update(fixed)
    return s

def to_batch(images):
    return torch.from_numpy(np.stack([preprocess(i) for i in images])).unsqueeze(1)

# ---- 3. The models --------------------------------------------------------------------
N_R, N_A, R_MAX = 120, 180, 3.0                      # radius 0-3" in 0.025" steps, angle in 2 deg steps
r = (torch.arange(N_R) + 0.5) * (R_MAX / N_R) / 0.1  # radius of each row, in pixels
t = torch.arange(N_A) * (2 * np.pi / N_A)            # angle of each column
GRID = torch.stack([r[:, None] * torch.cos(t) * 2 / 100, r[:, None] * torch.sin(t) * 2 / 100], -1)

def unwrap(img):
    """(batch, 1, 100, 100) -> (batch, 1, 120 radius, 180 angle)."""
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
        self.n_out = 2 * ch

    def forward(self, img):
        x = unwrap(img)
        for b in self.blocks:
            x = b(F.pad(x, (3, 3, 0, 0), mode="circular"))    # 360 deg wraps to 0 deg
        x = self.radial(x.mean(dim=3))                        # average all the way around
        return torch.cat([x.amax(dim=2), x.mean(dim=2)], dim=1)

class PolarClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.polar = PolarBranch()
        self.out = nn.Linear(self.polar.n_out, 2)

    def forward(self, img):
        return self.out(self.polar(img))

def trained_zoobot(num_classes):
    """The Zoobot we fine-tuned on real lenses, without its old 2-answer layer."""
    model = timm.create_model("convnext_nano", pretrained=False, num_classes=num_classes, in_chans=1)
    state = torch.load(ROOT / f"runs/zoobot_lr0.0001_seed{args.seed}/best.pt", map_location="cpu")
    model.load_state_dict({k: v for k, v in state.items() if not k.startswith("head.fc")}, strict=False)
    return model

class ZoobotClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.zoobot = trained_zoobot(num_classes=2)

    def forward(self, img):
        return self.zoobot(to_model_input(img))

class PolarZoobot(nn.Module):
    def __init__(self):
        super().__init__()
        self.zoobot = trained_zoobot(num_classes=0)
        self.polar = PolarBranch()
        self.out = nn.Linear(640 + self.polar.n_out, 2)

    def forward(self, img):
        return self.out(torch.cat([self.zoobot(to_model_input(img)), self.polar(img)], dim=1))

@torch.no_grad()
def scores(model, x):
    """Probability of 'double' for every image."""
    model.eval()
    return torch.cat([torch.softmax(model(x[i:i + 64].to(device)), 1)[:, 1].cpu() for i in range(0, len(x), 64)]).numpy()

def train(model, x_train, y_train, x_val, y_val, params, batch, patience):
    """Same loop as train.py: keep the epoch with the best validation AUC."""
    model.to(device)
    optimizer = torch.optim.AdamW(params, weight_decay=0.05)
    best_auc, best_state, bad = -1.0, None, 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        order = torch.randperm(len(y_train))
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            loss = F.cross_entropy(model(augment(x_train[idx]).to(device)), y_train[idx].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_auc = roc_auc_score(y_val.numpy(), scores(model, x_val))
        print(f"    epoch {epoch}: val AUC {val_auc:.4f}", flush=True)
        if val_auc > best_auc:
            best_auc, bad = val_auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_auc

# ---- 4. Images ------------------------------------------------------------------------
split = pd.read_csv(ROOT / "data" / "split.csv")
non_lens = split[split["label"] == 0]
raws = {name: [np.load(ROOT / p) for p in non_lens[non_lens["split"] == name]["path"]] for name in ("train", "val", "test")}
raws["test"] = raws["test"][:args.n_test]
if args.limit:
    raws = {name: galaxies[:args.limit] for name, galaxies in raws.items()}

def make_set(galaxies, views):
    """Every galaxy gives a double and a plain single per view, with the same inner ring."""
    images, labels = [], []
    for raw in galaxies:
        for _ in range(views):
            s = random_settings()
            images += [add_light(raw, rings("double", s)), add_light(raw, rings("plain", s))]
            labels += [1, 0]
    return to_batch(images), torch.tensor(labels)

x_train, y_train = make_set(raws["train"], args.views)
x_val, y_val = make_set(raws["val"], 1)
say(f"Seed {args.seed}: {len(y_train)} training images, {len(y_val)} validation images on {device}")

# ---- 5. Train everything --------------------------------------------------------------
stage("Training polar branch")
polar_net = PolarClassifier()
polar, auc = train(polar_net, x_train, y_train, x_val, y_val, [{"params": polar_net.parameters(), "lr": 1e-3}], batch=64, patience=5)
say(f"  best validation AUC {auc:.3f} ({sum(p.numel() for p in polar.parameters()) / 1e3:.0f}k parameters)")
torch.save(polar.state_dict(), out_dir / "polar.pt")

stage("Fitting classifiers on frozen Zoobot features and on the plain profile")
frozen_net = trained_zoobot(num_classes=0).to(device).eval()

@torch.no_grad()
def features(x):
    return torch.cat([frozen_net(to_model_input(x[i:i + 64].to(device))).cpu() for i in range(0, len(x), 64)]).numpy()

def profile(x):
    return unwrap(x).mean(dim=3).flatten(1).numpy()

def fit_simple(extract):
    """Simple classifier on some fixed numbers, strength chosen on validation."""
    scaler = StandardScaler().fit(extract(x_train))
    f_train, f_val = scaler.transform(extract(x_train)), scaler.transform(extract(x_val))
    best_auc, best = -1.0, None
    for C in (0.001, 0.01, 0.1, 1.0):
        clf = LogisticRegression(C=C, max_iter=3000).fit(f_train, y_train.numpy())
        auc = roc_auc_score(y_val.numpy(), clf.decision_function(f_val))
        if auc > best_auc:
            best_auc, best = auc, clf
    return (lambda x: best.decision_function(scaler.transform(extract(x)))), best_auc

frozen_score, auc = fit_simple(features)
say(f"  frozen Zoobot: best validation AUC {auc:.3f}")
profile_score, auc = fit_simple(profile)
say(f"  profile only: best validation AUC {auc:.3f}")

stage("Fine-tuning Zoobot end to end (train.py's settings)")
zoobot_net = ZoobotClassifier()
finetuned, auc = train(zoobot_net, x_train, y_train, x_val, y_val, [{"params": zoobot_net.parameters(), "lr": 1e-4}], batch=32, patience=3)
say(f"  best validation AUC {auc:.3f}")
torch.save(finetuned.state_dict(), out_dir / "zoobot_finetuned.pt")

stage("Training Polar-Zoobot end to end (Zoobot at train.py's rate, new parts faster)")
joined_net = PolarZoobot()
joined, auc = train(joined_net, x_train, y_train, x_val, y_val,
                    [{"params": joined_net.zoobot.parameters(), "lr": 1e-4},
                     {"params": list(joined_net.polar.parameters()) + list(joined_net.out.parameters()), "lr": 1e-3}],
                    batch=32, patience=3)
say(f"  best validation AUC {auc:.3f}")
torch.save(joined.state_dict(), out_dir / "polar_zoobot.pt")

models = {
    "polar + zoobot": lambda x: scores(joined, x),
    "zoobot finetuned": lambda x: scores(finetuned, x),
    "polar": lambda x: scores(polar, x),
    "zoobot frozen": frozen_score,
    "profile only": profile_score,
    "brightness only": lambda x: x.mean(dim=(1, 2, 3)).numpy(),
}

# ---- 6. Test on held-out galaxies ------------------------------------------------------
def test_cell(single_kind, **fixed):
    """AUC of each model at one setting; every galaxy appears once in each class."""
    settings = [random_settings(**fixed) for _ in raws["test"]]
    doubles = to_batch([add_light(g, rings("double", s)) for g, s in zip(raws["test"], settings)])
    singles = to_batch([add_light(g, rings(single_kind, s)) for g, s in zip(raws["test"], settings)])
    labels = np.r_[np.ones(len(doubles)), np.zeros(len(singles))]
    both = torch.cat([doubles, singles])
    return {name: roc_auc_score(labels, score(both)) for name, score in models.items()}

rows = []
stage("Testing")
for width in TEST_WIDTH:
    for d in TEST_DISTANCE:
        for b in TEST_BRIGHTNESS:
            fixed = dict(r1=R, inner=INNER, d=d, b=b, width=width)
            for label, against, extra in (("rings", "matched", dict(shape="rings")),
                                          ("arcs", "matched", dict(shape="arcs")),
                                          ("round centred", "matched", dict(shape="rings", q=1.0, cx=0.0, cy=0.0)),
                                          ("rings", "plain", dict(shape="rings"))):
                for name, auc in test_cell(against, **extra, **fixed).items():
                    rows.append(dict(model=name, width=width, distance=d, brightness=b, shape=label, against=against, auc=auc))
    print(f"  width {width} done", flush=True)
results = pd.DataFrame(rows)
results.to_csv(out_dir / "results.csv", index=False)

def table(frame, by):
    return frame.pivot_table(index=by, columns="model", values="auc")[list(models)].round(3).to_string()

hard = results[results["against"] == "matched"]
faint = hard[hard["brightness"] <= 2]
say("\nDouble vs matched single, oval and off-centre rings, by outer ring brightness (0.5 = guessing, 1 = perfect):")
say(table(hard[hard["shape"] == "rings"], "brightness"))
say("\nFaint only (0.5-2x the noise), by distance:")
say(table(faint[faint["shape"] == "rings"], "distance"))
say("\nFaint only, arcs instead of full outer rings:")
say(table(faint[faint["shape"] == "arcs"], "distance"))
say("\nFaint only, perfect circles exactly centred:")
say(table(faint[faint["shape"] == "round centred"], "distance"))
say("\nFaint only, full rings, by ring thickness:")
say(table(faint[faint["shape"] == "rings"], "width"))
say("\nFaint only, double vs plain single (the same inner ring, nothing else differs):")
plain = results[(results["against"] == "plain") & (results["brightness"] <= 2)]
say(table(plain, "distance"))

# ---- 7. First look at real lenses: no painting, test-set lenses only --------------------
stage("Real lenses")
lenses = split[(split["split"] == "test") & (split["label"] == 1)].reset_index(drop=True)
x_real = to_batch([np.load(ROOT / p) for p in lenses["path"]])
real = lenses[["id", "grade", "notes"]].copy()
for name, score in models.items():
    real[name] = score(x_real)
real.to_csv(out_dir / "real_lenses.csv", index=False)
ranks = real[list(models)].rank(ascending=False)          # 1 = most double-like
is_compound = real["notes"] == "dspl"
say(f"Rank among {len(real)} real test lenses, 1 = most double-like (guessing averages {(len(real) + 1) / 2:.0f}):")
say(pd.concat([real.loc[is_compound, ["id"]], ranks[is_compound].astype(int)], axis=1).to_string(index=False))
say("Mean rank of the compound lenses: " + ", ".join(f"{m} {ranks.loc[is_compound, m].mean():.0f}" for m in models))

# ---- 8. A look at what the models are being asked to do --------------------------------
galaxy = raws["test"][1 if len(raws["test"]) > 1 else 0]
fig, axes = plt.subplots(3, 4, figsize=(11, 7.5))
for col, b in enumerate((0.5, 2)):
    s = random_settings(r1=R, inner=INNER, d=1.6, b=b, width=0.68, shape="rings", q=1.0, cx=0.0, cy=0.0)
    for k, kind in enumerate(("double", "plain")):
        x = to_batch([add_light(galaxy, rings(kind, s))])
        axes[0, 2 * col + k].imshow(x[0, 0][20:80, 20:80].numpy(), cmap="gray", origin="lower")
        axes[0, 2 * col + k].set_title(f"{kind}, outer {b}x noise" if kind == "double" else "no outer ring", fontsize=9)
        axes[1, 2 * col + k].imshow(unwrap(x)[0, 0].numpy(), cmap="gray", origin="lower", aspect="auto", extent=(0, 360, 0, R_MAX))
        axes[1, 2 * col + k].set_title("unwrapped", fontsize=9)
        axes[2, 2 * col + k].plot((np.arange(N_R) + 0.5) * R_MAX / N_R, unwrap(x)[0, 0].mean(1).numpy())
        axes[2, 2 * col + k].set_title("profile vs radius (\")", fontsize=9)
for ax in axes[:2].ravel():
    ax.set_xticks([]), ax.set_yticks([])
plt.tight_layout()
plt.savefig(out_dir / "examples.png", dpi=130)

stage("Done")
(out_dir / "summary.txt").write_text("\n".join(report) + "\n")
print(f"\nSaved to {out_dir.relative_to(ROOT)}/")
