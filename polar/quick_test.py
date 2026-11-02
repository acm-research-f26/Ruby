# quick_test.py: can a polar branch see close double rings that Zoobot can't?
#
# Usage (from ~/ruby, with lens_env311):
#   python polar/quick_test.py --seed 1
#   python polar/quick_test.py --seed 1 --skip-finetune   # only the two fast models
#
# Same question for every model: is this one close double ring, or one thick single ring?
#   double           inner ring + outer ring at 1.1-1.3x the inner radius
#   matched single   one thick ring with the SAME total light, mean radius and radial
#                    spread, painted on the SAME galaxy. Only the shape of the brightness
#                    profile differs (two bumps or one), so a model can't win on brightness.
#   thin single      an ordinary one-ring lens, to catch models that call every sharp ring double
#
# Four models see the same images:
#   polar            PolarBranch trained from scratch on the unwrapped strip
#   polar wider      a redesign that sees 0.6" of radius at once. Added after seed 1 showed
#                    polar failing, so it did not exist when the pass mark was set
#   zoobot frozen    trained Zoobot, untouched, plus a simple classifier on its 640 features
#   zoobot finetuned trained Zoobot, retrained end to end on this task (as train.py does)
# The fine-tuned model is the fair opponent: the polar branch gets to train on this task, so
# Zoobot must too. "Zoobot never registers close rings" came from a Zoobot never asked to.
#
# Built-in checks:
#   1.1x             the gap is smaller than Euclid's blur, so every model should be near 0.5.
#                    A high score here would mean a shortcut, not vision.
#   brightness only  total painted-image light as the score. Should be near 0.5 at every distance.
#
# Pass mark, set before any run:
#   polar is clearly better if, on oval and off-centre full rings at 1.2-1.3x, its AUC beats
#   the fine-tuned Zoobot by at least 0.05 on all 3 seeds.
#   The test itself is trusted only if brightness only stays within 0.4-0.6 at every distance
#   and no model is far above 0.5 at 1.1x.
#
# Galaxies come from the existing split, so no test galaxy is ever seen in training.
# Nothing is sent anywhere.
#
# Output (in polar/out/quick_test/seed<k>/):
#   results.csv     AUC for each model, distance, brightness, thickness, shape
#   summary.txt     the printed report
#   examples.png    what the two classes look like, as images and as strips
#   real_lenses.csv every model's score for the real test-set lenses, compound ones marked
#   zoobot_finetuned.pt, polar_wider.pt    the trained models (large: don't commit them)
#
# The real-lens scores are a first look, not the real-lens test: these models only ever saw
# close pairs (1.1-1.3x), and real compound lenses can be more widely separated.

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
parser.add_argument("--skip-finetune", action="store_true")
args = parser.parse_args()

TEST_DISTANCE = [1.1, 1.2, 1.3]    # outer ring radius / inner ring radius
TEST_BRIGHTNESS = [2, 3, 4, 6]     # outer ring, in units of the image's noise
TEST_WIDTH = [0.68, 1.0]           # ring thickness: Euclid's blur alone, and thicker
R, INNER = 8.4, 6                  # test inner ring: 0.84" and 6x the noise, as in detection_map.py

IMG = 224
device = "mps" if torch.backends.mps.is_available() else "cpu"
out_dir = ROOT / "polar" / "out" / "quick_test" / f"seed{args.seed}"
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

def arc_mask(cx, cy, centre, half):
    """1 on two opposite arcs, 0 elsewhere."""
    angle = np.arctan2(yy - cy, xx - cx)
    near = lambda a: np.abs(np.angle(np.exp(1j * (angle - a)))) < half
    return (near(centre) | near(centre + np.pi)).astype(np.float32)

def rings(kind, s):
    """Painted light, in units of the noise. kind: double, matched or thin. s: settings."""
    rr = elliptical_radius(s["q"], s["phi"], s["cx"], s["cy"])
    mask = 1.0 if s["shape"] == "rings" else arc_mask(s["cx"], s["cy"], s["arc_centre"], s["arc_half"])
    r1, r2, w, inner, b = s["r1"], s["d"] * s["r1"], s["width"], s["inner"], s["b"]
    if kind == "thin":
        return inner * bump(rr, r1, w)
    if kind == "double":
        return inner * bump(rr, r1, w) + mask * b * bump(rr, r2, w)
    # matched: where the outer ring would be, one thick bump with the same light, mean
    # radius and spread as the two thin bumps; elsewhere just the inner ring.
    # A ring's light goes as brightness x width x radius.
    light1, light2 = inner * r1, b * r2
    p = light2 / (light1 + light2)
    radius = (light1 * r1 + light2 * r2) / (light1 + light2)
    spread = np.sqrt(w ** 2 + p * (1 - p) * (r2 - r1) ** 2)
    amplitude = (light1 + light2) * w / (spread * radius)
    return (1 - mask) * inner * bump(rr, r1, w) + mask * amplitude * bump(rr, radius, spread)

def add_light(raw, painted):
    x = np.nan_to_num(raw.astype(np.float32)).copy()
    noise = 1.4826 * np.median(np.abs(x - np.median(x)))
    return x + noise * painted

def random_settings(**fixed):
    """Oval and slightly off-centre on purpose: perfect circles would flatter the polar branch."""
    s = dict(q=rng.uniform(0.75, 1.0), phi=rng.uniform(0, np.pi), cx=rng.uniform(-1, 1), cy=rng.uniform(-1, 1),
             r1=rng.uniform(6, 12), d=rng.uniform(1.1, 1.3), b=rng.uniform(2, 6), inner=rng.uniform(4, 8),
             width=rng.uniform(0.68, 1.0), shape=rng.choice(["rings", "arcs"]),
             arc_centre=rng.uniform(0, np.pi), arc_half=np.radians(rng.uniform(30, 70)))
    s.update(fixed)
    return s

def to_batch(images):
    return torch.from_numpy(np.stack([preprocess(i) for i in images])).unsqueeze(1)

# ---- 3. The models --------------------------------------------------------------------
N_R, N_A, R_MAX = 80, 180, 2.0                       # radius 0-2" in 0.025" steps, angle in 2 deg steps
r = (torch.arange(N_R) + 0.5) * (R_MAX / N_R) / 0.1  # radius of each row, in pixels
t = torch.arange(N_A) * (2 * np.pi / N_A)            # angle of each column
GRID = torch.stack([r[:, None] * torch.cos(t) * 2 / 100, r[:, None] * torch.sin(t) * 2 / 100], -1)

def unwrap(img):
    """(batch, 1, 100, 100) -> (batch, 1, 80 radius, 180 angle)."""
    return F.grid_sample(img, GRID.to(img.device).expand(len(img), -1, -1, -1), mode="bilinear", align_corners=False)

class PolarBranch(nn.Module):
    def __init__(self, ch=32, out=64):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=(3, 7), padding=(1, 0)),  # 3 radius x 7 angle steps
                nn.GELU(),
                nn.AvgPool2d(kernel_size=(1, 2)))                           # shrink angle only
        self.blocks = nn.ModuleList([block(1, ch), block(ch, ch), block(ch, ch)])
        self.head = nn.Linear(ch * N_R, out)

    def forward(self, x):
        for b in self.blocks:
            x = F.pad(x, (3, 3, 0, 0), mode="circular")    # 360 deg wraps to 0 deg
            x = b(x)
        return self.head(x.mean(dim=3).flatten(1))         # average all the way around

class WiderPolar(nn.Module):
    """Same strip, but sees 0.6" of radius at once and finds bumps wherever they sit."""
    def __init__(self, ch=32):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, kernel_size=(9, 7), padding=(4, 0)), nn.BatchNorm2d(cout),
                                 nn.GELU(), nn.AvgPool2d(kernel_size=(1, 2)))
        self.blocks = nn.ModuleList([block(1, ch), block(ch, ch), block(ch, ch)])
        self.radial = nn.Sequential(nn.Conv1d(ch, ch, 9, padding=4), nn.BatchNorm1d(ch), nn.GELU())
        self.out = nn.Linear(ch, 2)

    def forward(self, img):
        x = unwrap(img)
        for b in self.blocks:
            x = b(F.pad(x, (3, 3, 0, 0), mode="circular"))
        x = self.radial(x.mean(dim=3))                # (batch, ch, 80 radius)
        return self.out(x.amax(dim=2))                # strongest response at any radius

class PolarClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.polar = PolarBranch()
        self.out = nn.Linear(64, 2)

    def forward(self, img):
        return self.out(F.gelu(self.polar(unwrap(img))))

class ZoobotClassifier(nn.Module):
    """The Zoobot we fine-tuned on real lenses, with a new 2-answer layer."""
    def __init__(self):
        super().__init__()
        self.zoobot = timm.create_model("convnext_nano", pretrained=False, num_classes=2, in_chans=1)
        state = torch.load(ROOT / f"runs/zoobot_lr0.0001_seed{args.seed}/best.pt", map_location="cpu")
        self.zoobot.load_state_dict({k: v for k, v in state.items() if not k.startswith("head.fc")}, strict=False)

    def forward(self, img):
        return self.zoobot(to_model_input(img))

@torch.no_grad()
def scores(model, x):
    """Probability of 'double' for every image."""
    model.eval()
    return torch.cat([torch.softmax(model(x[i:i + 64].to(device)), 1)[:, 1].cpu() for i in range(0, len(x), 64)]).numpy()

def train(model, x_train, y_train, x_val, y_val, lr, batch, patience):
    """Same loop as train.py: keep the epoch with the best validation AUC."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
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
    """Every galaxy gives a double and a single (2/3 matched thick, 1/3 thin) per view."""
    images, labels = [], []
    for raw in galaxies:
        for _ in range(views):
            s = random_settings()
            single = "thin" if rng.random() < 1 / 3 else "matched"
            images += [add_light(raw, rings("double", s)), add_light(raw, rings(single, s))]
            labels += [1, 0]
    return to_batch(images), torch.tensor(labels)

x_train, y_train = make_set(raws["train"], args.views)
x_val, y_val = make_set(raws["val"], 1)
say(f"Seed {args.seed}: {len(y_train)} training images, {len(y_val)} validation images on {device}")

# ---- 5. Train all three ---------------------------------------------------------------
stage("Training polar branch")
polar, auc = train(PolarClassifier(), x_train, y_train, x_val, y_val, lr=1e-3, batch=64, patience=5)
n_params = sum(p.numel() for p in polar.parameters())
say(f"  best validation AUC {auc:.3f} ({n_params / 1e3:.0f}k parameters)")

stage("Training wider polar branch")
wider, auc = train(WiderPolar(), x_train, y_train, x_val, y_val, lr=1e-3, batch=64, patience=5)
say(f"  best validation AUC {auc:.3f}")
torch.save(wider.state_dict(), out_dir / "polar_wider.pt")

stage("Fitting classifier on frozen Zoobot features")
frozen_net = timm.create_model("convnext_nano", pretrained=False, num_classes=0, in_chans=1)
state = torch.load(ROOT / f"runs/zoobot_lr0.0001_seed{args.seed}/best.pt", map_location="cpu")
frozen_net.load_state_dict({k: v for k, v in state.items() if not k.startswith("head.fc")})
frozen_net.to(device).eval()

@torch.no_grad()
def features(x):
    return torch.cat([frozen_net(to_model_input(x[i:i + 64].to(device))).cpu() for i in range(0, len(x), 64)]).numpy()

scaler = StandardScaler().fit(features(x_train))
f_train, f_val = scaler.transform(features(x_train)), scaler.transform(features(x_val))
best_auc = -1.0
for C in (0.001, 0.01, 0.1, 1.0):      # strength of the classifier, chosen on validation
    clf = LogisticRegression(C=C, max_iter=2000).fit(f_train, y_train.numpy())
    auc = roc_auc_score(y_val.numpy(), clf.decision_function(f_val))
    if auc > best_auc:
        best_auc, frozen_clf = auc, clf
say(f"  best validation AUC {best_auc:.3f}")

models = {
    "polar": lambda x: scores(polar, x),
    "polar wider": lambda x: scores(wider, x),
    "zoobot frozen": lambda x: frozen_clf.decision_function(scaler.transform(features(x))),
    "brightness only": lambda x: x.mean(dim=(1, 2, 3)).numpy(),
}
if not args.skip_finetune:
    stage("Fine-tuning Zoobot end to end (train.py's settings)")
    finetuned, auc = train(ZoobotClassifier(), x_train, y_train, x_val, y_val, lr=1e-4, batch=32, patience=3)
    say(f"  best validation AUC {auc:.3f}")
    torch.save(finetuned.state_dict(), out_dir / "zoobot_finetuned.pt")
    models["zoobot finetuned"] = lambda x: scores(finetuned, x)

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
            for shape in ("rings", "arcs"):
                for name, auc in test_cell("matched", shape=shape, **fixed).items():
                    rows.append(dict(model=name, width=width, distance=d, brightness=b, shape=shape, against="matched", auc=auc))
            # perfect circles, perfectly centred: what the unwrap is best at
            for name, auc in test_cell("matched", shape="rings", q=1.0, cx=0.0, cy=0.0, **fixed).items():
                rows.append(dict(model=name, width=width, distance=d, brightness=b, shape="round centred", against="matched", auc=auc))
            for name, auc in test_cell("thin", shape="rings", **fixed).items():
                rows.append(dict(model=name, width=width, distance=d, brightness=b, shape="rings", against="thin", auc=auc))
    print(f"  width {width} done", flush=True)
results = pd.DataFrame(rows)
results.to_csv(out_dir / "results.csv", index=False)

def table(frame, by):
    return frame.pivot_table(index=by, columns="model", values="auc").round(3).to_string()

hard = results[results["against"] == "matched"]
say("\nDouble vs matched thick single, oval and off-centre rings, by distance (0.5 = guessing, 1 = perfect):")
say(table(hard[hard["shape"] == "rings"], "distance"))
say("\nSame, arcs instead of full outer rings:")
say(table(hard[hard["shape"] == "arcs"], "distance"))
say("\nSame, perfect circles exactly centred:")
say(table(hard[hard["shape"] == "round centred"], "distance"))
say("\nFull rings at 1.2-1.3x, by outer ring brightness:")
say(table(hard[(hard["shape"] == "rings") & (hard["distance"] > 1.15)], "brightness"))
say("\nFull rings at 1.2-1.3x, by ring thickness:")
say(table(hard[(hard["shape"] == "rings") & (hard["distance"] > 1.15)], "width"))
say("\nDouble vs ordinary thin single (false-alarm check, should be high):")
say(table(results[results["against"] == "thin"], "distance"))

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
galaxy = raws["test"][0]
fig, axes = plt.subplots(4, 4, figsize=(11, 8.5))
for col, d in enumerate((1.2, 1.3)):
    s = random_settings(r1=R, inner=INNER, d=d, b=4, width=0.68, shape="rings", q=1.0, cx=0.0, cy=0.0)
    for row, kind in enumerate(("double", "matched")):
        x = to_batch([add_light(galaxy, rings(kind, s))])
        axes[row, 2 * col].imshow(x[0, 0][30:70, 30:70].numpy(), cmap="gray", origin="lower")
        axes[row, 2 * col].set_title(f"{kind}, {d}x", fontsize=9)
        axes[row, 2 * col + 1].imshow(unwrap(x)[0, 0].numpy(), cmap="gray", origin="lower", aspect="auto", extent=(0, 360, 0, 2))
        axes[row, 2 * col + 1].set_title("unwrapped", fontsize=9)
        profile = unwrap(x)[0, 0].mean(1).numpy()
        axes[row + 2, 2 * col].plot((np.arange(N_R) + 0.5) * R_MAX / N_R, profile)
        axes[row + 2, 2 * col].set_title("profile vs radius (\")", fontsize=9)
        axes[row + 2, 2 * col + 1].axis("off")
for ax in axes[:2].ravel():
    ax.set_xticks([]), ax.set_yticks([])
plt.tight_layout()
plt.savefig(out_dir / "examples.png", dpi=130)

stage("Done")
(out_dir / "summary.txt").write_text("\n".join(report) + "\n")
print(f"\nSaved to {out_dir.relative_to(ROOT)}/")
