# transplant_test.py: do real outer rings raise the score more than simulated outer sources with the same light?
# Run after Step 4, so this explains the result and can't change the verdict in sim/out/final/results.txt.
#
# Usage (from ~/ruby, with lens_env311):
#   python -I sim/transplant_test.py
#
# Lifting: in each compound lens's outer zone (the zones in sim/second_ring_test.py), keep light above a wide median
# (which holds the smooth galaxy), lightly smoothed, where it is above 1.5x its own noise, in patches of 6 pixels or more.
# Targets: ordinary real lenses from the validation and test sets with a Euclid radius within 30% of the compound lens's
# inner radius. Matching by size means the ring's pixels don't have to be rescaled. Light is carried over in noise units.
# Each target gets 8 versions of each kind:
#   ring            the real ring, in its 8 flips and quarter turns
#   simulated       a simulated source at the ring's mean radius with the same total light (8 different sources)
#   control         light lifted the same way from the same zone of another ordinary lens of similar size
#   control scaled  the same, scaled up or down to the ring's total light
#
# How to read it, set before scoring, using each ring's mean rise in the score over its targets and the 3 seeds:
#   test doesn't work   the mean rise from control is more than a third of the mean rise from ring
#   shape explains it   ring rises at least 2x as much as simulated, on at least 3 of 4 rings
#   not shape           ring rises within 25% of simulated, on at least 3 of 4 rings
#   mixed               anything else
#
# Output (in sim/out/final/transplant/): results.txt, results.csv, examples.png

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, label, median_filter
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate as sim
from simulate import ROOT

SEEDS = [1, 2, 3]
VERSIONS = 8
DONORS = {"469177": dict(name="Cosmic Dartboard", inner=0.45, zone=(0.9, 2.4), m_s2=23.2),
          "107505": dict(name="Galileo's Lens", inner=1.09, zone=(1.6, 2.3), m_s2=24.4),
          "726389": dict(name="Cosmic Ammonite", inner=1.18, zone=(1.4, 1.9), m_s2=24.2),
          "378903": dict(name="Teapot Lens", inner=0.62, zone=(0.95, 1.7), m_s2=24.0)}
device = "mps" if torch.backends.mps.is_available() else "cpu"
out_dir = ROOT / "sim/out/final/transplant"
out_dir.mkdir(parents=True, exist_ok=True)

report = []
def say(line=""):
    print(line, flush=True)
    report.append(line)

yy, xx = np.mgrid[:100, :100]
R = np.hypot(yy - 49.5, xx - 49.5) * 0.1

def preprocess(pixels):
    x = np.nan_to_num(pixels.astype(np.float32))
    background = np.median(x)
    noise = 1.4826 * np.median(np.abs(x - background)) + 1e-12
    x = np.arcsinh((x - background) / noise)
    lo, hi = np.percentile(x, [1, 99.5])
    return np.clip((x - lo) / (hi - lo + 1e-12), 0, 1).astype(np.float32)

def to_tensor(images):
    return torch.from_numpy(np.stack([preprocess(x) for x in images]))[:, None]

def zoobot3(seed):
    net = timm.create_model("convnext_nano", pretrained=False, num_classes=3, in_chans=1)
    state = torch.load(ROOT / f"sim/out/models2/seed{seed}/zoobot3.pt", map_location="cpu")
    net.load_state_dict({k.removeprefix("net."): v for k, v in state.items()})
    return net.to(device).eval()

@torch.no_grad()
def p_compound(net, x):
    out = [torch.softmax(net(F.interpolate(x[i:i + 64].to(device), size=(224, 224), mode="bilinear", align_corners=False)), 1).cpu()
           for i in range(0, len(x), 64)]
    return torch.cat(out).numpy()[:, 2]

def lift(x, zone):
    """Thin light in a zone, in noise units of that image."""
    background, noise = sim.noise_level(x)
    y = (x - background) / noise
    res = gaussian_filter(y - median_filter(y, size=15), 0.7)
    far = res[R > 3.5]
    level = 1.4826 * np.median(np.abs(far - np.median(far)))
    keep = (res > 1.5 * level) & (R >= zone[0]) & (R < zone[1])
    patches, count = label(keep)
    sizes = np.bincount(patches.ravel(), minlength=count + 1)
    return np.where((sizes[patches] >= 6) & (patches > 0), res, 0.0)

def turn(image, k):
    out = np.rot90(image, k % 4)
    return np.fliplr(out) if k >= 4 else out

split, lens_models = sim.load_inputs()
table = pd.read_csv(ROOT / "data/zenodo_15003116/modeling_lens_mass.csv")
e = np.hypot(table.ell_comps_0_median_pdf.astype(float), table.ell_comps_1_median_pdf.astype(float))
table = pd.DataFrame(dict(id=table.id_str, theta_e=table.einstein_radius_effective_median_pdf.astype(float), q=(1 - e) / (1 + e)))
ordinary = split[split.split.isin(["val", "test"]) & (split.label == 1) & ~split.notes.eq("dspl")].merge(table, on="id")
ordinary = ordinary[ordinary.theta_e.between(0.2, 3)].reset_index(drop=True)
clean = {i: np.nan_to_num(np.load(ROOT / p).astype(np.float32)) for i, p in enumerate(ordinary.path)}
compound = split[split.notes.eq("dspl")].copy()
compound["key"] = compound.id.str[-6:]
nets = {k: zoobot3(k) for k in SEEDS}
ranks = pd.read_csv(ROOT / "sim/out/final/ranks.csv", dtype={"id": str})
ranks = ranks[(ranks.model == "zoobot3") & ranks.compound]

rows, per_seed, pictures = [], [], []
for key, info in DONORS.items():
    donor = compound[compound.key == key].iloc[0]
    x = np.nan_to_num(np.load(ROOT / donor.path).astype(np.float32))
    ring = lift(x, info["zone"])
    total = ring.sum()
    radius = (ring * R).sum() / total
    published = 10 ** ((23.9 - info["m_s2"]) / 2.5) / sim.noise_level(x)[1]
    group = ordinary.index[((ordinary.theta_e / info["inner"]) - 1).abs() <= 0.3].values
    rng = np.random.default_rng(int(key))
    images = {"plain": [], "ring": [], "simulated": [], "control": [], "control scaled": []}
    control_totals = []
    for target in group:
        raw, row = clean[target], ordinary.loc[target]
        _, noise = sim.noise_level(raw)
        angle = sim.host_angle(raw)
        images["plain"].append(raw)
        others = rng.choice(group[group != target], VERSIONS, replace=len(group) - 1 < VERSIONS)
        for version in range(VERSIONS):
            images["ring"].append(raw + noise * turn(ring, version))
            d = max((radius - 0.1) / row.theta_e, 1.05)
            system = sim.draw_system(rng, lens_models, angle=angle, theta_e=row.theta_e, q=row.q, d=d, b=1.0, companion=False)
            _, unit = sim.lensed_light(system)
            images["simulated"].append(raw + noise * total * unit / unit.sum())
            lifted = turn(lift(clean[others[version]], info["zone"]), version)
            control_totals.append(lifted.sum())
            images["control"].append(raw + noise * lifted)
            images["control scaled"].append(raw + noise * lifted * (total / lifted.sum() if lifted.sum() > 0 else 0))
    tensors = {k: to_tensor(v) for k, v in images.items()}
    n = len(group)
    summary = dict(lens=info["name"], targets=n, ring_light=total, published_light=published, ring_radius=radius, control_light=np.mean(control_totals))
    scores = {k: {seed: p_compound(net, v) for seed, net in nets.items()} for k, v in tensors.items()}
    for kind in ["ring", "simulated", "control", "control scaled"]:
        rises, aucs, wins, levels = [], [], [], []
        for seed in SEEDS:
            base = scores["plain"][seed]
            values = scores[kind][seed]
            shaped = values.reshape(n, VERSIONS)
            rises.append((shaped - base[:, None]).mean())
            wins.append((shaped > base[:, None]).mean())
            aucs.append(roc_auc_score(np.r_[np.ones(len(values)), np.zeros(n)], np.r_[values, base]))
            levels.append(values.mean())
            per_seed.append(dict(lens=info["name"], kind=kind, seed=seed, rise=rises[-1], auc=aucs[-1], beats_own=wins[-1], mean_score=levels[-1]))
        summary[f"rise {kind}"], summary[f"auc {kind}"], summary[f"own {kind}"], summary[f"score {kind}"] = np.mean(rises), np.mean(aucs), np.mean(wins), np.mean(levels)
    summary["score plain"] = np.mean([scores["plain"][s].mean() for s in SEEDS])
    summary["own score"] = ranks[ranks.id == donor.id].score.mean()
    rows.append(summary)
    pictures.append((info["name"], x, ring, images["plain"][0], images["ring"][0], images["simulated"][0], images["control"][0]))

frame = pd.DataFrame(rows)
frame.to_csv(out_dir / "results.csv", index=False)
pd.DataFrame(per_seed).to_csv(out_dir / "per_seed.csv", index=False)
for _, r in frame.iterrows():
    say(f"{r.lens}: {r.targets} target lenses. Lifted light {r.ring_light:.0f} noise units (published source 2: {r.published_light:.0f}), mean radius {r.ring_radius:.2f}\"; control lifts {r.control_light:.0f}")
    say(f"  plain score {r['score plain']:.2f}; own score {r['own score']:.2f}")
    for kind in ["ring", "simulated", "control", "control scaled"]:
        say(f"  {kind:15s} rise {r[f'rise {kind}']:+.3f}, score {r[f'score {kind}']:.2f}, AUC {r[f'auc {kind}']:.3f}, raises its own lens {r[f'own {kind}']:.0%}")

say("\nRise by seed:")
say(pd.DataFrame(per_seed).pivot_table(index=["lens", "kind"], columns="seed", values="rise").round(3).to_string())

works = frame["rise control"].mean() <= frame["rise ring"].mean() / 3
double = (frame["rise ring"] >= 2 * frame["rise simulated"]).sum()
close = ((frame["rise ring"] / frame["rise simulated"] - 1).abs() <= 0.25).sum()
reading = "the test doesn't work" if not works else "shape explains it" if double >= 3 else "not shape" if close >= 3 else "mixed"
say(f"\nMean rise: ring {frame['rise ring'].mean():+.3f}, simulated {frame['rise simulated'].mean():+.3f}, control {frame['rise control'].mean():+.3f}, control scaled {frame['rise control scaled'].mean():+.3f}")
say(f"Rings rising at least 2x simulated: {double} of 4; within 25%: {close} of 4. Reading set beforehand: {reading}")

fig, axes = plt.subplots(4, 6, figsize=(11, 7.6))
titles = ["compound lens", "lifted ring", "ordinary lens", "+ real ring", "+ simulated", "+ control"]
for line, (name, *panels) in zip(axes, pictures):
    for ax, panel, title in zip(line, panels, titles):
        ax.imshow(panel if title == "lifted ring" else preprocess(panel), cmap="gray", origin="lower")
        ax.set_title(f"{name}\n{title}" if title == "compound lens" else title, fontsize=8)
        ax.axis("off")
fig.tight_layout()
fig.savefig(out_dir / "examples.png", dpi=130)

(out_dir / "results.txt").write_text("\n".join(report) + "\n")
print("Done.", flush=True)
