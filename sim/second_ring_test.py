# second_ring_test.py: does the 3-answer Zoobot respond to a second ring, or to any extra structure around a lens?
# Run after Step 4, so this explains the result and can't change the verdict in sim/out/final/results.txt.
#
# Usage (from ~/ruby, with lens_env311):
#   python -I sim/second_ring_test.py
#
# Part 1, add light to real VALIDATION lenses (those with a Euclid radius). Every version adds the same total light:
#   outer     a simulated source lensed to the outer distance, d = 1.3-1.8 times the inner ring radius
#   same      the same source lensed to the inner ring's radius (d = 1): same light, same source, wrong place
#   blob      one small unlensed galaxy at the outer distance: clutter, not a ring
# Brightness 1, 2, 3, 4, 6 and 5 versions each. Only 2 and up count below, since the model can't see fainter ones.
# Rule, set before scoring:
#   second ring      on all 3 seeds, outer raises the score at least 2x as much as same and as blob,
#                    and the 95% bootstrap interval (over lenses) of each difference is above 0
#   extra structure  on all 3 seeds, same and blob each raise it at least 75% as much as outer
#   mixed            anything else
#
# Part 2, hide part of the 4 real compound lenses. Pixels in a zone are replaced with the lower quarter of the light at
# that radius plus noise, so the smooth galaxy stays and arcs go. Zones (arcsec) come from Li et al. 2026 Table 2 and the
# radial light of each image, and were fixed before scoring:
#   outer     the outer ring
#   inner     the inner ring
#   control   the same width as outer, beyond the outer ring
# The same zones are hidden on the 75 ordinary test lenses. Change in the score is measured in log-odds, because the
# ordinary lenses start much lower.
# Rule, set before scoring:
#   outer ring  on at least 3 of 4 lenses, hiding outer lowers the score more than on 90% of ordinary lenses,
#               and at least 2x as much as hiding control
#
# Output (in sim/out/final/checks/): second_ring.txt, added_light.csv, hidden_zones.csv, added_light.png, hidden_zones.png

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
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate as sim
from simulate import ROOT

SEEDS = [1, 2, 3]
BRIGHTNESS = [1, 2, 3, 4, 6]
VERSIONS = 5
DRAWS = 10
ZONES = {"469177": dict(name="Cosmic Dartboard", inner=(0.3, 0.7), outer=(0.9, 2.4), control=(2.5, 4.0)),
         "726389": dict(name="Cosmic Ammonite", inner=(0.9, 1.4), outer=(1.4, 1.9), control=(2.2, 2.7)),
         "107505": dict(name="Galileo's Lens", inner=(1.0, 1.6), outer=(1.6, 2.3), control=(2.6, 3.3)),
         "378903": dict(name="Teapot Lens", inner=(0.4, 0.95), outer=(0.95, 1.7), control=(2.0, 2.75))}
device = "mps" if torch.backends.mps.is_available() else "cpu"
out_dir = ROOT / "sim/out/final/checks"
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

nets = {k: zoobot3(k) for k in SEEDS}
split, lens_models = sim.load_inputs()

# ---- Part 1. Add light to real validation lenses --------------------------------------------
table = pd.read_csv(ROOT / "data/zenodo_15003116/modeling_lens_mass.csv")
e = np.hypot(table.ell_comps_0_median_pdf.astype(float), table.ell_comps_1_median_pdf.astype(float))
table = pd.DataFrame(dict(id=table.id_str, theta_e=table.einstein_radius_effective_median_pdf.astype(float), q=(1 - e) / (1 + e)))
known = split[(split.split == "val") & (split.label == 1)].merge(table, on="id")
known = known[known.theta_e.between(0.2, 3)].reset_index(drop=True)

rows, images = [], {"plain": [], "outer": [], "same": [], "blob": []}
for lens, row in known.iterrows():
    raw = np.nan_to_num(np.load(ROOT / row.path).astype(np.float32))
    _, noise = sim.noise_level(raw)
    angle = sim.host_angle(raw)
    images["plain"].append(raw)
    for b in BRIGHTNESS:
        for version in range(VERSIONS):
            key = [2028, lens, int(b), version]
            chooser = np.random.default_rng(key + [0])
            d = chooser.uniform(1.3, 1.8)
            shared = dict(angle=angle, theta_e=row.theta_e, q=row.q, b=b, companion=False)
            far = sim.draw_system(np.random.default_rng(key), lens_models, d=d, **shared)
            near = sim.draw_system(np.random.default_rng(key), lens_models, d=1.0, **shared)
            _, unit_far = sim.lensed_light(far)
            _, unit_near = sim.lensed_light(near)
            outer = (b * noise / sim.peak_level(unit_far)) * unit_far
            total = outer.sum()
            t = chooser.uniform(0, 2 * np.pi)
            blob = sim.observe(sim.source_light(dict(x=d * row.theta_e * np.cos(t), y=d * row.theta_e * np.sin(t), re=0.15, n=1.0,
                                                     q=chooser.uniform(0.5, 1), phi=chooser.uniform(0, np.pi), clumps=[]), sim.FX, sim.FY))
            images["outer"].append(raw + outer)
            images["same"].append(raw + unit_near * total / unit_near.sum())
            images["blob"].append(raw + blob * total / blob.sum())
            rows.append(dict(lens=lens, b=b, version=version, d=d))
frame = pd.DataFrame(rows)
tensors = {k: to_tensor(v) for k, v in images.items()}

say(f"Part 1: {len(known)} real validation lenses with a Euclid radius; every version adds the same total light")
clear = frame.b.values >= 2
results = {}
for seed, net in nets.items():
    plain = p_compound(net, tensors["plain"])
    base = plain[frame.lens.values]
    for arm in ["outer", "same", "blob"]:
        frame[f"rise_{arm}_{seed}"] = p_compound(net, tensors[arm]) - base
    rises = frame[clear].groupby("lens")[[f"rise_{a}_{seed}" for a in ["outer", "same", "blob"]]].mean().values
    rng = np.random.default_rng(seed)
    picks = rng.integers(len(rises), size=(5000, len(rises)))
    interval = lambda a: np.percentile((rises[:, 0] - rises[:, a])[picks].mean(1), [2.5, 97.5])
    aucs = {a: roc_auc_score(np.r_[np.ones(clear.sum()), np.zeros(clear.sum())], np.r_[frame[f"rise_{a}_{seed}"][clear] + base[clear], base[clear]]) for a in ["outer", "same", "blob"]}
    results[seed] = dict(outer=rises[:, 0].mean(), same=rises[:, 1].mean(), blob=rises[:, 2].mean(), low_same=interval(1), low_blob=interval(2), aucs=aucs)
    r = results[seed]
    say(f"  seed {seed}, brightness 2+: mean rise in score, outer {r['outer']:+.3f}, same radius {r['same']:+.3f}, blob {r['blob']:+.3f}; "
        f"with vs. without AUC {aucs['outer']:.3f}, {aucs['same']:.3f}, {aucs['blob']:.3f}")
    say(f"    outer minus same, 95% interval {r['low_same'].round(3)}; outer minus blob {r['low_blob'].round(3)}")
frame.to_csv(out_dir / "added_light.csv", index=False)

say("  mean rise by brightness, all seeds:")
long = frame.melt(id_vars=["b"], value_vars=[c for c in frame if c.startswith("rise_")])
long["arm"] = long.variable.str.split("_").str[1]
say(long.pivot_table(index="arm", columns="b", values="value").round(3).to_string())

second = all(r["outer"] >= 2 * r["same"] and r["outer"] >= 2 * r["blob"] and r["low_same"][0] > 0 and r["low_blob"][0] > 0 for r in results.values())
extra = all(r["same"] >= 0.75 * r["outer"] and r["blob"] >= 0.75 * r["outer"] for r in results.values())
say(f"  Verdict for part 1: {'second ring' if second else 'extra structure' if extra else 'mixed'}")

examples = frame[(frame.b == 3) & (frame.version == 0)].head(3).index
titles = ["real lens", "+ outer source", "+ same light at inner radius", "+ blob"]
fig, axes = plt.subplots(3, 4, figsize=(8, 6.2))
for line, index in zip(axes, examples):
    lens = frame.lens[index]
    panels = [images["plain"][lens], images["outer"][index], images["same"][index], images["blob"][index]]
    for ax, panel, title in zip(line, panels, titles):
        ax.imshow(preprocess(panel), cmap="gray", origin="lower")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
fig.tight_layout()
fig.savefig(out_dir / "added_light.png", dpi=130)
plt.close(fig)

# ---- Part 2. Hide zones on the 4 real compound lenses --------------------------------------
lenses = split[(split.split == "test") & (split.label == 1)].reset_index(drop=True)
is_compound = lenses.notes.eq("dspl").values
clean = [np.nan_to_num(np.load(ROOT / p).astype(np.float32)) for p in lenses.path]
yy, xx = np.mgrid[:100, :100]
R = np.hypot(yy - 49.5, xx - 49.5) * 0.1

def hide(x, zone, rng):
    _, noise = sim.noise_level(x)
    y = x.copy()
    for a in np.arange(zone[0], zone[1], 0.1):
        ring = (R >= a) & (R < a + 0.1)
        if ring.any():
            y[ring] = np.percentile(x[ring], 25) + rng.normal(0, noise, ring.sum())
    return y

logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
mean_score = lambda x: np.mean([p_compound(net, x) for net in nets.values()], axis=0)   # one number per image, over seeds
full = mean_score(to_tensor(clean))
ordinary = np.flatnonzero(~is_compound)

say("\nPart 2: hide a zone on the real compound lenses. Score = mean over 3 seeds and 10 noise draws")
rows, hits, pictures = [], 0, []
for i in np.flatnonzero(is_compound):
    zones = ZONES[lenses.id[i][-6:]]
    line = {}
    for kind in ["outer", "inner", "control"]:
        rng = np.random.default_rng([2029, int(i), len(kind)])
        hidden = [hide(clean[i], zones[kind], rng) for _ in range(DRAWS)]
        after = mean_score(to_tensor(hidden)).mean()
        fall = logit(full[i]) - logit(after)
        others = []
        for j in ordinary:
            shots = [hide(clean[j], zones[kind], rng) for _ in range(3)]
            others.append(logit(full[j]) - logit(mean_score(to_tensor(shots)).mean()))
        others = np.array(others)
        line[kind] = dict(after=after, fall=fall, beats=(fall > others).mean())
        rows.append(dict(lens=zones["name"], zone=kind, start=zones[kind][0], end=zones[kind][1], full=full[i], after=after, fall=fall,
                         ordinary_fall_median=np.median(others), ordinary_fall_90=np.percentile(others, 90), beats_ordinary=line[kind]["beats"]))
        if kind == "outer":
            pictures.append((zones["name"], clean[i], hidden[0], full[i], after))
    hit = line["outer"]["beats"] > 0.9 and line["outer"]["fall"] >= 2 * line["control"]["fall"]
    hits += hit
    say(f"  {zones['name']:17s} full {full[i]:.2f}. Hide outer {zones['outer']}: {line['outer']['after']:.2f} (falls more than {line['outer']['beats']:.0%} of ordinary); "
        f"hide inner {zones['inner']}: {line['inner']['after']:.2f}; hide control {zones['control']}: {line['control']['after']:.2f}. {'Meets' if hit else 'Misses'} the rule")
pd.DataFrame(rows).to_csv(out_dir / "hidden_zones.csv", index=False)
say(f"  Verdict for part 2: {'outer ring' if hits >= 3 else 'not shown'} ({hits} of 4 lenses)")

fig, axes = plt.subplots(2, 4, figsize=(8, 4.3))
for col, (name, before, after_image, p_before, p_after) in enumerate(pictures):
    axes[0, col].imshow(preprocess(before), cmap="gray", origin="lower")
    axes[0, col].set_title(f"{name}: {p_before:.2f}", fontsize=8)
    axes[1, col].imshow(preprocess(after_image), cmap="gray", origin="lower")
    axes[1, col].set_title(f"outer ring hidden: {p_after:.2f}", fontsize=8)
    axes[0, col].axis("off")
    axes[1, col].axis("off")
fig.tight_layout()
fig.savefig(out_dir / "hidden_zones.png", dpi=130)
plt.close(fig)

(out_dir / "second_ring.txt").write_text("\n".join(report) + "\n")
print("Done.", flush=True)
