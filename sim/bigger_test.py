# bigger_test.py: the frozen two-step setup on about 5,000 new non-lenses and the grade C lens candidates.
# These images are from sim/get_more_images.py, and no model or earlier test has used them. No model is trained or changed.
# Run after Step 4, so this can't change the verdict in sim/out/final/results.txt.
#
# Usage (from ~/ruby, with lens_env311):
#   python -I sim/bigger_test.py
#
# Same steps as sim/two_step.py. Step one keeps an image if the baseline's P(lens) clears a cutoff that keeps 95% of validation
# lenses, separately for each seed. Step two flags a kept image if the same seed's 3-answer Zoobot gives P(compound) > 0.5.
#
# How to read it, set before scoring:
#   New non-lenses, two-step flag rate:
#     under 0.1% on every seed   could help narrow a search of a whole catalog
#     otherwise                  stays a second step on lenses already found
#   Images with blocks of zero pixels count in the main number, as they did in the earlier test. The rate without them is
#   shown too, but doesn't decide the reading.
#   A few new non-lenses may be real lenses nobody has found, so the top ones are shown on a sheet.
#   Grade C candidates: experts weren't sure about these, so there is no pass or fail. They are compared with the ordinary
#   validation and test lenses, and the top 20 go on a sheet. A place on the sheet doesn't make one a compound lens.
#
# Trust check: the old test images must give the same scores as sim/out/final/two_step/scores.csv (within 0.01).
#
# Output (in sim/out/expand/): results.txt, scores.csv, grade_c.csv, non_lenses_top.csv, sheet_grade_c.png, sheet_non_lenses.png

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

ROOT = Path.home() / "ruby"
SEEDS = [1, 2, 3]
KEEP = 0.95
LIMIT = 0.001
GALAXIES = 1.09e6
TOP = 20
MAX_ZERO = 0.005
NAMES = {"107505": "Galileo's Lens", "726389": "Cosmic Ammonite", "469177": "Cosmic Dartboard", "378903": "Teapot Lens"}
device = "mps" if torch.backends.mps.is_available() else "cpu"
out_dir = ROOT / "sim/out/expand"
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

def interval(hits, n, z=1.96):
    """95% Wilson interval for a share."""
    p = hits / n
    middle = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(middle - half, 0), min(middle + half, 1)

split = pd.read_csv(ROOT / "data/split.csv", dtype={"id": str})
val_lenses = split[(split.split == "val") & (split.label == 1)].reset_index(drop=True)
test = split[split.split == "test"].reset_index(drop=True)
ordinary = split[split.split.isin(["val", "test"]) & (split.label == 1) & ~split.notes.eq("dspl")].reset_index(drop=True)
new = pd.read_csv(ROOT / "data/expand/manifest.csv", dtype={"id": str})
non_rows = new[new.kind == "non_lenses"].reset_index(drop=True)
c_rows = new[new.kind == "grade_c"].reset_index(drop=True)
assert not set(non_rows.id) & set(split.id), "a new non-lens is already in the old split"
say(f"{len(non_rows)} new non-lenses, {len(c_rows)} grade C candidates, {len(ordinary)} ordinary validation and test lenses")

x_val, x_test, x_ordinary, x_non, x_c = load(val_lenses), load(test), load(ordinary), load(non_rows), load(c_rows)
broken = np.array([(np.nan_to_num(np.load(ROOT / p)) == 0).mean() >= MAX_ZERO for p in non_rows.path])
is_lens, is_compound = test.label.eq(1).values, test.notes.eq("dspl").values
old_scores = pd.read_csv(ROOT / "sim/out/final/two_step/scores.csv", dtype={"id": str})

scores, table, drift = [], [], []
non_lens, non_compound, c_lens, c_compound, ord_kept_flag, known = {}, {}, {}, {}, {}, {}
for seed in SEEDS:
    baseline = zoobot(2, ROOT / f"runs/zoobot_lr0.0001_seed{seed}/best.pt")
    trained = zoobot(3, ROOT / f"sim/out/models2/seed{seed}/zoobot3.pt")
    cutoff = np.percentile(softmax(baseline, x_val)[:, 1], 100 * (1 - KEEP))

    test_lens, test_answers = softmax(baseline, x_test)[:, 1], softmax(trained, x_test)
    old = old_scores[old_scores.seed == seed].set_index("id").loc[test.id]
    drift.append(max(np.abs(test_lens - old.p_lens.values).max(), np.abs(test_answers[:, 2] - old.p_compound.values).max()))

    non_lens[seed], non_answers = softmax(baseline, x_non)[:, 1], softmax(trained, x_non)
    non_compound[seed] = non_answers[:, 2]
    c_lens[seed], c_compound[seed] = softmax(baseline, x_c)[:, 1], softmax(trained, x_c)[:, 2]
    ord_lens, ord_compound = softmax(baseline, x_ordinary)[:, 1], softmax(trained, x_ordinary)[:, 2]
    known[seed] = test_answers[is_compound, 2]

    kept, flagged = non_lens[seed] >= cutoff, non_compound[seed] > 0.5
    both = kept & flagged
    table.append(dict(seed=seed, cutoff=cutoff, non_n=len(non_rows), non_kept=kept.sum(), non_alone=flagged.sum(), non_flagged=both.sum(),
                      rate=both.mean(), low=interval(both.sum(), len(both))[0], high=interval(both.sum(), len(both))[1],
                      rate_unbroken=both[~broken].mean(),
                      baseline_auc_new=roc_auc_score(np.r_[np.ones(is_lens.sum()), np.zeros(len(both))], np.r_[test_lens[is_lens], non_lens[seed]]),
                      baseline_auc_old=roc_auc_score(is_lens, test_lens),
                      alone_auc_new=roc_auc_score(np.r_[np.ones(is_lens.sum()), np.zeros(len(both))], np.r_[1 - test_answers[is_lens, 0], 1 - non_answers[:, 0]]),
                      c_kept=(c_lens[seed] >= cutoff).mean(), c_flagged=((c_lens[seed] >= cutoff) & (c_compound[seed] > 0.5)).mean(),
                      ordinary_kept=(ord_lens >= cutoff).mean(), ordinary_flagged=((ord_lens >= cutoff) & (ord_compound > 0.5)).mean(),
                      compound_flagged=((test_lens >= cutoff) & (test_answers[:, 2] > 0.5))[is_compound].sum()))
    ord_kept_flag[seed] = cutoff
    for name, rows, lens, compound in [("non_lens", non_rows, non_lens[seed], non_compound[seed]), ("grade_c", c_rows, c_lens[seed], c_compound[seed])]:
        scores.append(pd.DataFrame(dict(seed=seed, kind=name, id=rows.id, p_lens=lens, p_compound=compound, kept=lens >= cutoff)))

frame = pd.DataFrame(table)
pd.concat(scores).to_csv(out_dir / "scores.csv", index=False)
say(f"\nTrust check: largest score change on old test images, per seed {np.round(drift, 4).tolist()} (limit 0.01): {'ok' if max(drift) <= 0.01 else 'FAILED'}")
say("Known compound lenses flagged, of 4, per seed: " + str(frame.compound_flagged.tolist()))

say("\nNew non-lenses")
for _, r in frame.iterrows():
    say(f"  seed {int(r.seed)}: step one keeps {int(r.non_kept)}; 3-answer alone flags {int(r.non_alone)}; two steps flag {int(r.non_flagged)} of {int(r.non_n)} = {r.rate:.3%} (95% interval {r.low:.3%} to {r.high:.3%}); {r.rate_unbroken:.3%} without broken images")
say(f"  images with blocks of zero pixels: {broken.sum()} of {len(broken)}")
all_three = np.all([(non_lens[k] >= frame.cutoff[i]) & (non_compound[k] > 0.5) for i, k in enumerate(SEEDS)], axis=0)
say(f"  flagged on all 3 seeds: {all_three.sum()}")
say(f"  on 1.09M galaxies, per seed: {[int(round(r * GALAXIES)) for r in frame.rate]} false alarms (rough, assumes this sample is representative)")
say("  earlier estimate, from 300 non-lenses: 1.2%")
reading = "could help narrow a search of a whole catalog" if (frame.rate < LIMIT).all() else "stays a second step on lenses already found"
say(f"  Reading set beforehand: {reading}")

say("\nLens finding on the 79 test lenses, against the new non-lenses (old 300 in brackets)")
for _, r in frame.iterrows():
    say(f"  seed {int(r.seed)}: baseline {r.baseline_auc_new:.4f} ({r.baseline_auc_old:.4f}); 3-answer alone {r.alone_auc_new:.4f}")

say("\nGrade C candidates vs. ordinary validation and test lenses (no pass or fail)")
for _, r in frame.iterrows():
    say(f"  seed {int(r.seed)}: step one keeps {r.c_kept:.1%} vs. {r.ordinary_kept:.1%}; two steps flag {r.c_flagged:.1%} vs. {r.ordinary_flagged:.1%}")

def summary(rows, lens, compound):
    out = rows[["id", "ra", "dec", "grade", "notes", "path"]].copy()
    for i, seed in enumerate(SEEDS):
        out[f"score_seed{seed}"] = compound[seed]
    columns = [f"score_seed{k}" for k in SEEDS]
    out["score"], out["lowest_seed"] = out[columns].mean(axis=1), out[columns].min(axis=1)
    out["kept_all_seeds"] = np.all([lens[k] >= frame.cutoff[i] for i, k in enumerate(SEEDS)], axis=0)
    out["flagged_all_seeds"] = out.kept_all_seeds & (out.lowest_seed > 0.5)
    return out.sort_values(["kept_all_seeds", "score"], ascending=False).reset_index(drop=True)

c_table, non_table = summary(c_rows, c_lens, c_compound), summary(non_rows, non_lens, non_compound)
c_table.drop(columns="path").round(4).to_csv(out_dir / "grade_c.csv", index=False)
non_table.head(100).drop(columns="path").round(4).to_csv(out_dir / "non_lenses_top.csv", index=False)
lowest_known = np.mean([known[k] for k in SEEDS], axis=0).min()
say(f"\nLowest mean score of a known compound lens: {lowest_known:.2f}")
say(f"  grade C at or above it and kept on all seeds: {(c_table.kept_all_seeds & (c_table.score >= lowest_known)).sum()}; flagged on all 3 seeds: {c_table.flagged_all_seeds.sum()}")
say(f"  new non-lenses at or above it and kept on all seeds: {(non_table.kept_all_seeds & (non_table.score >= lowest_known)).sum()}")

show = lambda ax, image: ax.imshow(image[20:80, 20:80], cmap="gray", origin="lower", extent=(-3, 3, -3, 3))
known_rows = test[is_compound].copy()
known_rows["score"] = np.mean([known[k] for k in SEEDS], axis=0)
def sheet(table, note, file):
    fig, axes = plt.subplots(5, 5, figsize=(11, 12.2))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, row) in zip(axes[0], known_rows.iterrows()):
        show(ax, preprocess(np.load(ROOT / row.path)))
        ax.set_title(f"KNOWN: {NAMES[row.id[-6:]]}\nscore {row.score:.2f}", fontsize=9, fontweight="bold")
    axes[0, 4].text(0.5, 0.5, note + "\nEach panel is 6\" across.", ha="center", va="center", fontsize=9, transform=axes[0, 4].transAxes)
    for ax, (_, row) in zip(axes[1:].ravel(), table[table.kept_all_seeds].head(TOP).iterrows()):
        show(ax, preprocess(np.load(ROOT / row.path)))
        ax.set_title(f"score {row.score:.2f}, lowest seed {row.lowest_seed:.2f}\n{row.ra:.4f}, {row.dec:.4f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / file, dpi=140)
sheet(c_table, "Top row: the 4 known\ncompound lenses.\nBelow: highest-scoring\ngrade C candidates,\nunconfirmed.", "sheet_grade_c.png")
sheet(non_table, "Top row: the 4 known\ncompound lenses.\nBelow: new non-lenses\nthe model scores highest.", "sheet_non_lenses.png")

(out_dir / "results.txt").write_text("\n".join(report) + "\n")
print("Done.", flush=True)
