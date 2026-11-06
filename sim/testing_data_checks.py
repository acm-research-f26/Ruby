# testing_data_checks.py: can the Testing-Data images be told apart without seeing a second ring?
# Reads the scores from sim/testing_data_metrics.py, so run that first. Nothing is trained except the simple-measure classifier.
#
# Usage (from ~/ruby, with lens_env311):
#   python -I sim/testing_data_checks.py data/testing_data
#
# Checks:
#   1. Ring size     the smallest ring radius of each lens (theta_E1 for compounds, theta_E for others), as a score on its own
#   2. Simple light  6 brightness measures per image, alone and combined in a logistic regression with 5-fold cross-validation
#   3. No giveaway   both models' AUROC using only compound lenses whose inner ring is 0.9" or larger
#
# Output: sim/out/testing_data/checks.txt

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

ROOT = Path.home() / "ruby"
DATA = ROOT / sys.argv[1]
CUT = 0.9
out_dir = ROOT / "sim/out/testing_data"

report = []
def say(line=""):
    print(line, flush=True)
    report.append(line)

def auc(yes, no):
    return roc_auc_score(np.r_[np.ones(len(yes)), np.zeros(len(no))], np.r_[yes, no])

labels = pd.read_csv(DATA / "test/labels.csv")
compound = labels.class_name.values == "compound"
inner = np.where(compound, labels.theta_E1, labels.theta_E)

say("1. Ring size")
for cls in ["single", "nearby"]:
    other = labels.class_name.values == cls
    say(f"  share with a ring under {CUT}\": compound {(inner[compound] < CUT).mean():.0%}, {cls} {(inner[other] < CUT).mean():.0%}; "
        f"smallest ring radius alone, compound vs. {cls}: AUROC {auc(-inner[compound], -inner[other]):.3f}")
real = pd.read_csv(ROOT / "data/zenodo_15003116/modeling_lens_mass.csv").einstein_radius_effective_median_pdf.astype(float).dropna()
say(f"  real Euclid lenses with a published radius under {CUT}\": {(real < CUT).mean():.0%} of {len(real)}")

say("\n2. Simple light measures (in noise units, within 3\" of the centre)")
yy, xx = np.mgrid[:100, :100]
R = np.hypot(yy - 49.5, xx - 49.5) * 0.1
rows = []
for i in labels.id:
    x = np.load(DATA / "test" / f"{i}.npy", allow_pickle=False)
    background = np.median(x)
    y = (x - background) / (1.4826 * np.median(np.abs(x - background)) + 1e-12)
    rows.append(dict(total_light=y[R < 3].clip(0).sum(), light_0p5_to_1=y[(R >= 0.5) & (R < 1)].clip(0).sum(),
                     light_1_to_2=y[(R >= 1) & (R < 2)].clip(0).sum(), light_outside_1=y[(R >= 1) & (R < 3)].clip(0).sum(),
                     bright_pixels=(y[R < 3] > 3).sum(), peak=y[R < 0.5].max()))
measures = pd.DataFrame(rows)
X = np.log1p(measures.values.clip(0))
for cls in ["single", "nearby", "non_lens", "all others"]:
    keep = compound | (labels.class_name.values == cls) if cls != "all others" else np.ones(len(labels), bool)
    alone = {k: round(auc(measures[k].values[keep & compound], measures[k].values[keep & ~compound]), 3) for k in measures}
    combined = cross_val_predict(LogisticRegression(max_iter=2000), X[keep], compound[keep], cv=5, method="predict_proba")[:, 1]
    say(f"  compound vs. {cls}: combined (5-fold) AUROC {roc_auc_score(compound[keep], combined):.3f}; alone {alone}")

say(f"\n3. Without the giveaway: only compound lenses with an inner ring of {CUT}\" or more")
scores = pd.read_csv(out_dir / "scores.csv").merge(labels[["id", "theta_E1"]], on="id")
for name in ["zoobot3", "baseline", "lens only"]:
    rows = []
    for seed in (1, 2, 3):
        g = scores[(scores.model == name) & (scores.seed == seed)]
        wide = g[(g.class_name == "compound") & (g.theta_E1 >= CUT)].score
        narrow = g[(g.class_name == "compound") & (g.theta_E1 < CUT)].score
        rows.append((auc(wide, g[g.class_name != "compound"].score), auc(wide, g[g.class_name == "single"].score),
                     auc(narrow, g[g.class_name == "single"].score)))
    r = np.array(rows)
    say(f"  {name:9s} wide-ring compounds (n={len(wide)}) vs. all others {np.round(r[:, 0], 3).tolist()} mean {r[:, 0].mean():.3f}; "
        f"vs. single {np.round(r[:, 1], 3).tolist()} mean {r[:, 1].mean():.3f}; narrow-ring compounds vs. single mean {r[:, 2].mean():.3f}")

(out_dir / "checks.txt").write_text("\n".join(report) + "\n")
