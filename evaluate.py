# evaluate.py: compare every finished training run on the held-out test set
#
# Usage:  python evaluate.py
#
# Reads runs/*/test_scores.csv (written by train.py) and reports, per run:
#   - AUC on the whole test set (all lenses vs non-lenses)
#   - where each of the 4 compound lenses ranks (1 = the model's top pick)
#   - inspection cost: how many images a human must check to find all 4
#   - how many ordinary lenses land in the top N
# Then averages each model over its seeds.
#
# Output: results/summary.csv, results/compound_ranks.csv, results/compound_ranks.png

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

os.makedirs("results", exist_ok=True)
runs = sorted(glob.glob("runs/*/test_scores.csv"))
if not runs:
    raise SystemExit("No runs found. Train first: python train.py --model zoobot --seed 1")

summary, ranks = [], []
for path in runs:
    run = os.path.basename(os.path.dirname(path))          # e.g. "zoobot_lr0.0001_seed1"
    model, seed = run.rsplit("_seed", 1)
    df = pd.read_csv(path)
    df["notes"] = df["notes"].fillna("")
    # rank every test image by lens score: rank 1 = the image the model is most sure is a lens
    df["rank"] = df["score"].rank(ascending=False, method="min").astype(int)

    compound = df[df["notes"] == "dspl"].sort_values("rank")
    normal_lenses = df[(df["label"] == 1) & (df["notes"] != "dspl")]
    n_lenses = int((df["label"] == 1).sum())

    summary.append({
        "run": run, "model": model, "seed": int(seed),
        "test_auc": roc_auc_score(df["label"], df["score"]),
        # a human reading down the ranked list must reach the worst-ranked compound lens
        "inspection_cost": int(compound["rank"].max()),
        "mean_compound_rank": compound["rank"].mean(),
        # of the ordinary lenses, how many are in the top N (N = number of lenses in test)
        f"normal_lenses_in_top{n_lenses}": int((normal_lenses["rank"] <= n_lenses).sum()),
        "n_normal_lenses": len(normal_lenses),
        "n_test": len(df),
    })
    for r in compound.itertuples():
        ranks.append({"model": model, "seed": int(seed), "id": r.id, "grade": r.grade,
                      "rank": r.rank, "score": round(r.score, 4)})

summary = pd.DataFrame(summary)
ranks = pd.DataFrame(ranks)
summary.to_csv("results/summary.csv", index=False)
ranks.to_csv("results/compound_ranks.csv", index=False)

pd.set_option("display.width", 200)
print("\n=== Every run ===")
print(summary.drop(columns="run").round(4).to_string(index=False))

print("\n=== Averaged over seeds (mean ± std) ===")
for model, g in summary.groupby("model"):
    line = f"{model:8s} | {len(g)} seed(s)"
    for col in ["test_auc", "inspection_cost", "mean_compound_rank"]:
        line += f" | {col} {g[col].mean():.3f} ± {g[col].std(ddof=0):.3f}"
    print(line)

print(f"\n=== Compound lens ranks (out of {summary['n_test'].iloc[0]} test images; lower is better) ===")
print(ranks.pivot_table(index="id", columns=["model", "seed"], values="rank").to_string())

# plot: each compound lens's rank under each model (one dot per seed)
fig, ax = plt.subplots(figsize=(7, 4))
models = sorted(ranks["model"].unique())
lenses = sorted(ranks["id"].unique())
for i, model in enumerate(models):
    sub = ranks[ranks["model"] == model]
    x = [lenses.index(l) + (i - (len(models) - 1) / 2) * 0.2 for l in sub["id"]]
    ax.scatter(x, sub["rank"], label=model, s=40, alpha=0.8)
ax.set_xticks(range(len(lenses)))
ax.set_xticklabels([f"DSPL {i + 1}" for i in range(len(lenses))])
ax.set_ylabel("rank among test images (1 = top pick)")
ax.set_yscale("log")
ax.invert_yaxis()
ax.set_title("Where each model ranks the 4 real compound lenses")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/compound_ranks.png", dpi=130)
print("\nSaved results/summary.csv, results/compound_ranks.csv, results/compound_ranks.png")
