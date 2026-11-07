#!/usr/bin/env python3
"""Evaluate the Zoobot lens head on held-out simulations and 79 real Euclid lenses.

Implements the project's three evaluation pieces:

1. AUROC on never-seen simulations. The head was trained on
   synthetic_lenses.csv (64x64 single-source-plane images), so the
   DSPL color simulations (100x100, two source planes, different priors)
   are genuinely held out. The non-ML baseline is the ring-contrast
   morphology metric from analyze_color_profiles.py, matched by image_id.
2. Flags at 80% compound recall. The score cutoff is the 20th percentile
   of simulated compound-lens scores (catches 80% of sim compounds); we
   then report what fraction of the 79 real Euclid lenses - and of the 4
   real compound candidates - each score flags at that cutoff.
3. Mean rank of the 4 real compound candidates within the 79-lens real
   set (rank 1 = highest lens score), a qualitative sim-to-reality check.

The real set contains only confirmed/graded lenses (no non-lenses), so no
real-data false-positive rate or real AUROC is computable; do not claim one.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def auroc(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney AUROC with average ranks for ties. No sklearn needed."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[index]]:
            end += 1
        mean_rank = (index + end) / 2 + 1
        for k in range(index, end + 1):
            ranks[order[k]] = mean_rank
        index = end + 1
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs both positive and negative examples")
    sum_pos_ranks = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-scores", type=Path, default=Path("dspl_color_experiment/zoobot_scores_r.csv"))
    parser.add_argument("--sim-metrics", type=Path, default=Path("dspl_color_experiment/analysis/metrics_per_image.csv"))
    parser.add_argument("--real-scores", type=Path, default=Path("test_lenses_79/zoobot_scores.csv"))
    parser.add_argument("--output", type=Path, default=Path("test_lenses_79/real_lens_evaluation.csv"))
    args = parser.parse_args()

    for path in (args.sim_scores, args.sim_metrics, args.real_scores):
        if not path.is_file():
            parser.error(f"Missing {path}")

    sim = load_csv(args.sim_scores)
    real = load_csv(args.real_scores)
    metrics_by_id = {row["image_id"]: row for row in load_csv(args.sim_metrics)}

    sim_scores = [float(row["lens_score"]) for row in sim]
    sim_labels = [int(row["label"]) for row in sim]
    matched = [metrics_by_id[row["image_id"]] for row in sim if row["image_id"] in metrics_by_id]
    if len(matched) != len(sim):
        raise ValueError("Sim scores and morphology metrics do not align by image_id")
    sim_contrast = [float(row["ring_contrast"]) for row in matched]

    # 1. Held-out-simulation AUROC: Zoobot head vs ring-contrast baseline.
    sim_compound = [(float(r["lens_score"]), int(r["label"])) for r in sim if r["lens_structure"] == "compound"]
    sim_simple = [(float(r["lens_score"]), int(r["label"])) for r in sim if r["lens_structure"] == "simple"]
    zoobot_auroc_all = auroc(sim_scores, sim_labels)
    contrast_auroc_all = auroc(sim_contrast, sim_labels)
    zoobot_auroc_compound = auroc([s for s, _ in sim_compound], [l for _, l in sim_compound])
    contrast_auroc_compound = auroc(
        [float(metrics_by_id[r["image_id"]]["ring_contrast"]) for r in sim if r["lens_structure"] == "compound"],
        [l for _, l in sim_compound],
    )
    zoobot_auroc_simple = auroc([s for s, _ in sim_simple], [l for _, l in sim_simple])

    # 2. Cutoff = 80% recall on simulated compounds; apply to the real set.
    compound_pos_scores = sorted(s for s, l in sim_compound if l == 1)
    cutoff_index = max(0, int(0.2 * len(compound_pos_scores)) - 1)
    cutoff = compound_pos_scores[cutoff_index]
    sim_compound_recall = sum(s >= cutoff for s in compound_pos_scores) / len(compound_pos_scores)
    sim_neg_flagged = sum(
        float(r["lens_score"]) >= cutoff for r in sim if r["label"] == "0"
    ) / sum(1 for r in sim if r["label"] == "0")

    real_scores = [float(row["lens_score"]) for row in real]
    real_compound = [row for row in real if row["compound"] == "True"]
    real_flagged = sum(s >= cutoff for s in real_scores) / len(real_scores)
    real_compound_flagged = (
        sum(float(row["lens_score"]) >= cutoff for row in real_compound) / len(real_compound)
    )
    real_ordinary_flagged = (
        sum(float(row["lens_score"]) >= cutoff for row in real if row["compound"] != "True")
        / (len(real) - len(real_compound))
    )

    # 3. Rank of the 4 real compounds within the 79 (rank 1 = highest score).
    order = sorted(range(len(real_scores)), key=lambda i: real_scores[i], reverse=True)
    rank_of = {order[position]: position + 1 for position in range(len(order))}
    compound_ranks = sorted(rank_of[i] for i, row in enumerate(real) if row["compound"] == "True")

    results = [
        ("heldout_sim_auroc_zoobot_all", zoobot_auroc_all, f"n={len(sim)} sims (900 lens / 60 non-lens)"),
        ("heldout_sim_auroc_ring_contrast_baseline_all", contrast_auroc_all, "non-ML morphology baseline"),
        ("heldout_sim_auroc_zoobot_compound_only", zoobot_auroc_compound, "compounds vs non-lens"),
        ("heldout_sim_auroc_ring_contrast_compound_only", contrast_auroc_compound, "compounds vs non-lens"),
        ("heldout_sim_auroc_zoobot_simple_only", zoobot_auroc_simple, "simples vs non-lens"),
        ("cutoff_at_80pct_sim_compound_recall", cutoff, f"achieved sim-compound recall {sim_compound_recall:.3f}"),
        ("sim_nonlens_flag_rate_at_cutoff", sim_neg_flagged, "fraction of 60 sim non-lenses above cutoff"),
        ("real_flag_fraction_at_cutoff", real_flagged, f"fraction of {len(real)} real lenses above cutoff"),
        ("real_compound_flag_fraction_at_cutoff", real_compound_flagged, f"n={len(real_compound)} real compounds"),
        ("real_ordinary_flag_fraction_at_cutoff", real_ordinary_flagged, "n=75 ordinary real lenses"),
        ("real_compound_mean_rank", statistics.mean(compound_ranks), f"ranks {compound_ranks} of {len(real)}"),
        ("real_compound_median_score", statistics.median(float(r['lens_score']) for r in real_compound), ""),
        ("real_ordinary_median_score", statistics.median(float(r['lens_score']) for r in real if r['compound'] != 'True'), ""),
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value", "note"])
        writer.writerows(results)

    print(f"Cutoff for 80% sim-compound recall: {cutoff:.4f}")
    print(f"{'metric':<50} {'value':>10}  note")
    for name, value, note in results:
        print(f"{name:<50} {value:>10.4f}  {note}")
    print()
    print("Per-compound detail (rank of 79, score):")
    for i, row in enumerate(real):
        if row["compound"] == "True":
            print(
                f"  {row['name'] or row['image_id']}: rank {rank_of[i]}/79, "
                f"score {float(row['lens_score']):.3f}, theta_E {row['einstein_radius_arcsec']}\""
            )
    print()
    print(f"Wrote {args.output}")
    print("Note: the real set has no non-lenses, so no real-data FPR or AUROC is computable.")


if __name__ == "__main__":
    main()
