#!/usr/bin/env python3
"""Summarize Zoobot scores on the multi-band DSPL color experiment.

Reads dspl_color_experiment/zoobot_scores_{g,r,i}.csv (produced by
run_zoobot_inference.py with --image-column image_path_{g,r,i}) and writes:

- accuracy_by_band.csv: overall accuracy, lens detection rate, and
  non-lens rejection rate per band, plus a band-pooled view.
- detection_by_condition.csv: detection rate per experiment/condition,
  split by lens structure (simple vs compound) and band.
- detection_by_lens_category.csv: the same binned category table format as
  the grayscale DSPL pilot, per band.

Class decisions use Zoobot's softmax argmax, matching the DSPL pilot.
All numbers describe one synthetic dataset scored by a head trained on
synthetic data - not real-survey accuracy.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

BANDS = ("g", "r", "i")

CATEGORY_BINS = [
    ("Source brightness", "Bright/moderate", "Magnitude offset 0-1", "faintness", {0.0, 1.0}),
    ("Source brightness", "Faint", "Magnitude offset 2-3", "faintness", {2.0, 3.0}),
    ("Arc size", "Very small arcs", "Einstein-radius scale 0.50-0.55", "arc_size", {0.5, 0.55}),
    ("Arc size", "Small arcs", "Einstein-radius scale 0.70-0.85", "arc_size", {0.7, 0.85}),
    ("Arc size", "Baseline arcs", "Einstein-radius scale 1.00", "arc_size", {1.0}),
    ("Arc size", "Large arcs", "Einstein-radius scale 1.15-1.30", "arc_size", {1.15, 1.3}),
    ("Foreground blending", "Low lens light", "Foreground-light flux scale 0.5", "blending", {0.5}),
    ("Foreground blending", "Nominal lens light", "Foreground-light flux scale 1.0", "blending", {1.0}),
    ("Foreground blending", "High lens light", "Foreground-light flux scale 2-4", "blending", {2.0, 4.0}),
]


def load_scores(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Score CSV contains no rows: {path}")
    required = {"label", "lens_score", "predicted_label", "experiment", "condition_value", "lens_structure"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return rows


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.96
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def band_accuracy(rows: list[dict[str, str]], band: str) -> dict[str, object]:
    positives = [row for row in rows if row["label"] == "1"]
    negatives = [row for row in rows if row["label"] == "0"]
    detected = sum(int(row["predicted_label"]) == 1 for row in positives)
    false_positives = sum(int(row["predicted_label"]) == 1 for row in negatives)
    correct = detected + (len(negatives) - false_positives)
    det_low, det_high = wilson_interval(detected, len(positives))
    fp_low, fp_high = wilson_interval(false_positives, len(negatives))
    return {
        "band": band,
        "n_images": len(rows),
        "n_lenses": len(positives),
        "n_non_lenses": len(negatives),
        "overall_accuracy": correct / len(rows),
        "lens_detection_rate": detected / len(positives),
        "lens_detection_ci95": f"[{det_low:.3f}, {det_high:.3f}]",
        "false_positive_rate": false_positives / len(negatives),
        "false_positive_ci95": f"[{fp_low:.3f}, {fp_high:.3f}]",
        "mean_lens_score_lenses": statistics.mean(float(row["lens_score"]) for row in positives),
        "mean_lens_score_non_lenses": statistics.mean(float(row["lens_score"]) for row in negatives),
    }


def condition_rows(rows: list[dict[str, str]], band: str) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["experiment"], row["condition_value"], row["lens_structure"])].append(row)

    output: list[dict[str, object]] = []
    for (experiment, condition_value, structure), group in sorted(
        groups.items(), key=lambda item: (item[0][0], float(item[0][1]), item[0][2])
    ):
        positives = [row for row in group if row["label"] == "1"]
        detected = sum(int(row["predicted_label"]) == 1 for row in positives)
        entry: dict[str, object] = {
            "band": band,
            "experiment": experiment,
            "condition_value": condition_value,
            "lens_structure": structure,
            "n_lenses": len(positives),
            "detected": detected,
            "detection_rate": detected / len(positives) if positives else "",
            "mean_lens_score": (
                statistics.mean(float(row["lens_score"]) for row in positives) if positives else ""
            ),
        }
        output.append(entry)
    return output


def category_table(rows: list[dict[str, str]], band: str) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for feature, category, definition, experiment, values in CATEGORY_BINS:
        for structure in ("simple", "compound", "both"):
            group = [
                row
                for row in rows
                if row["label"] == "1"
                and row["experiment"] == experiment
                and round(float(row["condition_value"]), 6) in values
                and (structure == "both" or row["lens_structure"] == structure)
            ]
            if not group:
                continue
            detected = sum(int(row["predicted_label"]) == 1 for row in group)
            low, high = wilson_interval(detected, len(group))
            output.append(
                {
                    "band": band,
                    "lens_feature": feature,
                    "category": category,
                    "simulation_definition": definition,
                    "lens_structure": structure,
                    "n_tested": len(group),
                    "classified_as_lens": detected,
                    "detection_rate": detected / len(group),
                    "rate_ci_95_low": low,
                    "rate_ci_95_high": high,
                    "mean_lens_score": statistics.mean(float(row["lens_score"]) for row in group),
                }
            )
    negatives = [row for row in rows if row["label"] == "0"]
    false_positives = sum(int(row["predicted_label"]) == 1 for row in negatives)
    low, high = wilson_interval(false_positives, len(negatives))
    output.append(
        {
            "band": band,
            "lens_feature": "Control behavior",
            "category": "Negative controls",
            "simulation_definition": "Unlensed/weakly lensed controls; label=0",
            "lens_structure": "both",
            "n_tested": len(negatives),
            "classified_as_lens": false_positives,
            "detection_rate": false_positives / len(negatives),
            "rate_ci_95_low": low,
            "rate_ci_95_high": high,
            "mean_lens_score": statistics.mean(float(row["lens_score"]) for row in negatives),
        }
    )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", type=Path, default=Path("dspl_color_experiment"))
    parser.add_argument("--output-dir", type=Path, default=Path("dspl_color_experiment/zoobot_analysis"))
    args = parser.parse_args()

    accuracy_rows: list[dict[str, object]] = []
    all_condition_rows: list[dict[str, object]] = []
    all_category_rows: list[dict[str, object]] = []
    pooled: list[dict[str, str]] = []

    for band in BANDS:
        path = args.score_dir / f"zoobot_scores_{band}.csv"
        if not path.is_file():
            parser.error(
                f"Missing {path}. Run run_zoobot_inference.py with "
                f"--image-column image_path_{band} first."
            )
        rows = load_scores(path)
        accuracy_rows.append(band_accuracy(rows, band))
        all_condition_rows.extend(condition_rows(rows, band))
        all_category_rows.extend(category_table(rows, band))
        pooled.extend(rows)

    accuracy_rows.append(band_accuracy(pooled, "pooled g+r+i (images double-counted per band)"))
    write_csv(args.output_dir / "accuracy_by_band.csv", accuracy_rows)
    write_csv(args.output_dir / "detection_by_condition.csv", all_condition_rows)
    write_csv(args.output_dir / "detection_by_lens_category.csv", all_category_rows)

    print(f"Wrote {args.output_dir / 'accuracy_by_band.csv'}")
    print(f"Wrote {args.output_dir / 'detection_by_condition.csv'}")
    print(f"Wrote {args.output_dir / 'detection_by_lens_category.csv'}")
    print()
    print("Accuracy by band:")
    for row in accuracy_rows:
        print(
            f"  {row['band']}: accuracy={row['overall_accuracy']:.3f} "
            f"detection={row['lens_detection_rate']:.3f} fp={row['false_positive_rate']:.3f} "
            f"(n={row['n_images']})"
        )
    print()
    print("Synthetic-data demo only: the head was trained on this project's simulations,")
    print("so these numbers are not real-survey accuracy.")


if __name__ == "__main__":
    main()
