#!/usr/bin/env python3
"""Summarize Zoobot DSPL scores and plot one-factor experiment results."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Score CSV contains no rows: {path}")
    required = {"experiment", "condition_value", "label", "lens_score", "predicted_label"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Score CSV is missing columns: {', '.join(sorted(missing))}")
    return rows


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["experiment"], row["condition_value"], row["label"])].append(row)

    summaries: list[dict[str, object]] = []
    control_stats: dict[str, tuple[float, float]] = {}
    for (experiment, condition_value, label), group in groups.items():
        scores = [float(row["lens_score"]) for row in group]
        predictions = [int(row["predicted_label"]) for row in group]
        mean_score = statistics.mean(scores)
        detection_rate = sum(prediction == 1 for prediction in predictions) / len(group)
        if label == "1" and float(condition_value) == 0.0 and experiment == "faintness":
            control_stats[experiment] = (mean_score, detection_rate)
        elif label == "1" and float(condition_value) == 1.0 and experiment in {"arc_size", "blending"}:
            control_stats[experiment] = (mean_score, detection_rate)

        summaries.append(
            {
                "experiment": experiment,
                "condition_value": condition_value,
                "condition_unit": group[0].get("condition_unit", ""),
                "label": label,
                "n_tested": len(group),
                "mean_lens_score": mean_score,
                "median_lens_score": statistics.median(scores),
                "std_lens_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "detection_rate": detection_rate if label == "1" else "",
                "false_positive_rate": detection_rate if label == "0" else "",
            }
        )

    for summary in summaries:
        experiment = str(summary["experiment"])
        baseline = control_stats.get(experiment)
        if baseline is None or summary["label"] != "1":
            summary["mean_score_change_vs_control"] = ""
            summary["detection_rate_change_vs_control"] = ""
        else:
            summary["mean_score_change_vs_control"] = float(summary["mean_lens_score"]) - baseline[0]
            summary["detection_rate_change_vs_control"] = float(summary["detection_rate"]) - baseline[1]
    return summaries


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.96
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def summarize_lens_categories(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    categories = [
        ("Source brightness", "Bright/moderate", "Magnitude offset 0-1", "faintness", {0.0, 1.0}),
        ("Source brightness", "Faint", "Magnitude offset 2-3", "faintness", {2.0, 3.0}),
        ("Arc size", "Small arcs", "Einstein-radius scale 0.70-0.85", "arc_size", {0.7, 0.85}),
        ("Arc size", "Baseline arcs", "Einstein-radius scale 1.00", "arc_size", {1.0}),
        ("Arc size", "Large arcs", "Einstein-radius scale 1.15-1.30", "arc_size", {1.15, 1.3}),
        ("Foreground blending", "Low lens light", "Foreground-light flux scale 0.5", "blending", {0.5}),
        ("Foreground blending", "Nominal lens light", "Foreground-light flux scale 1.0", "blending", {1.0}),
        ("Foreground blending", "High lens light", "Foreground-light flux scale 2-4", "blending", {2.0, 4.0}),
    ]

    category_rows: list[dict[str, object]] = []
    category_lookup: dict[tuple[str, str], dict[str, object]] = {}
    for feature, category, definition, experiment, values in categories:
        group = [
            row
            for row in rows
            if row["label"] == "1"
            and row["experiment"] == experiment
            and round(float(row["condition_value"]), 6) in values
        ]
        if not group:
            continue
        predictions = [int(row["predicted_label"]) for row in group]
        detected = sum(prediction == 1 for prediction in predictions)
        rate = detected / len(group)
        lower, upper = wilson_interval(detected, len(group))
        result: dict[str, object] = {
            "lens_feature": feature,
            "category": category,
            "simulation_definition": definition,
            "label": 1,
            "rate_type": "positive_detection_rate",
            "n_tested": len(group),
            "classified_as_lens": detected,
            "rate": rate,
            "rate_ci_95_low": lower,
            "rate_ci_95_high": upper,
            "mean_lens_score": statistics.mean(float(row["lens_score"]) for row in group),
        }
        category_rows.append(result)
        category_lookup[(feature, category)] = result

    control_rows = [row for row in rows if row["label"] == "0"]
    if control_rows:
        false_positives = sum(int(row["predicted_label"]) == 1 for row in control_rows)
        rate = false_positives / len(control_rows)
        lower, upper = wilson_interval(false_positives, len(control_rows))
        result = {
            "lens_feature": "Control behavior",
            "category": "Negative controls",
            "simulation_definition": "Unlensed/weakly lensed controls; label=0",
            "label": 0,
            "rate_type": "false_positive_rate",
            "n_tested": len(control_rows),
            "classified_as_lens": false_positives,
            "rate": rate,
            "rate_ci_95_low": lower,
            "rate_ci_95_high": upper,
            "mean_lens_score": statistics.mean(float(row["lens_score"]) for row in control_rows),
        }
        category_rows.append(result)

    reference_categories = {
        ("Source brightness", "Faint"): ("Source brightness", "Bright/moderate"),
        ("Arc size", "Small arcs"): ("Arc size", "Baseline arcs"),
        ("Arc size", "Large arcs"): ("Arc size", "Baseline arcs"),
        ("Foreground blending", "Low lens light"): ("Foreground blending", "Nominal lens light"),
        ("Foreground blending", "High lens light"): ("Foreground blending", "Nominal lens light"),
    }
    for result in category_rows:
        feature = str(result["lens_feature"])
        category = str(result["category"])
        if int(result["label"]) == 0:
            false_positives = int(result["classified_as_lens"])
            if false_positives == 0:
                result["project_interpretation"] = (
                    "No controls were flagged in this pilot; the small control sample cannot "
                    "establish a real-survey false-positive rate."
                )
            else:
                result["project_interpretation"] = (
                    f"{false_positives} control(s) were flagged; add diverse negative examples "
                    "before estimating or tuning for real-survey contamination."
                )
            continue

        reference_key = reference_categories.get((feature, category))
        if reference_key and reference_key in category_lookup:
            reference = category_lookup[reference_key]
            score_change = float(result["mean_lens_score"]) - float(reference["mean_lens_score"])
            rate_change = float(result["rate"]) - float(reference["rate"])
            comparison = (
                f"Versus {reference_key[1].lower()}, mean score changed by {score_change:+.3f} "
                f"and detection rate by {rate_change:+.0%}."
            )
        else:
            comparison = "This is the reference bin for its one-factor sweep."

        if feature == "Arc size" and float(result["rate"]) < 1:
            priority = "Prioritize small-arc examples for larger follow-up tests and training coverage."
        elif feature == "Source brightness" and category == "Faint":
            priority = "Retain faint sources in training and verify this trend with more examples."
        elif feature == "Foreground blending" and category == "High lens light":
            priority = "Keep strongly blended examples in evaluation; scores may soften as foreground light rises."
        elif float(result["rate"]) == 1:
            priority = "No misses occurred in this synthetic bin; confirm with a larger independent sample."
        else:
            priority = "Expand this bin before making training or deployment decisions."
        result["project_interpretation"] = f"{comparison} {priority}"

    return category_rows


def make_plot(summaries: list[dict[str, object]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    experiments = ["faintness", "arc_size", "blending"]
    titles = ["Source faintness", "Einstein-radius scale", "Foreground-light blending"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, experiment, title in zip(axes, experiments, titles):
        conditions = [
            row
            for row in summaries
            if row["experiment"] == experiment and row["label"] == "1"
        ]
        conditions.sort(key=lambda row: float(row["condition_value"]))
        x_values = [float(row["condition_value"]) for row in conditions]
        means = [float(row["mean_lens_score"]) for row in conditions]
        deviations = [float(row["std_lens_score"]) for row in conditions]
        rates = [float(row["detection_rate"]) for row in conditions]
        axes_score = axis.errorbar(
            x_values,
            means,
            yerr=deviations,
            marker="o",
            capsize=3,
            label="Mean lens score ± SD",
            color="#007C91",
        )
        axes_rate = axis.plot(
            x_values,
            rates,
            marker="s",
            linestyle="--",
            label="Detection rate (argmax)",
            color="#D55E00",
        )
        axis.set_title(title)
        axis.set_xlabel(conditions[0]["condition_unit"] if conditions else "Condition")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Zoobot class-1 probability / detection rate")
    figure.suptitle("DSPL-like image sensitivity by one-factor condition")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=Path("dspl_experiment/zoobot_scores.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("dspl_experiment"))
    args = parser.parse_args()
    if not args.scores.is_file():
        parser.error(
            f"Zoobot score file not found: {args.scores}. Run run_zoobot_inference.py "
            "with a trained classifier checkpoint first; no metrics will be fabricated."
        )

    rows = load_rows(args.scores)
    summaries = summarize(rows)
    category_summaries = summarize_lens_categories(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary_by_condition.csv"
    fields = list(summaries[0])
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    category_path = args.output_dir / "detection_by_lens_category.csv"
    if category_summaries:
        with category_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(category_summaries[0]))
            writer.writeheader()
            writer.writerows(category_summaries)
    plot_path = args.output_dir / "performance_by_parameter.png"
    make_plot(summaries, plot_path)
    print(f"Wrote summary table: {summary_path}")
    if category_summaries:
        print(f"Wrote lens-category interpretation table: {category_path}")
    print(f"Wrote plot: {plot_path}")
    print("Positive detection rate is reported for label=1; label=0 false-positive rate is separate.")
    print("Category rates include Wilson 95% intervals; they are not overall accuracy.")
    print("No overall accuracy or unconfigured probability threshold is reported.")


if __name__ == "__main__":
    main()