"""Grid of each target filter's top-5 real-image activators, for visual review
and naming. One grid for small-radius-only filters, one for baseline-only."""
from __future__ import annotations

import json
import matplotlib.pyplot as plt

from top_activating_images import load_pool, ROOT


def build_grid(filter_ids, images, labels, sources, results, out_name, title):
    n_rows, n_cols = len(filter_ids), 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.1 * n_rows))

    for r, filt in enumerate(filter_ids):
        entries = results[str(filt)]
        for c in range(n_cols):
            ax = axes[r, c]
            if c < len(entries):
                e = entries[c]
                idx = e["index"]
                ax.imshow(images[idx], cmap="gray", origin="lower", vmin=0, vmax=1)
                tag = "LENS" if e["label"] == 1 else "non"
                ax.set_title(f"{e['source']}/{tag}\nact={e['activation']:.2f}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        axes[r, 0].set_ylabel(f"f{filt}", fontsize=11, rotation=0, labelpad=25, va="center")

    fig.suptitle(title)
    fig.tight_layout()
    out_path = ROOT / "outputs" / out_name
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    report = json.loads((ROOT / "outputs" / "filter_firing_report.json").read_text())
    results = json.loads((ROOT / "outputs" / "top_activating_images.json").read_text())
    small_only = report["stages.3"]["small_radius_only"]
    baseline_only = report["stages.3"]["baseline_only"]

    images, labels, sources = load_pool()

    build_grid(small_only, images, labels, sources, results,
               "top_activating_small_only.png",
               "Top-5 activating real images per filter -- small-radius-only filters")
    build_grid(baseline_only, images, labels, sources, results,
               "top_activating_baseline_only.png",
               "Top-5 activating real images per filter -- baseline-only filters")


if __name__ == "__main__":
    main()
