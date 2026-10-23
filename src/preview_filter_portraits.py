"""Grid of the 28 per-filter DeepDream portraits, labeled, for visual review."""
from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def main():
    portrait_dir = ROOT / "outputs" / "filter_portraits"
    small = sorted(portrait_dir.glob("small_only_*.png"))
    baseline = sorted(portrait_dir.glob("baseline_only_*.png"))

    n_cols = 7
    fig, axes = plt.subplots(4, n_cols, figsize=(2.1 * n_cols, 9.5))

    for row_offset, (group_name, files) in enumerate([("small-only", small), ("baseline-only", baseline)]):
        for i in range(14):
            r = row_offset * 2 + i // n_cols
            c = i % n_cols
            ax = axes[r, c]
            if i < len(files):
                img = Image.open(files[i])
                ax.imshow(img, cmap="gray")
                filt_id = files[i].stem.split("filter")[-1]
                ax.set_title(f"f{filt_id}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        axes[row_offset * 2, 0].set_ylabel(group_name, fontsize=11)

    fig.suptitle("stages.3 filter portraits: small-radius-only (rows 1-2) vs baseline-only (rows 3-4)")
    fig.tight_layout()
    out_path = ROOT / "outputs" / "filter_portraits_grid.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
