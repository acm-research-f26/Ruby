# plot_real_map.py: the detection map on real lenses, from sim/real_lens_map.py.
#
# Usage (from ~/ruby, with lens_env311):
#   python -I sim/plot_real_map.py
#
# Output: sim/out/final/real_map/real_map.png

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb

ROOT = Path.home() / "ruby"
SURFACE, INK, SECONDARY = "#fcfcfb", "#0b0b0b", "#52514e"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("gain", ["#e34948", "#f0efec", "#2a78d6"])
plt.rcParams["font.family"] = ["Arial", "Helvetica", "DejaVu Sans"]

summary = pd.read_csv(ROOT / "sim/out/final/real_map/summary.csv")
table = lambda model, column: summary[summary.model == model].pivot(index="b", columns="d", values=column)
frozen, trained, own = table("frozen", "auc"), table("zoobot3", "auc"), table("zoobot3", "beats_own")
gain = trained - frozen

def text_colour(colour):
    r, g, b = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(colour)]
    return "#ffffff" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.25 else INK

def panel(ax, values, title, cmap, vmin, vmax, signed=False):
    ax.pcolormesh(values.values, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors=SURFACE, linewidth=2)
    ax.set_aspect("equal")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            v = values.values[row, col]
            label = f"{v:.2f}" if not signed else "0.00" if abs(v) < 0.005 else f"{v:+.2f}"
            ax.text(col + 0.5, row + 0.5, label, ha="center", va="center", fontsize=9.5, color=text_colour(cmap((v - vmin) / (vmax - vmin))))
    ax.add_patch(plt.Rectangle((0, 0), 6, 7, fill=False, ec=INK, lw=1.6, zorder=3))
    ax.set_title(title, fontsize=13, color=INK, fontweight="bold", loc="left", pad=8)
    ax.set_xticks(np.arange(values.shape[1]) + 0.5, [f"{d:g}×" for d in values.columns])
    ax.set_yticks(np.arange(values.shape[0]) + 0.5, [f"{b:g}×" for b in values.index])
    ax.tick_params(length=0, labelsize=10, labelcolor=SECONDARY)
    ax.set_xlabel("Outer ring distance (× inner ring radius)", fontsize=10.5, color=SECONDARY)
    ax.set_ylabel("Outer source brightness (× noise)", fontsize=10.5, color=SECONDARY)
    for spine in ax.spines.values():
        spine.set_visible(False)

fig, axes = plt.subplots(2, 2, figsize=(13, 13.2), facecolor=SURFACE)
panel(axes[0, 0], frozen, "Frozen Zoobot: AUC", SEQUENTIAL, 0.5, 0.85)
panel(axes[0, 1], trained, "Zoobot trained on realistic simulations: AUC", SEQUENTIAL, 0.5, 0.85)
panel(axes[1, 0], gain, "Gain from training: AUC", DIVERGING, -0.13, 0.13, signed=True)
panel(axes[1, 1], own, "Trained: share of sources that raise a lens's own score", SEQUENTIAL, 0.45, 1.0)
fig.suptitle("On real lenses, training helps most at 2–6× the noise", x=0.05, y=0.985, ha="left", fontsize=18, color=INK, fontweight="bold")
fig.text(0.05, 0.018, "Real lenses with a simulated outer source vs. the same lenses without one. 84 ordinary lenses, 3 sources each, mean of 3 seeds. "
         "Each AUC is good to about ±0.035.\nThe black box marks the training range. After-the-fact check.", fontsize=10, color=SECONDARY)
fig.tight_layout(rect=(0.02, 0.05, 1, 0.965))
fig.savefig(ROOT / "sim/out/final/real_map/real_map.png", dpi=170, facecolor=SURFACE)

inside = summary[summary.d <= 1.8]
print(inside[inside.model == "zoobot3"].groupby("b")[["auc", "beats_own"]].mean().round(3).join(inside[inside.model == "frozen"].groupby("b").auc.mean().round(3).rename("frozen")).iloc[::-1].to_string())
