# plot_final_map.py: the final detection map from Step 4.
#
# Usage (from ~/ruby, with lens_env311):
#   python -I sim/plot_final_map.py
#
# Output: sim/out/final/map.png

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

frame = pd.read_csv(ROOT / "sim/out/final/map.csv")
table = lambda model: frame[frame.model == model].pivot_table(index="b", columns="d", values="auc")
frozen, trained = table("frozen"), table("zoobot3")
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
            colour = cmap((v - vmin) / (vmax - vmin))
            label = f"{v:.2f}" if not signed else "0.00" if abs(v) < 0.005 else f"{v:+.2f}"
            ax.text(col + 0.5, row + 0.5, label, ha="center", va="center", fontsize=10, color=text_colour(colour))
    ax.set_title(title, fontsize=13, color=INK, fontweight="bold", loc="left", pad=8)
    ax.set_xticks(np.arange(values.shape[1]) + 0.5, [f"{d:g}×" for d in values.columns])
    ax.set_yticks(np.arange(values.shape[0]) + 0.5, [f"{b:g}×" for b in values.index])
    ax.tick_params(length=0, labelsize=10, labelcolor=SECONDARY)
    ax.set_xlabel("Outer ring distance (× inner ring radius)", fontsize=11, color=SECONDARY)
    for spine in ax.spines.values():
        spine.set_visible(False)

fig, axes = plt.subplots(1, 3, figsize=(14, 5.6), facecolor=SURFACE)
panel(axes[0], frozen, "Frozen Zoobot", SEQUENTIAL, 0.5, 0.85)
panel(axes[1], trained, "Zoobot trained on realistic simulations", SEQUENTIAL, 0.5, 0.85)
panel(axes[2], gain, "Gain from training", DIVERGING, -0.13, 0.13, signed=True)
axes[0].set_ylabel("Outer source brightness (× noise)", fontsize=11, color=SECONDARY)
fig.suptitle("Training helps clear outer sources, but not the faintest or closest", x=0.045, y=0.97, ha="left", fontsize=17, color=INK, fontweight="bold")
fig.text(0.045, 0.025, "AUC for double vs. single lens on simulated test galaxies (0.5 = guessing). Mean of 3 seeds, 300 pairs per cell.", fontsize=10.5, color=SECONDARY)
fig.tight_layout(rect=(0.03, 0.08, 1, 0.92))
fig.savefig(ROOT / "sim/out/final/map.png", dpi=200, facecolor=SURFACE)
