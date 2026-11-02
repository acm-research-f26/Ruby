# place_real_lenses.py: where do the 4 real compound lenses sit on the detection map?
# No model is run, so this uses up nothing from the test set.
#
# Usage (from ~/ruby, with lens_env311):
#   python -I sim/place_real_lenses.py
#
# Distance: the simulator's d is the ratio of the two Einstein radii, so the published ratio (Li et al. 2026, Table 2) goes
# on the map's axis. In the images the outer ring sits farther out, because the inner source's own mass bends the outer
# light as well.
# Brightness: Table 2 gives each source's lensed IE magnitude, so the outer source's total light is known without having
# to separate it from the rest of the image. That light, in units of the image's noise, is compared with simulated outer
# sources at a similar radius, whose total light and b are both known. (Measuring the ring directly in the image failed:
# the reading had a correlation of 0.03 with true b on simulated pairs.)
# The conversion from magnitude to pixel units assumes a zero point of 23.9 (microjansky). The lens galaxies check it.
#
# Output (in sim/out/final/placement/): placement.txt, placement.csv, map_with_lenses.png

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate as sim
from simulate import ROOT

out_dir = ROOT / "sim/out/final/placement"
out_dir.mkdir(parents=True, exist_ok=True)
ZERO_POINT = 23.9

# Table 2: magnitudes of the lens, source 1 and source 2, then theta_E1, theta_sis1, theta_E2 (arcsec)
TABLE = {"469177": ("Cosmic Dartboard", 21.2, 22.9, 23.2, 0.45, 0.48, 1.31),
         "107505": ("Galileo's Lens", 22.3, 22.7, 24.4, 1.09, 0.28, 1.52),
         "726389": ("Cosmic Ammonite", 20.0, 22.3, 24.2, 1.18, 0.15, 1.35),
         "378903": ("Teapot Lens", 20.0, 23.3, 24.0, 0.62, 0.13, 0.89)}

report = []
def say(line=""):
    print(line, flush=True)
    report.append(line)

flux = lambda magnitude: 10 ** ((ZERO_POINT - magnitude) / 2.5)
yy, xx = np.mgrid[:100, :100]
R = np.hypot(yy - 49.5, xx - 49.5) * 0.1

data = np.load(ROOT / "sim/out/v2/val.npz")
params = pd.read_csv(ROOT / "sim/out/v2/val_params.csv", dtype={"host": str})
noise = np.array([sim.noise_level(np.nan_to_num(x))[1] for x in data["single"]])
params["total"] = (data["double"] - data["single"]).sum(axis=(1, 2)) / noise      # outer source's total light, in noise units
params["per_b"] = params.total / params.b
params["radius"] = params.d * params.theta_e + params.inter
say(f"Simulated outer sources: total light per unit of b, all validation pairs, 16/50/84%: {params.per_b.quantile([.16, .5, .84]).round(1).tolist()}")
say(f"Simulated total light in noise units, 5/50/95%: {params.total.quantile([.05, .5, .95]).round(0).tolist()}")

split = pd.read_csv(ROOT / "data/split.csv", dtype={"id": str})
lenses = split[split.notes == "dspl"].copy()
lenses["key"] = lenses.id.str[-6:]
ranks = pd.read_csv(ROOT / "sim/out/final/ranks.csv", dtype={"id": str})
ranks = ranks[(ranks.model == "zoobot3") & ranks.compound]
grid = pd.read_csv(ROOT / "sim/out/final/map.csv")
table = lambda model: grid[grid.model == model].pivot_table(index="b", columns="d", values="auc")
trained, frozen = table("zoobot3"), table("frozen")

def on_map(values, d, b):
    """Nearest cell, with brightness and distance kept inside the map's range."""
    row = values.index[np.abs(np.log(values.index) - np.log(np.clip(b, 0.5, 6))).argmin()]
    column = values.columns[np.abs(values.columns - np.clip(d, 1.1, 1.8)).argmin()]
    return row, column, values.loc[row, column]

say("\nChecking the zero point: lens galaxy, published flux vs. light within 3\" of the centre (lens and arcs together):")
rows = []
for key, (name, m_lens, m_s1, m_s2, t1, sis, t2) in TABLE.items():
    row = lenses[lenses.key == key].iloc[0]
    x = np.nan_to_num(np.load(ROOT / row.path).astype(np.float32))
    background, sigma = sim.noise_level(x)
    measured = (x - background)[R < 3].sum()
    published = flux(m_lens) + flux(m_s1) + flux(m_s2)
    say(f"  {name:17s} published lens + sources {published:6.1f}, measured {measured:6.1f}, ratio {measured / published:.2f}")
    total = flux(m_s2) / sigma
    radius = t2 + sis
    near = params[(params.radius - radius).abs() < 0.3]
    per_b = near.per_b.quantile([.16, .5, .84]).values
    b_mid, b_low, b_high = total / per_b[1], total / per_b[2], total / per_b[0]
    rows.append(dict(lens=name, d=t2 / t1, radius=radius, total=total, b=b_mid, b_low=b_low, b_high=b_high, s2_to_s1=flux(m_s2) / flux(m_s1), noise=sigma,
                     pairs=len(near), share_of_sim_below=(params.total < total).mean(),
                     cell_b=on_map(trained, t2 / t1, b_mid)[0], cell_d=on_map(trained, t2 / t1, b_mid)[1], trained_auc=on_map(trained, t2 / t1, b_mid)[2],
                     frozen_auc=on_map(frozen, t2 / t1, b_mid)[2], score=ranks[ranks.id == row.id].score.mean(), rank=ranks[ranks.id == row.id]["rank"].mean()))
frame = pd.DataFrame(rows)
frame.to_csv(out_dir / "placement.csv", index=False)
say("\nThe 4 real compound lenses on the map:")
for _, r in frame.iterrows():
    say(f"  {r.lens:17s} distance {r.d:.2f}x; outer source total {r.total:.0f} noise units, more than {r.share_of_sim_below:.0%} of simulated ones; brightness about {r.b:.1f}x noise ({r.b_low:.1f}-{r.b_high:.1f}; {r.pairs} pairs); "
        f"outer/inner light {r.s2_to_s1:.2f}; nearest cell b={r.cell_b:g}, d={r.cell_d:g}: trained AUC {r.trained_auc:.2f}, frozen {r.frozen_auc:.2f}; real score {r.score:.2f}, mean rank {r['rank']:.1f}")
(out_dir / "placement.txt").write_text("\n".join(report) + "\n")

# ---- The map, with the 4 lenses on it -------------------------------------------------------
SURFACE, INK, SECONDARY, BAND, MARK = "#fcfcfb", "#0b0b0b", "#52514e", "#f0efec", "#eb6834"
plt.rcParams["font.family"] = ["Arial", "Helvetica", "DejaVu Sans"]
ramp = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
distances, levels = trained.columns.values, trained.index.values
to_x = lambda d: np.interp(d, distances, np.arange(len(distances)) + 0.5) if d <= distances[-1] else len(distances) + 0.5
to_y = lambda b: np.interp(np.log(b), np.log(levels), np.arange(len(levels)) + 0.5) if b <= levels[-1] else min(len(levels) - 0.5 + np.log(b / levels[-1]) / np.log(levels[-1] / levels[-2]), len(levels) + 0.95)

fig, ax = plt.subplots(figsize=(8.6, 7.4), facecolor=SURFACE)
ax.add_patch(plt.Rectangle((0, len(levels)), len(distances) + 1, 1, color=BAND, lw=0))
ax.add_patch(plt.Rectangle((len(distances), 0), 1, len(levels) + 1, color=BAND, lw=0))
ax.pcolormesh(trained.values, cmap=ramp, vmin=0.5, vmax=0.85, edgecolors=SURFACE, linewidth=2)
for row in range(len(levels)):
    for col in range(len(distances)):
        v = trained.values[row, col]
        ax.text(col + 0.1, row + 0.8, f"{v:.2f}", ha="left", va="center", fontsize=8.5, color="#ffffff" if v > 0.66 else INK)
ax.text(len(distances) + 0.5, 3.5, "beyond the map", rotation=90, ha="center", va="center", fontsize=10, color=SECONDARY)
ax.text(0.1, len(levels) + 0.82, "beyond the map", ha="left", va="center", fontsize=10, color=SECONDARY)
places = {"Cosmic Dartboard": (6.15, 7.72, "right"), "Galileo's Lens": (2.6, 7.3, "center"), "Cosmic Ammonite": (0.95, 7.3, "center"), "Teapot Lens": (4.3, 7.3, "center")}
for _, r in frame.iterrows():
    x, y = to_x(r.d), to_y(r.b)
    ax.plot([x, x], [to_y(max(r.b_low, 0.5)), to_y(r.b_high)], color=MARK, lw=2, zorder=3, solid_capstyle="round", path_effects=[pe.Stroke(linewidth=4, foreground=SURFACE), pe.Normal()])
    ax.plot(x, y, "o", ms=11, color=MARK, mec=SURFACE, mew=2, zorder=4)
    tx, ty, align = places[r.lens]
    ax.annotate(f"{r.lens}\n{r.d:.2f}×, about {r.b:.1f}×", (x, y), xytext=(tx, ty), textcoords="data", ha=align, va="center", fontsize=9, color=INK, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8, shrinkB=6), zorder=5)
ax.set_xlim(0, len(distances) + 1)
ax.set_ylim(0, len(levels) + 1)
ax.set_aspect("equal")
ax.set_xticks(np.arange(len(distances)) + 0.5, [f"{d:g}×" for d in distances])
ax.set_yticks(np.arange(len(levels)) + 0.5, [f"{b:g}×" for b in levels])
ax.tick_params(length=0, labelsize=10, labelcolor=SECONDARY)
ax.set_xlabel("Outer ring distance (ratio of the two Einstein radii)", fontsize=11, color=SECONDARY)
ax.set_ylabel("Outer source brightness (× noise)", fontsize=11, color=SECONDARY)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.suptitle("None of the 4 real compound lenses sits in the faint zone", x=0.06, y=0.97, ha="left", fontsize=15, color=INK, fontweight="bold")
fig.text(0.06, 0.025, "Cells: AUC for double vs. single on simulated test galaxies, trained Zoobot. Bars: range from simulated sources with similar total light.\n"
         "Brightness comes from published magnitudes and may be 2–3× higher, because the image holds more light than they imply.", fontsize=8.5, color=SECONDARY)
fig.tight_layout(rect=(0.02, 0.07, 1, 0.94))
fig.savefig(out_dir / "map_with_lenses.png", dpi=200, facecolor=SURFACE)
