"""Score the 79-image real-world test set (Euclid Q1 VIS cutouts, all
already-verified lenses -- 75 ordinary + 4 unconfirmed compound candidates)
for likelihood of being a COMPOUND (double-ring) lens, using everything built
on this branch: the radial-profile ring-counter (radial_profile.count_rings)
for the actual compound signal, and the Euclid Zoobot lens head (v2,
shortcut-fixed) as a secondary "does this even look like a lens" sanity
check. Produces a 1-79 ranked CSV (rank 1 = strongest compound candidate) for
human review triage, plus a validation check against the 4 known candidates
and a result grid.

Data: data/real_test_79/ (copied from the provided 79-image set: raw Euclid
VIS npy cutouts, 100x100px @ 0.1"/px, no scaling -- same format/survey as the
23-image real_euclid set already used throughout this project).
"""
from __future__ import annotations

import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from radial_profile import count_rings
from zoobot_classifier import ZoobotLensClassifier
from train_cnn import pick_device

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "real_test_79"
OUT_DIR = ROOT / "outputs"

# Score-formula constants (see score_components() docstring for why).
MARGIN_SATURATION = 0.10    # angular coverage at which the shape term saturates to 1.0
STRENGTH_SATURATION = 15.0  # prominence_sigma at which the strength term saturates to 1.0
NEAR_EDGE_PENALTY = 0.05


def load_dataset():
    rows = list(csv.DictReader(open(DATA_DIR / "test_lenses.csv")))
    images = []
    for row in rows:
        images.append(np.load(DATA_DIR / row["npy_file"]))
    images = np.stack(images).astype(np.float32)
    return rows, images


def fixed_stretch(images: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5):
    """One global percentile stretch fit across the whole 79-image set and
    applied identically to every image -- same convention as
    prepare_real_euclid.py, so pixel values stay comparable across images
    instead of each getting its own per-image contrast boost."""
    lo, hi = np.percentile(images, [lo_pct, hi_pct])
    stretched = np.clip((images - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return stretched, float(lo), float(hi)


@torch.no_grad()
def zoobot_p_lens(images: np.ndarray, device) -> np.ndarray:
    model = ZoobotLensClassifier(freeze_encoder=True)
    model.head.load_state_dict(torch.load(OUT_DIR / "zoobot_euclid_lens_injected_v2.pt", map_location="cpu"))
    model = model.to(device).eval()
    x = torch.from_numpy(images).unsqueeze(1).to(device)
    probs = []
    for i in range(0, len(x), 32):
        logits = model(x[i:i + 32])
        probs.append(logits.sigmoid().squeeze(-1).cpu().numpy())
    return np.concatenate(probs)


def score_components(image: np.ndarray, p_lens: float):
    """Compound-candidate score for ONE image. Documented formula (all terms
    bounded so the integer tier always dominates -- the other terms only
    ever break ties WITHIN a tier, never cross into the next):

      tier  = 2 if >=2 ring-like peaks found, else 1 if exactly 1, else 0
      score = tier
            + 0.4 * strength_norm(2nd-strongest ring-like peak)  <- in [0, 0.4]
            + 0.3 * shape_norm(2nd-strongest ring-like peak)     <- in [0, 0.3]
            + 0.1 * p_lens                                       <- in [0, 0.1]
            - 0.05 * (top-2 ring-like peaks flagged near_edge)   <- in [-0.1, 0]

    tier deliberately CAPS at 2 rather than using the raw peak count. First
    version of this script used raw n_ring_like as the tier, which caused a
    real bug found by inspecting the actual top-ranked images: several cases
    where the radial profile is one genuine ring plus two tiny noise ripples
    (each just barely clearing the detection floor) got n_ring_like=3 and
    outranked a known genuine 2-ring compound candidate (Cosmic Ammonite)
    outright, no matter how strong Cosmic Ammonite's two real peaks were --
    because 3 > 2 regardless of peak quality. Capping the tier and judging
    quality by the strongest PAIR of peaks (not the full count) fixes this:
    a 3-peak case with one weak peak now competes on equal footing with a
    clean 2-peak case, differentiated by which pair is actually stronger.

    "2nd-strongest" = sort ring-like peaks by prominence_sigma descending,
    take the SECOND one -- i.e. the weaker half of the best candidate pair.
    For an image with exactly 2 ring-like peaks this is just the weaker of
    the two (same weakest-link logic as before); for 3+, extra marginal
    peaks beyond the best pair are ignored rather than penalizing or boosting
    the score.

    strength_norm = prominence_sigma (how many multiples of the residual's
    own noise floor a peak's prominence reaches -- see
    radial_profile.count_rings), normalized to an empirical saturation point:
    Cosmic Ammonite's weaker real peak is 8.7 sigma, stronger is 24.2 sigma;
    15.0 was chosen from that as roughly "clearly real."

    shape_norm = how far a peak's angular coverage sits above the ring/
    companion threshold (0.02, see radial_profile.classify_peak), normalized
    against the empirical genuine-arc saturation point (~0.10 coverage)."""
    n_ring_like, classifications, radii, profile, residual = count_rings(image)
    ring_peaks = [c for c in classifications if c["classification"] == "ring-like"]
    best_pair = sorted(ring_peaks, key=lambda c: c["prominence_sigma"], reverse=True)[:2]

    tier = min(n_ring_like, 2)
    if len(best_pair) >= 2:
        weaker = min(best_pair, key=lambda c: c["prominence_sigma"])
        strength_norm = float(np.clip(weaker["prominence_sigma"] / STRENGTH_SATURATION, 0.0, 1.0))
        shape_norm = float(np.clip((weaker["angular_coverage"] - 0.02) / (MARGIN_SATURATION - 0.02), 0.0, 1.0))
        n_near_edge = sum(1 for c in best_pair if c["near_edge"])
    elif len(best_pair) == 1:
        c = best_pair[0]
        strength_norm = float(np.clip(c["prominence_sigma"] / STRENGTH_SATURATION, 0.0, 1.0))
        shape_norm = float(np.clip((c["angular_coverage"] - 0.02) / (MARGIN_SATURATION - 0.02), 0.0, 1.0))
        n_near_edge = 1 if c["near_edge"] else 0
    else:
        strength_norm = shape_norm = 0.0
        n_near_edge = 0

    score = tier + 0.4 * strength_norm + 0.3 * shape_norm + 0.1 * p_lens - NEAR_EDGE_PENALTY * n_near_edge

    peak_radii_px = [round(c["radius"], 1) for c in ring_peaks]
    separation_ratio = (max(peak_radii_px) / min(peak_radii_px)) if len(peak_radii_px) >= 2 else None

    return dict(
        score=score, n_peaks_total=len(classifications), n_ring_like=n_ring_like,
        ring_peak_radii_px=peak_radii_px,
        ring_peak_coverage_pct=[round(c["angular_coverage"] * 100, 1) for c in ring_peaks],
        ring_peak_sigma=[round(c["prominence_sigma"], 1) for c in ring_peaks],
        weakest_peak_sigma=round(min((c["prominence_sigma"] for c in ring_peaks), default=0.0), 1),
        separation_ratio=round(separation_ratio, 2) if separation_ratio else None,
        n_near_edge=n_near_edge,
        classifications=classifications, radii=radii, residual=residual,
    )


def recommended_label(n_ring_like: int, p_lens: float) -> str:
    if n_ring_like >= 2:
        return "COMPOUND CANDIDATE"
    if n_ring_like == 1:
        return "single lens" if p_lens >= 0.5 else "single ring, low lens-confidence -- review"
    return "lens-like, no clear ring peak -- review" if p_lens >= 0.5 else "no ring signal detected"


def main():
    rows, images = load_dataset()
    stretched, lo, hi = fixed_stretch(images)
    print(f"Loaded {len(rows)} images. Fixed stretch: low={lo:.5f} high={hi:.5f}\n")

    device = pick_device("cpu")
    p_lens_all = zoobot_p_lens(stretched, device)

    results = []
    for row, img, p_lens in zip(rows, stretched, p_lens_all):
        comp = score_components(img, float(p_lens))
        results.append(dict(
            id=row["id"], name=row["name"], ground_truth_compound=row["compound"],
            grade=row["grade"], einstein_radius_arcsec=row["einstein_radius_arcsec"],
            p_lens_zoobot=round(float(p_lens), 3),
            label=recommended_label(comp["n_ring_like"], p_lens),
            **comp,
        ))

    results.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(results, start=1):
        r["rank"] = i

    # --- Write the ranked CSV ---
    csv_path = OUT_DIR / "real_test_79_scores.csv"
    fieldnames = ["rank", "id", "name", "label", "score", "n_ring_like", "n_peaks_total",
                  "ring_peak_radii_px", "ring_peak_coverage_pct", "ring_peak_sigma",
                  "weakest_peak_sigma", "separation_ratio",
                  "p_lens_zoobot", "ground_truth_compound", "grade", "einstein_radius_arcsec"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"Saved {csv_path}\n")

    # --- Validation against the 4 known (candidate) compound lenses ---
    print("=== Validation: where do the 4 known compound candidates rank? ===")
    known = [r for r in results if r["ground_truth_compound"] == "True"]
    for r in sorted(known, key=lambda r: r["rank"]):
        print(f"  rank {r['rank']:2d}/79  {r['name']:16s}  score={r['score']:.2f}  "
              f"n_ring_like={r['n_ring_like']}  label={r['label']}")
    median_rank = np.median([r["rank"] for r in known])
    print(f"  Median rank of the 4 known candidates: {median_rank:.1f} (79=worst, 1=best; "
          f"random chance median would be ~40)\n")

    n_flagged = sum(1 for r in results if r["n_ring_like"] >= 2)
    print(f"Total flagged COMPOUND CANDIDATE (n_ring_like>=2): {n_flagged}/79")
    print(f"Of those, how many are in the known-4: "
          f"{sum(1 for r in known if r['n_ring_like'] >= 2)}/4\n")

    # --- Result grid: top 12 ranked images ---
    build_grid(results[:12], stretched_lookup=dict(zip([row["id"] for row in rows], stretched)))

    return results


def build_grid(top_results, stretched_lookup):
    n = len(top_results)
    fig, axes = plt.subplots(n, 2, figsize=(8, 2.6 * n))
    for i, r in enumerate(top_results):
        img = stretched_lookup[r["id"]]
        axes[i, 0].imshow(img, cmap="gray", origin="lower", vmin=0, vmax=1)
        gt = " [GT: compound candidate]" if r["ground_truth_compound"] == "True" else ""
        axes[i, 0].set_title(f"#{r['rank']} {r['name'] or r['id'][:12]}{gt}", fontsize=8)
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])

        axes[i, 1].plot(r["radii"], r["residual"], color="0.3", lw=1)
        axes[i, 1].axhline(0, color="0.8", lw=0.8)
        for c in r["classifications"]:
            color = "crimson" if c["classification"] == "ring-like" else "darkorange"
            axes[i, 1].axvline(c["radius"], color=color, lw=1, ls="--")
        axes[i, 1].set_title(f"score={r['score']:.2f}  label={r['label']}", fontsize=8)
        axes[i, 1].set_xlabel("radius (px)", fontsize=7)

    fig.suptitle("Top 12 ranked real test images (1-79 compound-candidate scoring)", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = OUT_DIR / "real_test_79_top12.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
