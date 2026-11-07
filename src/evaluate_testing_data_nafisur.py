"""Cross-team model evaluation: run our compound-lens scoring pipeline
(radial_profile.count_rings + the Euclid Zoobot lens head -- exactly the
score built in score_real_test_set.py, reused unchanged here) on the
independent 3000-image simulated test set pushed to the `Testing-Data`
branch by a teammate, and report the two metrics requested for
cross-model comparison:

1. AUCROC -- is_compound (ground truth) vs. our continuous score, since
   this set (unlike our own real-world data) has labels.
2. FPR @ 80% recall -- the team's specified protocol for comparing models
   on equal footing: pick each model's own score threshold so it catches
   80% of the TRUE simulated compounds, then report what fraction of REAL,
   unlabeled Euclid candidates (our existing 79-image real test set) that
   same threshold flags. Lower = a quieter, more useful review queue at
   matched sensitivity.

Data: data/testing_data_nafisur/ (manifest.csv + test/*.npy + test/labels.csv,
copied from the `Testing-Data` branch -- 1000 single, 1000 compound, 500
"nearby" [lens + unrelated companion object, a deliberate hard negative for
a radius/shape-based compound detector], 500 non_lens).
"""
from __future__ import annotations

import csv
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score

from radial_profile import count_rings
from train_cnn import pick_device
from score_real_test_set import score_components, zoobot_p_lens

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "testing_data_nafisur"
OUT_DIR = ROOT / "outputs"
TARGET_RECALL = 0.80


def load_dataset():
    manifest = {r["id"]: r for r in csv.DictReader(open(DATA_DIR / "manifest.csv"))}
    labels = {r["id"]: r for r in csv.DictReader(open(DATA_DIR / "test" / "labels.csv"))}
    ids = sorted(manifest.keys())
    images = np.stack([np.load(DATA_DIR / manifest[i]["path"]) for i in ids]).astype(np.float32)
    class_name = [labels[i]["class_name"] for i in ids]
    is_compound = np.array([int(labels[i]["is_compound"]) for i in ids])
    return ids, images, class_name, is_compound


def fixed_stretch(images: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5):
    """Same convention as score_real_test_set.py / prepare_real_euclid.py:
    ONE global percentile stretch fit across this whole new set (its pixel
    units are raw lenstronomy flux, not already [0,1]), not per-image."""
    lo, hi = np.percentile(images, [lo_pct, hi_pct])
    stretched = np.clip((images - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return stretched, float(lo), float(hi)


def main():
    ids, images, class_name, is_compound = load_dataset()
    stretched, lo, hi = fixed_stretch(images)
    print(f"Loaded {len(ids)} images from Testing-Data. Fixed stretch: low={lo:.5f} high={hi:.5f}")
    print(f"Class breakdown: {dict((c, class_name.count(c)) for c in set(class_name))}\n")

    device = pick_device("cpu")
    p_lens = zoobot_p_lens(stretched, device)

    scores = np.array([score_components(img, float(p))["score"]
                        for img, p in zip(stretched, p_lens)])

    # --- AUCROC ---
    auc_full = roc_auc_score(is_compound, scores)
    single_or_compound = np.array([c in ("single", "compound") for c in class_name])
    auc_core = roc_auc_score(is_compound[single_or_compound], scores[single_or_compound])
    print(f"AUCROC (is_compound vs. score, all {len(ids)} images incl. nearby/non_lens): {auc_full:.3f}")
    print(f"AUCROC (single vs. compound only, n={single_or_compound.sum()}, the method's "
          f"actual intended task): {auc_core:.3f}\n")

    # --- FPR @ 80% recall, per the team's protocol ---
    compound_scores = scores[is_compound == 1]
    threshold = float(np.percentile(compound_scores, (1 - TARGET_RECALL) * 100))
    achieved_recall = float((compound_scores >= threshold).mean())
    print(f"Threshold for {TARGET_RECALL:.0%} recall on {len(compound_scores)} sim compounds: "
          f"score >= {threshold:.3f} (achieved recall: {achieved_recall:.1%})\n")

    for name in ["single", "nearby", "non_lens"]:
        mask = np.array([c == name for c in class_name])
        flagged = (scores[mask] >= threshold).mean()
        print(f"  At that threshold, sim '{name}' (n={mask.sum()}) flagged: {flagged:.1%}")

    # Apply the same threshold to our own real, (mostly) unlabeled 79-image set.
    real_csv = OUT_DIR / "real_test_79_scores.csv"
    real_rows = list(csv.DictReader(open(real_csv)))
    real_scores = np.array([float(r["score"]) for r in real_rows])
    real_flagged = (real_scores >= threshold).mean()
    n_flagged = int((real_scores >= threshold).sum())
    print(f"\n  At that threshold, REAL Euclid candidates (n={len(real_rows)}) flagged: "
          f"{real_flagged:.1%} ({n_flagged}/{len(real_rows)})")

    # Save the per-image scored table.
    out_csv = OUT_DIR / "testing_data_nafisur_scores.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "class_name", "is_compound", "score", "p_lens_zoobot",
                    "flagged_at_80pct_recall_threshold"])
        for i, c, ic, s, p in zip(ids, class_name, is_compound, scores, p_lens):
            w.writerow([i, c, ic, round(float(s), 4), round(float(p), 3), int(s >= threshold)])
    print(f"\nSaved {out_csv}")

    return dict(auc_full=auc_full, auc_core=auc_core, threshold=threshold,
                real_flagged_frac=real_flagged, real_flagged_n=n_flagged, n_real=len(real_rows))


if __name__ == "__main__":
    main()
