# Model Evaluation: AUCROC, FPR @ 80% Recall, Mean Rank

Cross-team comparison metrics for our compound-lens pipeline (the Euclid
Zoobot lens head + `radial_profile.count_rings` compound-candidate score,
documented in [`COMPOUND_LENS_DETECTION.md`](COMPOUND_LENS_DETECTION.md) and
[`REAL_WORLD_TEST.md`](REAL_WORLD_TEST.md)), run on the independent
3000-image simulated test set pushed to the `Testing-Data` branch, plus the
real-data numbers already established on this branch.

All code: `src/evaluate_testing_data_nafisur.py`. Output table (per-image
score, class, flag decision): `outputs/testing_data_nafisur_scores.csv`.

## Summary

| Metric | Value |
|---|---|
| AUCROC (is_compound vs. score, all 3000 Testing-Data images) | **0.642** |
| AUCROC (single vs. compound only, n=2000 -- the method's actual intended task) | **0.614** |
| FPR @ 80% recall: real Euclid candidates flagged (n=79) | **55.7% (44/79)** |
| FPR @ 80% recall: sim singles flagged (n=1000, false positives) | **68.6%** |
| FPR @ 80% recall: sim "nearby" flagged (n=500, hard negative) | **74.2%** |
| FPR @ 80% recall: sim non-lenses flagged (n=500) | **27.0%** |
| Mean rank of the 4 known real compound candidates (of 79, 1=best) | **22.0** (median 23.0) |

**Read plainly: this is a weak result, not a good one, on the Testing-Data
set specifically** (see diagnosis below) -- reported as measured, not
adjusted or spun. The real-data mean-rank result (22.0/79, vs. ~40 random)
remains the stronger piece of evidence for the method; see
[`REAL_WORLD_TEST.md`](REAL_WORLD_TEST.md) for that full writeup, including
which 2 of the 4 known candidates it actually ranked well.

The AUCROC on a separate dataset that was mentioned as coming later hasn't
arrived yet -- this page will be updated once it does.

## Methodology

**AUCROC**: our score (continuous, unbounded above ~0-3) against the
Testing-Data set's `is_compound` ground-truth label, via
`sklearn.metrics.roc_auc_score`. Reported two ways: across all 3000 images
(including the "nearby" and "non_lens" classes, which aren't really what
the method is trying to separate), and restricted to just the 2000
single/compound images (the method's actual job: distinguish a lens with
one ring from a lens with two).

**FPR @ 80% recall**, per the team's specified protocol for comparing
models on equal footing: find the score threshold that catches 80% of the
TRUE simulated compounds (recall = 0.80 on the known-compound subset), then
apply that exact threshold to real, mostly-unlabeled Euclid candidates (our
own 79-image real-world test set) and report what fraction gets flagged.
This simulates the actual deployment question: "at a sensitivity level the
team is comfortable with, how big is the resulting human-review queue?" A
lower flagged fraction at fixed recall = a quieter, more useful queue. We
also ran the same threshold against the Testing-Data set's own single/
nearby/non_lens classes, since those give an additional, larger-sample
false-positive estimate for free.

**Preprocessing**: one global 1st-99.5th percentile intensity stretch fit
across the whole 3000-image Testing-Data set (same convention used
throughout this project for new real/raw-flux image sets -- see
`prepare_real_euclid.py` and `score_real_test_set.py`), not a per-image
stretch.

## Diagnosis: why the Testing-Data numbers are weak

This isn't unexplained. Checking the actual ring-separation values behind
the `is_compound=1` systems in Testing-Data's `labels.csv` against our
detector's own calibrated resolution floor:

- Our `count_rings()` has a **hard minimum peak separation of 9px (0.9" at
  this project's 0.1"/px scale)**, calibrated specifically to suppress false
  "second rings" from a single asymmetric arc (documented in
  `COMPOUND_LENS_DETECTION.md`) -- below that separation, two rings
  physically cannot be resolved as two distinct radial peaks by this
  method, by construction.
- Testing-Data's compound systems have a **median ring separation of only
  0.41" (4.1px)** -- and **99.7% of them sit below our 0.9" floor**
  entirely. This matches that branch's own README, which notes its
  separation range (0.17"-0.86") was deliberately chosen to bracket the 4
  real confirmed compound lenses' actual measured separations -- i.e. this
  is a more astrophysically realistic (and harder) separation regime than
  our own earlier internal sensitivity sweep used (1.3x-4.0x of a 1.0"
  inner radius, i.e. 0.3"-3.0" separations).
- Consistent with that: comparing the 800 compounds our threshold caught
  vs. the 200 it missed, their mean separations are nearly identical
  (0.419" caught vs. 0.455" missed) -- separation isn't cleanly predicting
  hit/miss here, because almost the entire set is already below the
  method's resolution floor. At that point, which ones "pass" the
  threshold is driven more by overall image brightness/SNR than by genuine
  two-ring structure -- which also explains why true singles get flagged
  at nearly the same rate (68.6%) as true compounds are caught (80%): the
  method isn't cleanly separating the two classes in this regime, matching
  the near-chance 0.614 AUCROC.

**Bottom line**: our method's calibrated resolution floor (0.9" minimum
separation) is coarser than the separations most real compound lenses
actually have (0.17"-0.86", per the 4 confirmed systems). The 47-80%
detection rates reported in `COMPOUND_LENS_DETECTION.md`'s own sensitivity
sweep are real, but were measured at separations wider than what's
physically typical -- so they likely overstate real-world recall. This is
a genuine limitation to flag for the poster, not a bug to quietly patch:
lowering the separation floor would need the same specificity-regression
testing done for the earlier `local_window_px` fix (re-running the 0/12
and 1/12 false-positive checks) before it could be trusted, which hasn't
been done yet -- a natural next step, not completed here.

## Reproducing

```bash
cd src
python evaluate_testing_data_nafisur.py
```
