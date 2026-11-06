# FPR test: Zoobot3 vs. baseline

The team's three metrics for our 3-answer Zoobot, measured on the `Testing-Data` branch. This branch builds on [`realistic-sim-zoobot`](https://github.com/acm-research-f26/Ruby/tree/realistic-sim-zoobot), which explains how the models were made and tested.

**Zoobot3** is our 3-answer Zoobot (not a lens / lens / compound). The **baseline** is untouched Zoobot with a simple compound classifier on its features. Each model was trained 3 times (seeds 1, 2, 3), and every number below is the mean of the 3 runs, with per-run values in brackets. Neither model was trained or tuned on the test data.

## Results

| Metric | Data | Zoobot3 | Baseline | Better is |
|---|---|---|---|---|
| **AUROC**, compound vs. everything else | 3,000 simulated images from the `Testing-Data` branch | **0.924** (0.916, 0.931, 0.924) | 0.880 (0.842, 0.900, 0.897) | Higher (1 = perfect, 0.5 = guessing) |
| **FPR at 80% recall** | Same 3,000 images | **11.5%** (14.1%, 10.5%, 10.0%) | 21.2% (30.1%, 16.8%, 16.7%) | Lower |
| **Mean ranking of the 4 real compound lenses** | 79 real Euclid test lenses | **3.9** (3.75, 4.5, 3.5) | 9.6 (6.75, 12.0, 10.0) | Lower (perfect = 2.5, guessing = about 40) |

Zoobot3 is ahead of the baseline on all three metrics, on every run.

## What each metric means

- **AUROC:** how well the model's compound score separates the 1,000 simulated compound lenses from the 2,000 other images (single lenses, lenses with a nearby galaxy and non-lenses).
- **FPR at 80% recall:** each model's cutoff is set so it catches 800 of the 1,000 compound lenses. The FPR is the share of the other 2,000 images that also score above that cutoff. A lower FPR means a shorter, cleaner list for people to review.
- **Mean ranking:** all 79 real test lenses (75 ordinary, 4 compound candidates) are sorted by compound score, and this is the average position of the 4 compound candidates.

## Breakdown by image type

| Compound lenses vs. | AUROC, Zoobot3 | AUROC, baseline | FPR at 80% recall, Zoobot3 | FPR at 80% recall, baseline |
|---|---|---|---|---|
| Single lenses (1,000) | **0.897** | 0.862 | **15.5%** | 24.4% |
| Lenses with a nearby galaxy (500) | **0.903** | 0.823 | **15.1%** | 33.9% |
| Non-lenses (500) | **0.997** | 0.970 | **0.0%** | 2.1% |

For reference, the original 0.99 Zoobot can't tell compound lenses from single ones: AUROC 0.475 vs. single lenses, and 0.999 vs. non-lenses, because it only answers lens or not a lens.

## Caveats

- **The test images are simulated.** The mean ranking is the only real-data check, and it rests on 4 lenses.
- **These simulations have a giveaway.** In `Testing-Data`, no single lens or lens with a nearby galaxy has a ring smaller than 0.9″, while 77% of compound lenses have an inner ring smaller than that. Ring size alone would score an AUROC of 0.963 for compound vs. single. Real Euclid lenses aren't like this: 58% of them have rings under 0.9″.
- **Without the giveaway, the two models are about even.** On the 233 compound lenses whose inner ring is 0.9″ or larger, Zoobot3 scores 0.911 vs. everything else and the baseline 0.901, and against single lenses 0.880 vs. 0.887. Zoobot3's lead comes mostly from compound lenses with a small inner ring.
- **Simple measures do almost as well on this data.** A classifier using 6 simple brightness measures, trained and tested on these images with 5-fold cross-validation, reaches an AUROC of 0.916. These simulations are easier to separate than real lenses, so the absolute numbers are optimistic for every team model.
- **Suggested fix:** draw single-lens ring sizes from the same range as compound inner and outer rings (about 0.45–1.5″), so ring size alone can't give the answer.

## Reproducing

From `~/ruby`, with `lens_env311`, after cloning the `Testing-Data` branch (commit `2c0b8af`) into `data/testing_data`:

```bash
python -I sim/testing_data_metrics.py data/testing_data
python -I sim/testing_data_checks.py data/testing_data
```

The models come from `sim/train_models.py`. Outputs are in `sim/out/testing_data/`: `results.txt`, `metrics.csv`, `scores.csv` (every image's score for every model and run), `checks.txt` (the caveat checks) and `classes.png` (example test images of each type). The metrics script also re-scores our own simulated test pairs and checks that their AUROC matches the final test, which it does exactly.
