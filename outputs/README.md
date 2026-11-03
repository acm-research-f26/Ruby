# outputs/ -- what every file is

Everything here is generated -- nothing is hand-written. Regenerate any of it
with the matching script in `src/` (see `src/README.md`).

## Trained model checkpoints

| File | What it is |
|---|---|
| `simple_lens_cnn.pt` | Trained `SimpleLensCNN` weights. |
| `lens_vit.pt` | Trained ViT weights. |
| `zoobot_lens_classifier.pt` | Zoobot linear head trained against the **old, general-purpose** Zoobot encoder. Kept for comparison. |
| `zoobot_euclid_lens_classifier.pt` | Zoobot linear head trained against the **Euclid-specific** encoder, sim data only -- the baseline before lens injection. Only ~5KB because the head is just one linear layer; the 15M-parameter encoder itself is downloaded separately (not stored in the repo) and shared/frozen. |
| `zoobot_euclid_lens_injected.pt` | Same, trained with lens-injected real backgrounds (v1) -- recall is much better but it learned a real-texture shortcut (see `evaluate_lens_injection.py` in src/README.md). Kept for comparison, not recommended to use as-is. |
| `zoobot_euclid_lens_injected_v2.pt` | Same, v2 training set (injection + plain real negatives from the same donors) -- the shortcut-fix attempt. Recall gain mostly reverted too, which is itself the important finding: v1's gain was substantially the shortcut, not genuine arc-detection improvement. See `outputs/lens_injection_before_after.png`. |

## Evaluation

| File | What it is |
|---|---|
| `evaluation_summary.txt` | Accuracy/precision/recall for all 3 models across every test set (simulated + real). The headline sim-to-real gap number comes from here. |

## Whole-model DeepDream (what does the *whole* model consider lens-like)

| File / folder | What it is |
|---|---|
| `deepdream_cnn/`, `deepdream_vit/`, `deepdream_zoobot/` | Raw DeepDream output images per model (one per seed image: noise, a real lens, a real non-lens). |
| `deepdream_compare_grid.png` | Side-by-side comparison across all 3 models -- the "checkerboard shortcut" finding is visible here. |
| `deepdream_vit_grid.png` | ViT-specific version of the same comparison. |

## Filter-level interpretability (new on this branch)

| File / folder | What it is |
|---|---|
| `filter_firing_report.json` | The core result: for each of the Zoobot encoder's 4 stages, the top-10%-active filters for small-radius rings vs. baseline rings, which filters overlap, and the Jaccard overlap score. Start here for the raw numbers. |
| `filter_portraits/` | 28 small images -- the **failed** attempt to visualize what each divergent filter responds to via per-filter DeepDream. Every image looks like generic noise; this is a documented negative result, not a bug. |
| `filter_portraits_grid.png` | All 28 of the above laid out in one labeled grid, for a quick "yes, these really are all indistinguishable" check. |
| `top_activating_images.json` | For each divergent filter, the 5 real images (from the full simulated set + the real Euclid images) that activate it most strongly, with activation values. This is what the filter names are actually based on. |
| `top_activating_small_only.png` | Grid: one row per small-radius-only filter, its top-5 real activating images. **This is the one to look at** to see why each filter got the name it did. |
| `top_activating_baseline_only.png` | Same, for the baseline (larger-ring) filters. |

## Lens injection results

| File | What it is |
|---|---|
| `lens_injection_sanity_check.png` | 6 real non-lens backgrounds next to their injected-arc versions, so you can see the injection actually looks reasonable before trusting any downstream result. |
| `lens_injection_before_after.png` | Accuracy/precision/recall/F1, sim-only vs. lens-injected, on 17 held-out real images. Headline: recall went from 0.42 to 0.92 (v1), but precision collapsed from 1.00 to 0.69 -- diagnosed as a real-texture shortcut, not genuine improvement (see src/README.md). |

## Compound lens detection (classical method, new on this branch)

| File | What it is |
|---|---|
| `radial_profile_real_lenses.png` | The 12 real verified single lenses, their radial profile residuals, and which peaks (if any) get classified ring-like vs. point-like/companion. Currently 1/12 flagged, with the flagged case's ambiguity explained in the plot/report. |
| `radial_profile_sim_lenses.png` | Same, on 12 simulated single lenses where the ground truth is known (exactly 1 ring) -- the false-positive-rate check. Currently 0/12 false positives. |
| `double_ring_sensitivity.npz` | Raw detection-rate grid (5 separation ratios x 4 brightness ratios x 15 trials each) from the synthetic double-ring sensitivity sweep -- the validation that the method can actually detect a second ring when one exists, not just stay quiet on single lenses. |
| `double_ring_sensitivity_heatmap.png` | The same grid as a heatmap. **Read this one first** -- it's the honest summary: detection is weak (7-20%) when the two rings are close together (<1.5x radius ratio, where they genuinely blend -- matches what the team already expected), and much better (47-80%) once they're clearly separated. |

## Real Euclid Q1 data previews

| File / folder | What it is |
|---|---|
| `real_euclid_grid.png` | Grid of the 23 real Euclid cutouts (lens candidates + non-lenses). |
| `real_euclid_preview/lens/`, `real_euclid_preview/nonlens/` | The same 23 images as individual PNGs. |

## Simulated data previews

| File | What it is |
|---|---|
| `sample_grid.png` | Grid of simulated lens vs. non-lens examples (balanced batch). |
| `sample_grid_batch2.png` | Same, for the second (imbalanced, 10%/90%) simulated batch. |

## Research write-up

The literature-review report on improving Zoobot (fine-tuning strategy, lens injection,
cascade approaches, domain adaptation) is not on this branch -- it's on the `BD5` branch,
under `reports/Zoobot lens detection improvements.md`.
