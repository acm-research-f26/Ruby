# Multi-Band DSPL Color Pilot

This pilot extends the DSPL generator with three simulated bands (g, r, i) and a color-redshift model, then unravels the ringed images into radial profiles to see whether simple and compound lens structures look measurably different.

## What the generator does

`generate_color_dspl_experiment.py` renders each scene in three bands. All bands share the project's camera-style settings (100x100 pixels, 0.1 arcsec/pixel, 90-second exposure, zero point 27, read noise 7, gain 6.083, sky brightness 23.5, Gaussian PSF with 0.16 arcsec FWHM). Bands differ only through the source and lens magnitudes fed into each one.

### Redshift-dependent color (placeholder model)

Background sources get redder with redshift through simple linear laws:

- `g - r = 0.2 + 0.25 * z`
- `r - i = 0.15 + 0.15 * z`

Source 1 (z=1.0) therefore has g-r = 0.45 and r-i = 0.30; source 2 (z=2.0) has g-r = 0.70 and r-i = 0.45. The lens galaxy has fixed red colors (g-r = 1.0, r-i = 0.6). **These are pilot assumptions, not K-corrections.** Real band fluxes require SED templates and filter curves; treat the colors as knobs for testing the pipeline, not as measured astrophysics.

### Simple vs compound structure

Each scene is rendered twice: once with a single SIE deflector ("simple") and once with two offset SIE deflectors ("compound"), using the same two-component convention as the earlier `generate_lenses.py` compound set. The compound lens here is two components on the **same** lens plane at z=0.3 — not a multi-plane compound model with a second lens redshift. Compound lens centers and component Einstein radii are sampled per replicate seed from the same ranges used by the earlier generator and are paired across all conditions.

### Sweeps and controls

The same three one-factor sweeps from the DSPL pilot apply to each structure: source faintness offsets (0-3 mag, applied in all bands), Einstein-radius scale (0.50, 0.55, 0.70, 0.85, 1.00, 1.15, 1.30), and foreground lens-light flux scale (0.5-4x). Negative controls keep both sources and lens light but shrink the Einstein radius to 0.01 arcsec. The 30-replicate run produces 960 scenes x 3 bands: 900 positive images (15 conditions x 2 structures x 30 replicates) plus 60 negative controls (30 per structure).

### PNGs vs raw arrays

Every scene saves a viewable grayscale PNG per band **and** the raw flux array per band as `.npy`. This matters because the PNGs are normalized image-by-image, which destroys relative band flux and therefore color. All quantitative analysis uses the `.npy` arrays.

## What the analyzer does

`analyze_color_profiles.py` reads the raw arrays and unravels each image into radial profiles: mean flux per annulus around the lens center (x = radius, y = flux). From those it computes per-image morphology metrics:

- `ring_contrast` - outer-ring peak relative to the central lens light.
- `peak_radius_arcsec` - radius of the brightest ring annulus (tracks arc size).
- `outer_to_inner_flux_ratio` - how much light sits in the ring region vs the center.
- `quadrant_asymmetry` - (max-min)/(max+min) ring flux across four quadrants.
- `radial_peak_count` - number of bumps in the radial profile.
- `inner/outer g-r` and `color_gradient_g_minus_r` - color of the central region vs the ring region.

Outputs in `dspl_color_experiment/analysis/`:

- `metrics_per_image.csv` and `summary_by_condition.csv`.
- `fig1_example_radial_profiles.png` - unraveled rings for a simple and a compound control.
- `fig2_ring_contrast_by_condition.png` - ring contrast across each sweep, by structure.
- `fig3_color_profile_by_structure.png` - mean g-r vs radius by structure (annuli where noise makes flux non-positive are skipped).
- `fig4_structure_metric_scatter.png` - asymmetry vs ring contrast; a quick look at simple/compound separability.
- `fig5_class_radial_profiles.png` - mean unraveled-ring profile per band (rows) for four classes (columns): non-lens controls, simple lenses, compound lenses, and all lenses pooled. Control condition only, so classes are compared like-for-like.
- `fig6_class_color_profiles.png` - mean g-r color vs radius for the same four classes; shows the color variation between non-lenses and each lens type.
- `profiles/{experiment}/{structure}/{image_id}_profile.png` - one unraveled-ring graph per image: g/r/i annulus profiles on top (band colors match the RGB composites), g-r color profile below. Pass `--skip-per-image-plots` to skip these.

These are morphology measurements, not Zoobot scores, accuracy, or detections.

## Viewable color images

`make_rgb_previews.py` builds true RGB composites from the raw arrays: i band to the red channel, r to green, g to blue, with one asinh stretch shared across the three bands of each scene so band ratios (colors) survive. Run:

```powershell
python make_rgb_previews.py
```

This writes one composite per scene under `dspl_color_experiment/rgb/{experiment}/{structure}/` plus `analysis/fig0_rgb_gallery.png`, a grid of the faintness sweep with simple and compound structures side by side. In these composites the red elliptical lens galaxy should look yellow-red while the bluer high-redshift arcs lean blue-green - that color difference is exactly the signal the color-gradient metrics measure.

## Run it

```powershell
python generate_color_dspl_experiment.py --replicates 30
python analyze_color_profiles.py
python make_rgb_previews.py
```

Both scripts use fixed seeds, so rerunning reproduces the same scenes. Generated images, arrays, and analysis outputs are excluded from Git via `.gitignore`.

## Zoobot inference on the color dataset

After training the classifier head (see the DSPL README), each band can be scored with the shared checkpoint. The manifest carries per-band image paths, so pick the band with `--image-column`:

```powershell
python run_zoobot_inference.py --manifest dspl_color_experiment/simulation_manifest.csv --checkpoint model_checkpoints/ruby_synthetic_lens_head/checkpoints/0.ckpt --output dspl_color_experiment/zoobot_scores_r.csv --image-column image_path_r
python summarize_color_zoobot.py
```

`run_zoobot_inference.py` converts each band PNG to the same grayscale 224-pixel single-channel input used in training - Zoobot never sees color; bands are scored independently. `summarize_color_zoobot.py` writes `zoobot_analysis/accuracy_by_band.csv`, `detection_by_condition.csv`, and `detection_by_lens_category.csv` (the same binned format as the grayscale pilot, plus a simple/compound split).

Measured on the 960-scene run (900 lenses, 60 negative controls), argmax decisions:

- Overall accuracy 0.92 in every band (g 0.929, r 0.926, i 0.919); zero false positives on the 60 controls.
- Detection is 100% in every faintness and blending bin for both structures.
- Arc size remains the only failure mode: at the 0.50-0.55 scale, simple lenses are detected 25-30% of the time while compound lenses reach 65-72% (two arc systems give the classifier more signal). At 0.70-0.85, simple reaches 85% and compound 100%.
- Band choice barely matters (a few percentage points at most), as expected for a grayscale single-channel model; g is marginally best, i marginally worst.

Per-value detection rates (r band, n=30 per cell; g and i differ by a few points - see `zoobot_analysis/detection_by_condition.csv`):

| Arc-size scale | Simple | Compound |
| ---: | ---: | ---: |
| 0.50 | 4/30 (13%) | 21/30 (70%) |
| 0.55 | 11/30 (37%) | 22/30 (73%) |
| 0.70 | 21/30 (70%) | 30/30 (100%) |
| 0.85 | 30/30 (100%) | 30/30 (100%) |
| 1.00 | 30/30 (100%) | 30/30 (100%) |
| 1.15 | 30/30 (100%) | 30/30 (100%) |
| 1.30 | 30/30 (100%) | 30/30 (100%) |

| Faintness offset | Simple | Compound |
| ---: | ---: | ---: |
| 0 mag (control) | 30/30 (100%) | 30/30 (100%) |
| +1 mag | 30/30 (100%) | 30/30 (100%) |
| +2 mag | 30/30 (100%) | 30/30 (100%) |
| +3 mag | 30/30 (100%) | 30/30 (100%) |

| Blending scale | Simple | Compound |
| ---: | ---: | ---: |
| 0.5x | 30/30 (100%) | 30/30 (100%) |
| 1x (control) | 30/30 (100%) | 30/30 (100%) |
| 2x | 30/30 (100%) | 30/30 (100%) |
| 4x | 30/30 (100%) | 30/30 (100%) |

Negative controls: 0/60 flagged in every band. Mean lens scores fall smoothly through the small-arc zone (simple: 0.37 at 0.50 to 0.73 at 0.85), so most "misses" are low-confidence calls, not confident non-lens classifications.

These are synthetic-demo numbers: the head was trained on this project's simulations, so they demonstrate the pipeline rather than real-survey accuracy.

## What the 30-replicate run showed

With 30 replicates per condition (n=30 per cell, 60 negative controls), the intended trends were measurable with tightened standard deviations:

- Faintness lowered ring contrast monotonically (compound: 1.93 at control to 0.30 at +3 mag; simple: 1.07 to 0.09; negative controls sit at 0.16/0.06).
- Blending lowered ring contrast monotonically (compound: 3.33 at 0.5x lens light to 0.59 at 4x; simple: 2.00 to 0.29).
- The radial peak position tracked the Einstein-radius scale outward (simple: 0.73 to 1.62 arcsec across the 0.50-1.30 scale range; compound: 0.77 to 1.93 arcsec).
- Compound structures separated from simple ones at matched conditions: higher ring contrast (e.g. 1.93 vs 1.07 at control), higher outer-to-inner flux ratio (13.1 vs 10.7), and slightly higher quadrant asymmetry (0.39 vs 0.31). Standard deviations shrink roughly as expected with 6x more samples than the pilot.

These are morphology numbers describing how the metrics respond to the knobs on simulated data - not evidence about real-survey performance.

## Real-data check: 79 Euclid lenses

`test_lenses_79/` holds 79 real Euclid Q1 VIS cutouts (75 ordinary lenses, 4 compound candidates from Li et al. 2026; same 100x100, 0.1 arcsec/pixel format as the sims; see `test_lenses_79/README.txt` for provenance). Score them and evaluate with:

```powershell
python run_zoobot_inference.py --manifest test_lenses_79/manifest.csv --checkpoint model_checkpoints/ruby_synthetic_lens_head/checkpoints/0.ckpt --output test_lenses_79/zoobot_scores.csv
python evaluate_real_lenses.py
```

`evaluate_real_lenses.py` writes `test_lenses_79/real_lens_evaluation.csv` and reports the project's three evaluation pieces:

**1. AUROC on never-seen simulations** - the head was trained on the 64x64 single-source-plane `synthetic_lenses.csv` set, so the 100x100 two-plane color DSPL sims are genuinely held out.

| Scorer | All sims | Compound vs non-lens | Simple vs non-lens |
| --- | ---: | ---: | ---: |
| Zoobot head | 0.9986 | 1.0000 | 0.9966 |
| Ring-contrast baseline (non-ML) | 0.9791 | 0.9999 | - |

The baseline nearly matches the model, meaning the sims alone cannot strongly discriminate methods.

**2. Flags at the 80% compound-recall cutoff** - cutoff 0.730 catches 80.2% of sim compounds and 0/60 sim non-lenses. Applied to the real set:

| Real subset | Flagged at cutoff |
| --- | ---: |
| 4 compound candidates | 3/4 (75%) |
| 75 ordinary lenses | 23/75 (30.7%) |
| All 79 lenses | 26/79 (32.9%) |

**3. Real compound rankings** - mean rank 21.8 of 79 by lens score (rank 1 = highest). Real compounds outscore ordinary lenses (median 0.756 vs 0.683), and plain argmax detects 72/79 real lenses including 4/4 compounds.

| Compound candidate | Rank of 79 | Lens score | Euclid theta_E |
| --- | ---: | ---: | ---: |
| Cosmic Ammonite | 1 | 0.863 | 1.16" |
| Cosmic Dartboard | 10 | 0.778 | 0.46" |
| Galileo's Lens | 26 | 0.733 | 1.09" |
| Teapot Lens | 50 | 0.646 | 0.63" |

Caveats: the real set contains no non-lenses, so no real-data false-positive rate or AUROC is computable (the 32.9% flag rate on ordinary lenses is not an FPR - they are all true lenses). Four compounds is an anecdote-sized sample. The compound candidates are not spectroscopically confirmed. Sim AUROCs near 1.0 partly reflect that sims and training data come from the same generator family even though the specific images are held out.
