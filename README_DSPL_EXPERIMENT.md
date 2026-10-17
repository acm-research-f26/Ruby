# DSPL-Like Sensitivity Pilot

This pilot generates 100x100 grayscale images with one foreground SIE lens and two background Sérsic sources at different redshifts. It changes one factor at a time: source faintness, lens Einstein-radius scale, or foreground lens-light flux. A small unlensed/weakly lensed negative-control group is included.

## What is fixed and what is provisional

| Simulation setting | Value | Meaning |
| --- | --- | --- |
| Rendered image | 100 x 100 pixels | Simulated cutout size. |
| Pixel scale / field of view | 0.1 arcsec/pixel / 10 x 10 arcsec | Angular size represented by the image. |
| Zoobot input | 224 x 224 grayscale | Resized input expected by this encoder. |
| Exposure / zero point | 90 seconds / 27 | Reused project camera-style settings. |
| Read noise / gain | 7 / 6.083 | Reused detector settings; gain is the project value. |
| Sky brightness | 23.5 | Reused project sky setting. |
| PSF | Gaussian, 0.16 arcsec FWHM | Approximate seeing blur. |
| Pixel scaling | 0.5th-99.5th percentile to 8-bit | Existing project PNG processing convention. |

The lens/source assumptions are provisional, not established Ruby priors: one SIE lens at redshift 0.3, two Sérsic-ellipse sources at redshifts 1.0 and 2.0, and a baseline Einstein radius of 1.2 arcsec. Baseline source magnitudes use the notebook relation `20 + 1.5*z`, giving 21.5 and 23.0. Source effective radius uses `0.3/(1 + 0.5*z)` arcsec. Confirm these assumptions against a larger real sample before scientific claims. Random positions are paired by replicate seed across conditions.

| Sensitivity factor | Tested values | Physical interpretation |
| --- | --- | --- |
| Source magnitude offset | 0, 1, 2, 3 mag | Each +1 mag makes the source about 2.512 times fainter. Source magnitudes are (21.5, 23.0), (22.5, 24.0), (23.5, 25.0), and (24.5, 26.0) for the two planes. |
| Einstein-radius scale | 0.70, 0.85, 1.00, 1.15, 1.30 | Multiplies the 1.2 arcsec baseline, giving 0.84, 1.02, 1.20, 1.38, and 1.56 arcsec. |
| Foreground-light flux scale | 0.5, 1, 2, 4 | Multiplies lens-galaxy flux; corresponding lens magnitudes are 21.503, 20.750, 19.997, and 19.245. Larger flux means a brighter foreground lens and a stronger blending stress test. |
| Replicates | 5 per positive condition | 13 positive conditions x 5 = 65 DSPL-like images; plus 5 negative controls = 70 images total. |

The mean lensed-flux contributions recorded in the faintness sweep are:

| Magnitude offset | Source 1 mean lensed flux | Source 2 mean lensed flux |
| ---: | ---: | ---: |
| 0 | 1914.3 | 626.6 |
| 1 | 762.1 | 249.5 |
| 2 | 303.4 | 99.3 |
| 3 | 120.8 | 39.5 |

Flux values are Lenstronomy manifest values, not calibrated detector counts. Both source planes contribute nonzero lensed flux to every positive image.

## Generate the pilot

```powershell
python generate_dspl_experiment.py --replicates 5
```

This writes `dspl_experiment/simulation_manifest.csv` and individual PNGs under `dspl_experiment/images/`. Five replicates per condition is a small pilot, not a final power analysis. Each positive manifest row records nonzero lensed flux contributions from both source planes.

## Zoobot inference status

The notebook's `FinetuneableZoobotClassifier.predict_step` returns softmax probabilities. Since the notebook labels lenses as 1, the second output column is the lens probability. Its existing class decision uses `argmax`, not a separately tuned threshold. The notebook's `CatalogDataModule` test transform uses the project's minimal galaxy transform: 224-pixel input, grayscale PNGs become one-channel tensors, values convert to float in [0,1], and no mean/std normalization is enabled.

The Hugging Face model name in the notebook loads a pretrained galaxy encoder; it is not a lens classifier by itself. The classification head starts untrained, so inference must wait until that head has been trained and saved.

For a pipeline demonstration, you can train the head on the existing labeled synthetic images. This is necessary because no Ruby lens-classifier checkpoint is currently in the repository:

```powershell
python -m pip install -r requirements-zoobot.txt
python train_zoobot_lens_classifier.py --dataset synthetic_lenses.csv --epochs 3
```

The training script makes a stratified 80/20 train/validation split and trains only the classifier head on the pretrained encoder. It prints the best validation checkpoint path. This is **not** an observational benchmark: both the training set and DSPL evaluation images are simulations, so the scores only demonstrate the pipeline on synthetic data.

Use the checkpoint path printed by training for inference, for example:

```powershell
python run_zoobot_inference.py --checkpoint model_checkpoints/ruby_synthetic_lens_head/checkpoints/0.ckpt
```

The inference script writes `dspl_experiment/zoobot_scores.csv` only after model predictions are available. The training and inference transforms both produce grayscale, 224-pixel inputs scaled to [0, 1]. If you have an existing Ruby-trained classifier checkpoint, use it instead of running the synthetic training demo.

## Analyze scores

```powershell
python analyze_dspl_results.py
```

With `zoobot_scores.csv` present, this writes a per-condition summary CSV and a three-panel plot. It reports sample count, mean/median lens probability, sample standard deviation, label-1 detection rate using the notebook's argmax rule, and false-positive rate for negatives. It compares each condition with that experiment's control. Detection rate is not called overall accuracy.

## Pilot results

| Model-run measure | Value | Interpretation |
| --- | ---: | --- |
| Synthetic training / validation split | 1,600 / 400 images | Stratified split of the existing Ruby synthetic dataset. |
| Training duration | 1 epoch, 50 steps, CPU | Single-epoch pipeline demonstration; CUDA was unavailable. |
| Train accuracy / loss | 83.19% / 0.4491 | End-of-epoch metrics on training images. |
| Validation accuracy / loss | 90.75% / 0.2925 | End-of-epoch metrics on the held-out synthetic validation split. |
| Trainable classifier head / frozen encoder | 1.3 thousand / 14.9 million parameters | Only the small classifier head was trained; the ConvNeXt encoder was pretrained. |
| DSPL evaluation set | 65 positives + 5 controls = 70 | All images are synthetic; each positive is included in its one-factor sweep. |
| Mean lens score, positives / controls | 0.662 / 0.198 | Average class-1 softmax output by label. |
| Positive detections | 59/65 = 90.8% | Positive detection rate (recall) using the model's argmax class decision. |
| Control false positives | 0/5 = 0% | False-positive rate in this small synthetic control sample. |

The complete per-condition table follows. Score standard deviation (SD) is the sample SD across five images. Baseline changes compare magnitude offset 0, radius scale 1.00, or foreground-light scale 1.00, as appropriate.

| Sweep | Setting | n | Mean score +/- SD | Median | Detected as lens | Mean-score / detection-rate change vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Faintness offset (mag) | 0 | 5 | 0.687 +/- 0.041 | 0.669 | 5/5 (100%) | 0.000 / 0 pp |
| Faintness offset (mag) | 1 | 5 | 0.678 +/- 0.047 | 0.669 | 5/5 (100%) | -0.009 / 0 pp |
| Faintness offset (mag) | 2 | 5 | 0.631 +/- 0.051 | 0.613 | 5/5 (100%) | -0.056 / 0 pp |
| Faintness offset (mag) | 3 | 5 | 0.672 +/- 0.057 | 0.683 | 5/5 (100%) | -0.014 / 0 pp |
| Arc-radius scale | 0.70 | 5 | 0.373 +/- 0.070 | 0.376 | 0/5 (0%) | -0.314 / -100 pp |
| Arc-radius scale | 0.85 | 5 | 0.545 +/- 0.050 | 0.549 | 4/5 (80%) | -0.141 / -20 pp |
| Arc-radius scale | 1.00 | 5 | 0.687 +/- 0.041 | 0.669 | 5/5 (100%) | 0.000 / 0 pp |
| Arc-radius scale | 1.15 | 5 | 0.789 +/- 0.036 | 0.781 | 5/5 (100%) | +0.103 / 0 pp |
| Arc-radius scale | 1.30 | 5 | 0.851 +/- 0.024 | 0.846 | 5/5 (100%) | +0.164 / 0 pp |
| Foreground-light scale | 0.5 | 5 | 0.698 +/- 0.039 | 0.684 | 5/5 (100%) | +0.012 / 0 pp |
| Foreground-light scale | 1.0 | 5 | 0.687 +/- 0.041 | 0.669 | 5/5 (100%) | 0.000 / 0 pp |
| Foreground-light scale | 2.0 | 5 | 0.673 +/- 0.044 | 0.654 | 5/5 (100%) | -0.013 / 0 pp |
| Foreground-light scale | 4.0 | 5 | 0.630 +/- 0.047 | 0.611 | 5/5 (100%) | -0.057 / 0 pp |
| Negative controls | label 0 | 5 | 0.198 +/- 0.007 | 0.198 | 0/5 flagged (FPR 0%) | Not applicable |

Scores are Zoobot class-1 softmax outputs, not guaranteed calibrated probabilities. A detection uses argmax (the larger class probability). `pp` means percentage points. The faintness means are not monotonic in this small sample. The large score/rate change for the smallest arcs is the clearest candidate sensitivity gap; five images per setting are not enough to claim a stable population rate.

Faintness scores are not monotonic in this five-image sample. Full means, medians, sample standard deviations, and per-condition detection rates are in `dspl_experiment/summary_by_condition.csv`; the plot is `dspl_experiment/performance_by_parameter.png`.

## Lens-category interpretation

The additional `dspl_experiment/detection_by_lens_category.csv` groups positive examples into practical sensitivity bins. These are categories of the simulated properties, not distinct astrophysical morphologies. The rate is positive detection rate (recall/sensitivity), not overall accuracy; Wilson 95% intervals show how uncertain each small-sample rate is. The negative-control row reports false-positive rate separately.

| Simulated category | Detected | Detection rate (95% interval) | Mean lens score | Project takeaway |
| --- | ---: | ---: | ---: | --- |
| Bright/moderate sources, offset 0-1 mag | 10/10 | 100% (72-100%) | 0.682 | Reference brightness bin; verify with more examples. |
| Faint sources, offset 2-3 mag | 10/10 | 100% (72-100%) | 0.652 | Mean score is 0.030 lower than the brighter bin; retain faint cases in training and retest. |
| Small arcs, radius scale 0.70-0.85 | 4/10 | 40% (17-69%) | 0.459 | Current sensitivity gap; prioritize small-arc examples in follow-up training and evaluation. |
| Baseline arcs, radius scale 1.00 | 5/5 | 100% (57-100%) | 0.687 | Reference arc-size bin; five examples are not enough to establish completeness. |
| Large arcs, radius scale 1.15-1.30 | 10/10 | 100% (72-100%) | 0.820 | No misses in this pilot; verify on larger and independent samples. |
| Low foreground light, flux scale 0.5 | 5/5 | 100% (57-100%) | 0.698 | No misses in five examples; compare against the nominal-light bin. |
| Nominal foreground light, flux scale 1.0 | 5/5 | 100% (57-100%) | 0.687 | Reference blending bin. |
| High foreground light, flux scale 2-4 | 10/10 | 100% (72-100%) | 0.652 | Mean score is 0.035 below nominal; retain strongly blended cases in evaluation. |
| Negative controls | 0/5 flagged | 0% (0-43%) false-positive rate | 0.198 | Too few controls to estimate real-survey contamination. |

The categories within each feature are compared only within that feature's sweep. All rows are synthetic, and the confidence intervals are broad because each bin contains only 5 or 10 images.

## Reproducibility and limits

The generator uses deterministic seeds. Outputs and checkpoints are excluded from Git via `.gitignore`; scripts and manifest schemas remain versionable. Training, validation, and DSPL evaluation images are all synthetic. The reported classifier scores are a pipeline demonstration, not a measurement of real-survey performance; the small pilot and five negative controls do not support broad generalization claims.