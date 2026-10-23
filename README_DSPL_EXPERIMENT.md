# DSPL-Like Sensitivity Pilot

This pilot generates 100x100 grayscale images with one foreground SIE lens and two background Sérsic sources at different redshifts. It changes one factor at a time: source faintness, lens Einstein-radius scale, or foreground lens-light flux. Negative controls retain the simulated source and foreground light but reduce the Einstein radius to 0.01 arcsec; they are weak-lens controls, not a representative population of real non-lens galaxies.

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
| Einstein-radius scale | 0.50, 0.55, 0.70, 0.85, 1.00, 1.15, 1.30 | Multiplies the 1.2 arcsec baseline, giving 0.60, 0.66, 0.84, 1.02, 1.20, 1.38, and 1.56 arcsec. |
| Foreground-light flux scale | 0.5, 1, 2, 4 | Multiplies lens-galaxy flux; corresponding lens magnitudes are 21.503, 20.750, 19.997, and 19.245. This varies foreground brightness only; it does not vary source-lens separation, neighboring objects, or overlap geometry. |
| Negative controls | theta_E = 0.01 arcsec, label 0 | Same general source/lens-light setup with negligible lensing. These test rejection of this specific weak-lens case, not broad real-world non-lens contamination. |
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

To create the expanded follow-up without replacing the original 5-replicate pilot, generate 30 examples per positive condition and 100 controls in a separate directory:

```powershell
python generate_dspl_experiment.py --replicates 30 --negative-controls 100 --output dspl_experiment/expanded_30rep/simulation_manifest.csv --image-dir dspl_experiment/images
python run_zoobot_inference.py --manifest dspl_experiment/expanded_30rep/simulation_manifest.csv --checkpoint model_checkpoints/ruby_synthetic_lens_head/checkpoints/0.ckpt --output dspl_experiment/expanded_30rep/zoobot_scores.csv
python analyze_dspl_results.py --scores dspl_experiment/expanded_30rep/zoobot_scores.csv --output-dir dspl_experiment/expanded_30rep
```

This design writes 450 positive condition images (15 conditions x 30 paired replicates) plus 100 negative controls, 550 condition rows total. PNGs are stored with the original DSPL pilot under `dspl_experiment/images/{faintness,arc_size,blending,negative_control}`. The first five replicates reuse byte-identical pilot images; the remaining replicates are added alongside them. The same 30 baseline scenes are shared across the three sweeps, so the expanded run has 490 unique image contents (390 positive, 100 controls). The original 70-image pilot manifest and scores remain unchanged.

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

## Expanded 30-replicate follow-up

The follow-up used the same one-epoch checkpoint and the same argmax class decision. Each condition has 30 paired replicates; there are 450 positive condition rows and 100 negative-control rows. The three baseline settings share the same 30 scenes across sweeps, so row-level totals weight those baseline scenes three times. Unique-image metrics deduplicate identical rendered PNGs.

### How to read the units and metrics

| Field or term | Units / definition | How to interpret it |
| --- | --- | --- |
| `lens_score`, mean, median, SD | Unitless score on a 0-1 scale | Zoobot's class-1 softmax output, where class 1 means lens. It is not guaranteed to be calibrated as a probability. SD is the sample standard deviation of scores within a condition. |
| Detection / recall | Percent of positive examples classified as class 1 | Uses argmax: the image is called a lens when class 1 has the larger score. It is not a separately tuned threshold. |
| False-positive rate (FPR) | Percent of label-0 controls classified as class 1 | Reported separately from lens detection rate. |
| False negative (FN) | Count of label-1 lenses classified as class 0 | In the expanded run there are 87 unique FNs; all are from the smallest arc conditions (scales 0.50, 0.55, 0.70, and 0.85). |
| Magnitude offset | Magnitudes (`mag`) added to both sources | Higher magnitude means fainter light; +1 mag is about 2.512 times less flux. |
| Einstein-radius scale | Unitless multiplier of the 1.20 arcsec baseline | For example, 0.70 means $0.70 \times 1.20 = 0.84$ arcsec, not 0.70 arcsec. |
| Foreground-light flux scale | Unitless multiple of baseline lens-galaxy flux | For example, 4 means four times the baseline foreground light. |
| Lensed source flux | Lenstronomy model flux units | Sum of the modeled lensed source light, not calibrated detector counts or magnitudes. |
| `n` / condition rows | Number of manifest entries at a setting | There are 30 entries per positive condition. Identical baseline scenes are intentionally shared across sweeps. |
| Unique images | Distinct rendered image contents | The 550 condition rows correspond to 490 unique image contents because the three sweep baselines reuse the same 30 scenes. |

The 87 false negatives are 30/30 at scale 0.50 (0.60 arcsec), 30/30 at scale 0.55 (0.66 arcsec), 23/30 at scale 0.70 (0.84 arcsec), and 4/30 at scale 0.85 (1.02 arcsec). Row-based recall (363/450 = 80.67%) counts shared baseline scenes in each sweep; deduplicated recall (303/390 = 77.69%) counts each rendered image once. Both are synthetic-data summaries, not estimates from observed lenses.

| Measure | Across condition rows | Deduplicated unique images | Meaning |
| --- | ---: | ---: | --- |
| Positive / control examples | 450 / 100 | 390 / 100 | There are 550 manifest rows and 490 unique image contents. |
| True positive / false negative | 363 / 87 | 303 / 87 | Positive images classified as lens / missed. |
| True negative / false positive | 100 / 0 | 100 / 0 | Controls classified correctly / incorrectly flagged as lenses. |
| Positive detection rate (recall) | 80.67% | 77.69% | Deduplicated estimate counts each rendered image once. |
| Negative-control false-positive rate | 0/100 = 0% | 0/100 = 0% | Wilson 95% interval is 0-3.7%; zero observed is not proof of zero real-world contamination. |
| Overall accuracy | 84.18% | 82.24% | Correct classifications divided by rows or unique images; synthetic data only. |
| Mean lens score, positives / controls | 0.6199 / 0.1993 | See category means below | Class-1 softmax output; not guaranteed calibrated. |

Per-condition results (mean +/- sample SD and median lens score). `FN` is the number of label-1 lenses classified as non-lenses; it is blank for controls because that row reports false positives instead.

| Sweep | Setting | n | Mean +/- SD | Median | Detected as lens | FN | Change vs. baseline: mean / detection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Faintness offset (mag) | 0 | 30 | 0.697 +/- 0.051 | 0.696 | 30/30 (100%) | 0 | 0.000 / 0 pp |
| Faintness offset (mag) | 1 | 30 | 0.692 +/- 0.051 | 0.684 | 30/30 (100%) | 0 | -0.006 / 0 pp |
| Faintness offset (mag) | 2 | 30 | 0.636 +/- 0.058 | 0.634 | 30/30 (100%) | 0 | -0.062 / 0 pp |
| Faintness offset (mag) | 3 | 30 | 0.659 +/- 0.065 | 0.661 | 30/30 (100%) | 0 | -0.039 / 0 pp |
| Arc-radius scale | 0.50 | 30 | 0.259 +/- 0.091 | 0.243 | 0/30 (0%) | 30 | -0.438 / -100 pp |
| Arc-radius scale | 0.55 | 30 | 0.291 +/- 0.093 | 0.280 | 0/30 (0%) | 30 | -0.406 / -100 pp |
| Arc-radius scale | 0.70 | 30 | 0.429 +/- 0.085 | 0.437 | 7/30 (23.3%) | 23 | -0.268 / -76.7 pp |
| Arc-radius scale | 0.85 | 30 | 0.581 +/- 0.059 | 0.586 | 26/30 (86.7%) | 4 | -0.116 / -13.3 pp |
| Arc-radius scale | 1.00 | 30 | 0.697 +/- 0.051 | 0.696 | 30/30 (100%) | 0 | 0.000 / 0 pp |
| Arc-radius scale | 1.15 | 30 | 0.787 +/- 0.041 | 0.785 | 30/30 (100%) | 0 | +0.090 / 0 pp |
| Arc-radius scale | 1.30 | 30 | 0.841 +/- 0.031 | 0.850 | 30/30 (100%) | 0 | +0.144 / 0 pp |
| Foreground-light scale | 0.5 | 30 | 0.707 +/- 0.048 | 0.706 | 30/30 (100%) | 0 | +0.009 / 0 pp |
| Foreground-light scale | 1.0 | 30 | 0.697 +/- 0.051 | 0.696 | 30/30 (100%) | 0 | 0.000 / 0 pp |
| Foreground-light scale | 2.0 | 30 | 0.684 +/- 0.052 | 0.682 | 30/30 (100%) | 0 | -0.013 / 0 pp |
| Foreground-light scale | 4.0 | 30 | 0.640 +/- 0.058 | 0.629 | 30/30 (100%) | 0 | -0.057 / 0 pp |
| Negative controls | label 0 | 100 | 0.199 +/- 0.020 | 0.198 | 0/100 flagged | - | FPR 0% |

**Main interpretation:** detection is complete for the tested faintness and foreground-light settings, but not for small simulated arcs. All 60 very-small-arc examples (0.50-0.55 scale; 0.60-0.66 arcsec) were missed, followed by 23/30 misses at 0.70 scale (0.84 arcsec) and 4/30 at 0.85 scale (1.02 arcsec). Baseline and larger arcs (1.20-1.56 arcsec) were all detected in this synthetic sample.

Grouped sensitivity categories use Wilson 95% intervals. The categories are property bins within separate sweeps, not unique physical lens morphologies; baseline scenes are paired across the sweeps.

| Simulated category | Detected | Rate (95% interval) | Mean lens score | Project interpretation |
| --- | ---: | ---: | ---: | --- |
| Bright/moderate sources, offsets 0-1 mag | 60/60 | 100% (94.0-100%) | 0.695 | No misses in this bin; the interval still allows a nonzero miss rate. |
| Faint sources, offsets 2-3 mag | 60/60 | 100% (94.0-100%) | 0.647 | Mean score is 0.047 below the brighter bin; retain faint cases and retest. |
| Very small arcs, radius scales 0.50-0.55 | 0/60 | 0% (0-6.0%) | 0.275 | The model missed every very small arc in this synthetic test; strongest sensitivity gap. |
| Small arcs, radius scales 0.70-0.85 | 33/60 | 55.0% (42.5-66.9%) | 0.505 | Persistent sensitivity gap; prioritize small arcs in training/evaluation coverage. |
| Baseline arcs, radius scale 1.00 | 30/30 | 100% (88.6-100%) | 0.697 | Reference bin; no misses in 30 paired examples. |
| Large arcs, radius scales 1.15-1.30 | 60/60 | 100% (94.0-100%) | 0.814 | Strong detection in these simulations; validate on independent data. |
| Low foreground light, scale 0.5 | 30/30 | 100% (88.6-100%) | 0.707 | No misses in this bin. |
| Nominal foreground light, scale 1.0 | 30/30 | 100% (88.6-100%) | 0.697 | Reference blending bin. |
| High foreground light, scales 2-4 | 60/60 | 100% (94.0-100%) | 0.662 | Mean score is 0.035 below nominal; detection remained complete in this sample. |
| Negative controls | 0/100 flagged | 0% FPR (0-3.7%) | 0.199 | Better constrained than the 5-control pilot, but these are weak-lens controls, not a diverse non-lens galaxy sample. |

Across faintness offsets, all 120 examples were detected; mean scores dipped at +2 mag and rose slightly at +3, so the score trend is not monotonic. Across blending settings, all 120 examples were detected while the mean score declined from 0.697 at nominal foreground light to 0.640 at scale 4. Arc size is the dominant weakness: detection was 0/30 at scales 0.50 and 0.55, 7/30 at scale 0.70, and 26/30 at scale 0.85, compared with 30/30 at baseline and larger scales.

The expanded faintness-sweep mean lensed-flux contributions were:

| Magnitude offset | Source 1 mean lensed flux | Source 2 mean lensed flux |
| ---: | ---: | ---: |
| 0 | 1716.8 | 579.5 |
| 1 | 683.5 | 230.7 |
| 2 | 272.1 | 91.8 |
| 3 | 108.3 | 36.6 |

The expanded artifacts are `dspl_experiment/expanded_30rep/simulation_manifest.csv`, `zoobot_scores.csv`, `summary_by_condition.csv`, `detection_by_lens_category.csv`, and `performance_by_parameter.png`.

**Scope of the blending result:** the blending sweep increases only the brightness of the central foreground lens galaxy. It does not test spatial offsets, crowded fields, nearby contaminants, or different source-lens overlap geometries. Therefore the observed small score decline at high foreground flux should not be generalized to all forms of astronomical blending.

## Reproducibility and limits

The generator uses deterministic seeds. The original 5-replicate pilot and expanded 30-replicate follow-up are separate. Training, validation, and DSPL evaluation images are all synthetic. These scores are a pipeline demonstration, not a measurement of real-survey performance; larger synthetic samples reduce sampling uncertainty but do not resolve the simulation-to-observation gap.