# Gravitational Lens Classifier: DeepDream Testing

Pipeline for simulating single (galaxy-galaxy) strong gravitational lenses, training and
comparing three classifier architectures (a custom CNN, a custom Vision Transformer, and a
model built on the pretrained [Zoobot](https://github.com/mwalmsley/zoobot) galaxy-morphology
encoder), visualizing what each has learned via DeepDream, and evaluating all three on both
simulated data and real Euclid Q1 imagery.

Builds on the approach in Wilde et al. 2022, *"Detecting gravitational lenses using machine
learning: exploring interpretability and sensitivity to rare lensing configurations"*
(MNRAS 512, 3464, [doi:10.1093/mnras/stac562](https://doi.org/10.1093/mnras/stac562)).

## Headline result

All three models score 94-98% accuracy on held-out *simulated* test data but collapse to
48-61% on 23 real Euclid Q1 images -- a sim-to-real generalization gap, not model choice,
is the main bottleneck. See `outputs/evaluation_summary.txt` and
`outputs/deepdream_compare_grid.png` for the full picture, including a DeepDream finding that
the CNN converges to the same checkerboard texture regardless of input, suggestive of a
simulation-specific shortcut rather than true arc/ring morphology.

## Filter-level interpretability (this branch)

This branch (`Filter-Analysis`) swaps the Zoobot classifier to the **Euclid-specific
pretrained encoder** (`mwalmsley/zoobot-encoder-euclid`, verified ConvNeXt-Nano, 15.0M
params) -- this is now the baseline going forward, not the general-purpose one. On top of
it, we built filter-level (not just whole-image) interpretability: which of the encoder's
individual channels respond to small-radius arcs (0.70"-1.00") vs. clearly larger ones, at
each of its 4 processing depths ("stages").

**Finding:** overlap between the two configs' top-activated filters is near-total at the
shallowest stage (Jaccard 1.0 -- generic edge detectors don't care about ring size) and
drops sharply at the deepest stage (Jaccard 0.64) -- ring-size sensitivity lives almost
entirely in the deepest, most semantic layer. Naming the divergent deep-layer filters (via
which real images activate them most, since per-filter DeepDream collapsed into
indistinguishable noise -- see `outputs/filter_portraits/` for that negative result) showed
the small-radius-specific filters are noisier, less clean ring detectors than the
large-radius ones, and several aren't ring detectors at all (generic brightness/shape
filters) -- a plausible mechanistic reason small rings are harder to detect: the network
falls back on weaker, non-ring-specific cues. Full writeup: `outputs/filter_firing_report.json`
and `outputs/top_activating_images.json`; see `outputs/README.md` for what every file is.

## Lens injection: a real fix, with a real caveat

Painting simulated arcs onto REAL non-lens Euclid backgrounds (HOLISMOKES-style) instead of
training on pure simulation. Headline: recall went from 0.42 to 0.92 but precision collapsed
from 1.00 to 0.69, diagnosed (not assumed) as a shortcut -- the model partly learned "real
image texture = lens" rather than "arc = lens" -- and the fix attempt showed that shortcut
was substantially *where the recall gain came from* in the first place.
**Full writeup, including how the diagnosis was done: [`LENS_INJECTION.md`](LENS_INJECTION.md).**

## Compound lens detection: a classical (non-ML) method, validated both ways

Given a verified single lens, does its radial brightness profile show evidence of a second
ring? No training data needed: azimuthally-averaged flux vs. radius, peak-finding on the
residual after subtracting the smooth galaxy-light decline, then an angular-shape check
(a genuine ring spreads flux across a wide angle range; a companion galaxy or artifact
concentrates in one narrow wedge) to tell a real second ring apart from an unrelated nearby
object. Validated both for false positives (0/12 on known-single lenses) AND for whether it
actually detects a real second ring when one exists (a quantified sensitivity sweep, 7-80%
depending on how separated the rings are -- weakest exactly where rings visually blend,
matching what the team already expected).
**Full writeup, function by function: [`COMPOUND_LENS_DETECTION.md`](COMPOUND_LENS_DETECTION.md).**

## Real-world test: ranking 79 real Euclid lenses for compound likelihood

The full pipeline above, run end-to-end on an independent 79-image real
Euclid test set (75 ordinary lenses + 4 unconfirmed compound candidates),
producing a 1-79 ranked score for human-review triage rather than a hard
yes/no. This run also surfaced and fixed a real bug in the ring-counter
(a `scipy.find_peaks` prominence-measurement quirk that was inflating noise
ripples into false "second rings"), and a related scoring bug (raw peak
count outranking true peak quality). 2 of the 4 known candidates land at
rank 1 and 15 of 79; the other 2 show only a single strong ring with no
second peak at any strength -- read as a real, stated result, not spun.
**Full writeup: [`REAL_WORLD_TEST.md`](REAL_WORLD_TEST.md).**

AUCROC and FPR @ 80% recall against a separate, independent 3000-image
simulated test set (for cross-team model comparison), plus a concrete
diagnosis of why those numbers are weak (a real mismatch between this
method's 0.9" minimum resolvable ring separation and real compound lenses'
typical 0.17"-0.86" separations): **[`MODEL_EVALUATION.md`](MODEL_EVALUATION.md).**

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # or a CUDA index on Linux/NVIDIA
pip install -r requirements.txt
```

## Repo layout

See `src/README.md` for a full description of every script (grouped by purpose) and
`outputs/README.md` for what every generated artifact is -- short version:

```
configs/                  YAML configs for the lenstronomy simulator (physics, image, dataset, training params)
src/                       All code -- see src/README.md for the full file-by-file guide
data/                       Datasets (.npz) -- large ones gitignored, regenerate via dataset.py
outputs/                    Trained checkpoints, comparison grids, reports -- see outputs/README.md
```

## Reproducing

```bash
cd src
python dataset.py                                   # generates data/lens_dataset.npz from configs/sim_config.yaml
python -c "from dataset import load_config, generate_dataset; \
  generate_dataset(load_config('../configs/sim_config_large.yaml'), out_path='../data/lens_dataset_large.npz')"
python train_cnn.py      # or: from dataset import load_config; from train_cnn import train; ...
python train_vit.py
python train_zoobot.py
python deepdream_cnn.py && python deepdream_vit.py && python deepdream_zoobot.py
python evaluate_all.py

# Filter-level interpretability (this branch):
python train_zoobot_euclid.py       # trains the lens head on the Euclid-specific encoder
python filter_firing.py             # ring-size sweep + per-stage filter overlap comparison
python visualize_filters.py         # per-filter DeepDream (negative result -- see outputs/README.md)
python top_activating_images.py     # the method that actually worked: real images per filter
python preview_filter_portraits.py && python preview_top_activating.py   # build the review grids

# Lens injection:
python lens_injection.py            # builds the v2 (injection + plain-negative) training set
python train_zoobot_injected_v2.py
python evaluate_lens_injection.py   # before/after on the clean held-out real set

# Compound lens detection (no training needed):
python preview_radial_profiles.py       # specificity check + the main result grids
python double_ring_sensitivity.py       # sensitivity sweep (takes a few minutes)

# Real-world test (79 real Euclid lenses, ranked):
python validate_on_real_compound.py     # 3-system archival-HST companion check
python score_real_test_set.py           # the full 79-image ranked scoring
```

All simulated images use one **fixed, dataset-wide intensity scale** (`scale_low`/`scale_high`
in the config, not a per-image min/max stretch) so pixel values stay comparable across images
and batches -- important for training a model without per-image contrast bias.

## Real test data

`data/real_euclid_dataset.npz` and `data/real_euclid/metadata.csv` hold 23 real Euclid Q1
cutouts (12 lens candidates incl. the confirmed lens NGC6505, 11 non-lenses), sourced from the
public **Euclid Quick Release 1 Strong Lensing Discovery Engine** catalog (Euclid Collaboration:
Walmsley et al. 2025, [arXiv:2503.15326](https://arxiv.org/abs/2503.15326);
[catalog + cutouts on Zenodo](https://zenodo.org/records/15025832)), pulled via IRSA's public
Euclid Q1 mirror (no login required).

## Next steps

See the full findings/recommendations report for details. In short: acquire more real labeled
data (we used 23 of the ~497 public candidates -- this is now also the direct blocker on
lens injection, not just general accuracy), make the simulator more realistic (real
backgrounds, PSF variation, detector artifacts), fine-tune more of the Zoobot encoder once more
real data is available, and investigate the CNN's checkerboard-texture shortcut.

For compound lens detection specifically: the close-separation regime (<1.5x) is the clear
weak spot and matches a known hard case, so improving sensitivity there (vs. accepting it as
a physical blending limit) would need a different approach, not just more threshold tuning.
Also worth real validation against the 4 known real compound lenses named in Wilde et al.
2022 (archival HST/HSC imaging) rather than only the synthetic two-arc proxy used here.
