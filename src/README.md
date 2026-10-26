# src/ -- file guide

Every script here, grouped by what it does. Run everything from inside `src/`
(e.g. `python dataset.py`), with the project's `.venv` activated.

## Simulation

| File | What it does |
|---|---|
| `lens_sim.py` | The lens simulator itself (SIE + Sersic, via lenstronomy). `LensSimulator.sample_lens()` / `.sample_nonlens()` draw a random lens/non-lens image from a config. `.sample_lens_with_ring_size(rng, theta_e)` fixes the ring size (Einstein radius) to a chosen value instead of deriving it from physics -- added for the ring-size sweep experiments on this branch. Also has the fixed (non-per-image) intensity scaling used everywhere. |
| `dataset.py` | Generates and saves a full labeled lens/non-lens `.npz` dataset from a config file (e.g. `configs/sim_config_large.yaml`). |
| `export_images.py` | Dumps a saved dataset's images as individual grayscale PNGs, for browsing. |
| `prepare_real_euclid.py` | Turns the real Euclid Q1 FITS cutouts (`data/real_euclid/`) into the same `.npz` format as the simulated data, so every model can be evaluated on real images the same way. |

## Model definitions

| File | What it does |
|---|---|
| `models.py` | `SimpleLensCNN` -- a small CNN trained from scratch. |
| `vit_model.py` | A small Vision Transformer (timm), also trained from scratch. |
| `zoobot_classifier.py` | `ZoobotLensClassifier` -- wraps a **frozen** pretrained Zoobot encoder (see `ENCODER_NAME` at the top of the file -- currently the Euclid-specific checkpoint) with a small trainable linear head on top. This is the "linear probe" approach, not fine-tuning. |

## Training

| File | What it does |
|---|---|
| `train_cnn.py` | Trains `SimpleLensCNN`. |
| `train_vit.py` | Trains the ViT. |
| `train_zoobot.py` | Trains the Zoobot linear head (works with whatever encoder `zoobot_classifier.ENCODER_NAME` currently points to). |
| `train_zoobot_euclid.py` | Thin wrapper around `train_zoobot.py` that trains specifically against the Euclid encoder and saves to `outputs/zoobot_euclid_lens_classifier.pt`, kept separate from the older general-purpose-encoder checkpoint so the two can still be compared. |

## Whole-image interpretability (DeepDream)

| File | What it does |
|---|---|
| `deepdream.py` | The model-agnostic DeepDream engine: gradient ascent on a chosen layer's activation, with optional blur + total-variation regularization (needed to stop results collapsing into noise). |
| `deepdream_cnn.py` / `deepdream_vit.py` / `deepdream_zoobot.py` | Runs DeepDream on each of the three trained models, for the "what does this whole model consider lens-like" question. |

## Filter-level interpretability (this branch's new work)

| File | What it does |
|---|---|
| `filter_firing.py` | The main ring-size experiment: generates a small-radius (0.70"-1.00") and a baseline (1.60"-2.40") image set using `lens_sim.sample_lens_with_ring_size`, runs both through the Zoobot encoder, and at each of its 4 stages compares which filters are most active (top 10%) via Jaccard overlap. Saves `outputs/filter_firing_report.json`. |
| `visualize_filters.py` | Attempted to name the divergent filters by running per-filter DeepDream (gradient ascent targeting one channel at a time) from noise. **This didn't work** -- every filter converged to indistinguishable noise. Kept as a documented negative result; see `outputs/filter_portraits/`. |
| `top_activating_images.py` | The method that actually worked: instead of synthesizing an image, finds which *real* images (from the full simulated pool + the 23 real Euclid images) activate each divergent filter most strongly. This is what the filter names in the writeup are based on. Saves `outputs/top_activating_images.json`. |

## Evaluation

| File | What it does |
|---|---|
| `evaluate_all.py` | Scores all 3 models (CNN, ViT, Zoobot) on every test set we have -- both simulated batches, the held-out simulated split, and the real Euclid images. Saves `outputs/evaluation_summary.txt`. |

## Preview / plotting utilities

All of these just load saved results and render a comparison grid PNG -- no new computation.

| File | What it renders |
|---|---|
| `preview.py` | Grid of simulated lens vs. non-lens examples. |
| `preview_real_euclid.py` | Grid of the real Euclid Q1 cutouts. |
| `preview_deepdream_compare.py` | Side-by-side DeepDream comparison across the 3 models. |
| `preview_deepdream_vit.py` | ViT-specific DeepDream grid. |
| `preview_filter_portraits.py` | Grid of the (failed) per-filter DeepDream portraits from `visualize_filters.py`. |
| `preview_top_activating.py` | Grid of each divergent filter's top-5 real activating images from `top_activating_images.py` -- this is the one actually worth looking at for the filter names. |
