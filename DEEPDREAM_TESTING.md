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

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # or a CUDA index on Linux/NVIDIA
pip install -r requirements.txt
```

## Repo layout

```
configs/                  YAML configs for the lenstronomy simulator (physics, image, dataset, training params)
src/
  lens_sim.py              SIE + Sersic single-lens simulator (lenstronomy), fixed dataset-wide intensity scaling
  dataset.py                Generates/saves a lens vs. non-lens .npz dataset from a config
  export_images.py          Dumps a dataset's images as individual grayscale PNGs
  preview.py / preview_*.py Comparison grids (lens vs non-lens, real vs sim, DeepDream outputs)
  prepare_real_euclid.py    Preprocesses real Euclid Q1 FITS cutouts into the same dataset format
  models.py                 SimpleLensCNN (small CNN classifier)
  vit_model.py               Small ViT classifier (timm, trained from scratch)
  zoobot_classifier.py       Frozen pretrained Zoobot encoder + trainable linear head
  train_cnn.py / train_vit.py / train_zoobot.py   Training loops for each model
  deepdream.py               Model-agnostic DeepDream engine (octave-based gradient ascent, optional
                              blur/total-variation regularization)
  deepdream_cnn.py / deepdream_vit.py / deepdream_zoobot.py   Per-model DeepDream runs
  evaluate_all.py            Scores all 3 models on every test set (simulated + real)
data/                       Datasets (.npz) -- large ones gitignored, regenerate via dataset.py
outputs/                    Trained checkpoints, comparison grids, evaluation_summary.txt
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
data (we used 23 of the ~497 public candidates), make the simulator more realistic (real
backgrounds, PSF variation, detector artifacts), fine-tune more of the Zoobot encoder once more
real data is available, and investigate the CNN's checkerboard-texture shortcut.
