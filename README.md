# NeSyFOLD on Zoobot: which filters make the lens decision?

DeepDream on Zoobot was inconclusive, so this folder tries NeSyFOLD
(Padalkar, Wang & Gupta, AAAI 2024; [arXiv:2301.12667](https://arxiv.org/abs/2301.12667)).
It replaces Zoobot's final layer with readable if-then rules over its last-layer filters,
then tests what those filters respond to.

The model analysed here is a Zoobot fine-tuned on real Euclid Q1 lenses vs. non-lenses
(`runs/zoobot_lr0.0001_seed1`). The scripts expect the real-image dataset (`data/split.csv`,
`data/cutouts/`) and those model weights to sit next to this folder. The dataset and training
code are on branch `vit-vs-zoobot` (`make_cutouts.py`, `train.py`). The weights (`best.pt`) are
too large for GitHub; re-create them with `python train.py --model zoobot --lr 1e-4 --seed 1`.

## Scripts (run from the repo root, in order)

| Script | What it does |
|---|---|
| `extract.py` | Runs every image through the trained Zoobot. Each of the 640 filters in the last layer becomes ON/OFF per image (ON = stronger than 0.6 × mean + 0.7 × std over training images, as in the paper). |
| `train_rules.py` | Learns FOLD-SE rules over those ON/OFF filters and tests them. `fit()` sends an encoded, encrypted copy of the training table to the FOLD-SE server; everything else runs locally. Needs `pip install git+https://github.com/hwd404/foldse_client.git`. |
| `name_filters.py` | For each filter in the rules: its 10 strongest training images, with a heat map of where it fires. |
| `probe_filters.py` | Paints rings, arcs, a second ring, and lookalike controls onto 200 real non-lens galaxies, and records which filters switch on. |

`extract.py` writes large tables (`strengths.npz`, `*_bin.csv`) that are not committed. Re-run it to regenerate them.

## Results so far (`out/zoobot_lr0.0001_seed1_norm/`)

**Rules.** 3 to 6 rules using 4 to 13 of the 640 filters (`rules_*.txt`). The rules agree with Zoobot on 93–98% of the 379 test images. Zoobot itself gets 96.0% test accuracy at a cutoff chosen on validation; the rules get 92.6–95.0%. Filter k445 appears in every rule set.

**What the filters detect** (`probes/`, labels in `naming.csv`). Painting a ring onto a non-lens makes Zoobot call it a lens 92% of the time. A companion galaxy or a brighter centre: 0%. The rule filters behave as ring/arc detectors, not as "bright centre" or "something nearby" detectors.

**Compound lenses.**
- All 4 real compound lenses follow the same rule as ordinary lenses.
- k339 fires for one ring but mostly switches off when a second ring is added.
- k216 and k459 fire for two rings (92–94%), but rarely for one ring, an outer ring alone, or one ring carrying the same total light (10–35%). They are not in any rule set, because lens vs. non-lens never needed them.
- On the real compound lenses, only the Cosmic Ammonite switches most two-ring filters on, and k339 is on for all four.

**Limits.** The painted rings are simple Gaussian shapes, not physical lens models. This is one trained model and only 4 real compound lenses, so treat it as evidence, not proof.
