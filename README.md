# Ruby: Zoobot trained on realistic simulated compound lenses

This branch asks how faint and how close a second ring can be before a lens finder stops recognizing it, and whether training on simulated compound lenses expands that limit without hurting ordinary lens finding.

All numbers, caveats and run commands are in **[sim/RESULTS.md](sim/RESULTS.md)**.

## What we did

1. **Built a realistic simulator** (`sim/simulate.py`): two source distances, Euclid's published lens models, measured blur, lumpy sources and real host galaxies. [How it works](sim/SIMULATION.md).
2. **Retrained Zoobot with a third answer** (`sim/train_models.py`): not a lens, lens or compound. The 4 real compound lenses were never used for training.
3. **Tested once on held-out real Euclid Q1 images** (`sim/final_test.py`), against 5 pass marks set before training.
4. **Ran follow-up tests**, including a two-step setup, a bigger test on 5,000 new non-lenses and a close-ring test.

## Main results

- **Verdict:** no model passed all 5 pass marks. The 3-answer Zoobot passed 2, so the planned result is the detection map and the gap between simulated and real lenses.
- **Real compound lenses:** the 4 known ones average rank 3.9 of 79 real lenses. Guessing gives 40, and painted-ring training gave 28–56. This rests on only 4 lenses, all obvious, and earlier models had already scored them.
- **Faint limit:** on real lenses, the model stops reliably noticing an added outer source at about 1.5–2× the noise. Training helps at 2–6× and does nothing at 1× or below.
- **Close limit:** an added outer source counts for clearly more than the same extra light down to about 1.2× the inner ring's radius, compared with 1.3× for frozen Zoobot. This is a working limit that depends on the cutoff, and it doesn't show the model can separate two rings.
- **Two-step setup:** when the baseline Zoobot finds lenses and the new model only checks them, lens finding stays at the baseline's 0.99 and all 4 compound lenses are still flagged. It flags 0.4–1.4% of 5,000 new non-lenses, so it stays a second step on lenses already found.

**Still open:** whether the model reads second rings specifically, whether it can separate close rings at all, where the faint limit sits on real compound lenses, and why the 4 real lenses do better than the detection map predicts.

## Why Polar-Zoobot failed

Polar-Zoobot unwrapped each lens into a radius × angle strip. It did see faint and close rings on painted images, but Zoobot retrained on the same images saw them just as well or better, so it added nothing. On realistic simulations, polar alone fell below frozen Zoobot. See [the full explanation](sim/RESULTS.md#why-polar-zoobot-failed).

## What's in this branch

| Folder | Contents |
|---|---|
| `sim/` | Simulator, training and test scripts, `RESULTS.md` and `SIMULATION.md` |
| `sim/out/` | Summaries, CSVs and figures |
| `polar/` | Polar-Zoobot tests and their outputs |
| `data/split.csv` | Which images are train, validation and test |

Trained models, simulated images, Euclid cutouts, other people's data and unconfirmed candidate lists are left out. `sim/RESULTS.md` lists them and how to get or remake them.

## Running it

The scripts expect this folder at `~/ruby` and were run with Python 3.11 (`requirements-py311.txt`). Their inputs and commands are in [sim/RESULTS.md](sim/RESULTS.md#reproducing-the-results). The commands haven't been re-run end to end as one sequence.

## Sources

- Lens catalog and lens models: Euclid Q1 Strong Lensing Discovery Engine, Zenodo, doi:[10.5281/zenodo.15003116](https://doi.org/10.5281/zenodo.15003116)
- Compound lens properties: Li et al. 2026, Euclid Q1 Strong Lensing Discovery Engine D, [arXiv:2503.15327](https://arxiv.org/abs/2503.15327)
- Euclid Q1 MER catalog, NASA/IPAC IRSA, doi:[10.26131/IRSA602](https://doi.org/10.26131/IRSA602)
