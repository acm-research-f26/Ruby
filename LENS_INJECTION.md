# Lens Injection: How It Works, and What It Actually Taught Us

An attempt at the single most evidence-backed fix identified in our earlier
literature research for closing the sim-to-real accuracy gap: instead of
training only on fully-simulated images, paint simulated lensed arcs onto
**real** non-lens Euclid background images, so the training data keeps real
noise/PSF/background statistics and only the arc itself is synthetic
(the HOLISMOKES technique, Cañameras et al.). The result is a genuine,
measured improvement -- with an equally genuine, measured complication that's
more informative than a clean success would have been.

All code: `src/lens_injection.py` (builds the data), `src/train_zoobot_injected.py`
/ `train_zoobot_injected_v2.py` (train on it), `src/evaluate_lens_injection.py`
(the before/after comparison that caught the problem below).

## How the injection itself works

1. `lens_sim.sample_lensed_arc_only(rng)` renders JUST the lensed source
   light -- no lens-galaxy light, no sky noise -- using the same lenstronomy
   physics as the rest of the simulator.
2. That arc's flux is rescaled into the real background image's own fixed
   intensity range (not a rigorous photometric flux calibration between
   simulated and real physical units -- a known, documented simplification)
   and added directly onto a real non-lens Euclid cutout.
3. 6 of the 11 real non-lens Euclid images are used as injection donors, each
   painted with 25 different random arcs (150 synthetic-on-real training
   images total). The remaining 5 non-lens + all 12 real lens images (17
   total) are held out, untouched by injection or training, specifically so
   the before/after comparison below is honest and not leaked.

## What happened (v1): a real improvement, with a catch

Retraining the Zoobot head on simulated data + these 150 injected images,
evaluated on the clean 17-image held-out real set:

| | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Before (sim-only) | 0.59 | 1.00 | 0.42 | 0.59 |
| After (v1, injection) | 0.65 | 0.69 | **0.92** | **0.79** |

Recall more than doubled; F1 jumped meaningfully. But precision collapsed --
the v1 model started calling every one of the 5 held-out real non-lens images
a lens.

## Diagnosing *why*, instead of just reporting the number

A generic "the model got more lens-happy" explanation would predict degraded
performance everywhere, including on simulated non-lens images. That's not
what happened: simulated non-lens validation accuracy actually *improved*
(94.6% -> 99.3%) under v1. The failure was isolated entirely to real images,
and the model wasn't uncertain about them -- checking raw predicted
probabilities (not just the thresholded decision) showed confident wrongness
(0.76-0.96 predicted P(lens) on real non-lens images that should score near
0).

That pattern -- better on sim, worse and *confident* on real -- pointed at a
specific mechanism: the 6 real backgrounds used for injection only ever
appeared in training as **positive** examples. The model had never seen what
a real-image noise/PSF signature looks like paired with a non-lens label, and
likely learned "this real-texture signature -> lens" as a shortcut instead of
genuinely keying on the arc.

## The fix attempt (v2), and the more important finding underneath it

Fix: add the *same* 6 donor backgrounds, unmodified (no injected arc), as
plain negative examples too -- so every real-texture signature appears in
training with both labels, and the only thing left that correlates with
"lens" is the arc itself.

Result: precision fully recovered (back to 1.00) -- but so did the
conservatism. v2's accuracy/precision/recall/F1 landed almost exactly back
at the original sim-only baseline (0.59 / 1.00 / 0.42 / 0.59), and checking
raw probabilities confirmed this wasn't a coincidence of the 0.5 threshold --
v2's predictions are quantitatively close to v0's, not just on the same side
of a cutoff.

**This is the real finding, and it's more useful than a clean win would have
been:** removing the shortcut took almost all of v1's apparent recall gain
with it. That means v1's improvement wasn't mostly "the model learned to
detect arcs better" -- it was substantially *the same shortcut* doing the
work. With only 6 unique real background images to draw from, there likely
wasn't enough diversity for the model to learn a general "arc vs. no arc"
rule; enough to memorize those 6 textures, not enough to generalize past
them.

## What this means for next steps

This directly reinforces, with first-hand measured evidence rather than only
a literature citation, the highest-leverage recommendation from our earlier
research: **more real images -- specifically more real non-lens examples --
is the direct next step**, not further tuning on this same small pool. We
used 23 of the ~497 public Euclid Q1 candidates; pulling more, especially
more non-lens examples to diversify the injection donor pool, is the
most likely way to get a genuine (not shortcut-driven) recall improvement
out of this technique.

## Reproducing

```bash
cd src
python lens_injection.py            # builds the v2 (injection + plain-negative) training set
python train_zoobot_injected_v2.py
python evaluate_lens_injection.py   # before/after comparison on the clean held-out real set
```
