# How the simulated lenses are made

`sim/simulate.py` makes the single and double (compound) lenses used for training and testing. This page explains what is real, what is simulated, and how the lensing is computed.

**In one sentence:** we start from a real Euclid photo of a galaxy that isn't a lens, give it the mass of a real lens, and add the light of one or two background galaxies bent by that mass.

## What's real and what's simulated

| Part of the image | Where it comes from |
|---|---|
| Foreground galaxy and its light | Real Euclid Q1 cutout of a non-lens galaxy |
| Noise and neighbouring galaxies | Real, already in the cutout |
| Lens mass: size, oval shape, shear, centre | Borrowed from Euclid's published model of a real training lens |
| Blur | Set so a simulated point source measures like point sources in our real cutouts |
| Background galaxies | Simulated |
| Inner and outer rings | Computed by tracing light through the lens |

The galaxy in the photo isn't really a lens. Its mass comes from a different lens, lined up with the photo galaxy's light.

## How the lensing is computed

The code works backward from the camera, at 16 points per Euclid pixel (a 4 × 4 grid):

1. **Get the bend.** The lens model gives how much the mass bends light at that point.
2. **Trace it back.** Position minus bend gives the spot behind the lens where that light came from.
3. **Check for a source.** If that spot lands on a background galaxy, the point takes that galaxy's brightness. If not, it adds nothing.

Then the light is blurred, averaged into 0.1″ pixels, scaled to the image's noise and added to the real photo.

These are the equations, where `t` is the position on the image:

```text
inner source:  b1 = t − a(t)
outer source:  b2 = t − d·a(t) − a_s(b1)
```

| Symbol | Meaning |
|---|---|
| `a(t)` | The main lens's bend: an elliptical mass (SIE) plus external shear |
| `d` | The distance ratio, equal to the outer ring's radius divided by the inner ring's radius |
| `a_s(b1)` | The extra bend from the inner source's own mass |

**Why rings form.** Take a round lens that bends light by 0.87″ everywhere, with a source right behind its centre. A point 0.87″ from the centre traces back exactly onto the source, and so does every other point on that circle, so a full ring appears. An off-centre source only catches part of the circle, which gives arcs. A farther source gets bent more: with `d` = 1.4 the bend is 1.22″, so the second ring appears at 1.4× the first.

**Check:** `check_sim.py` compares these equations with lenstronomy's full multi-plane calculation, and they agree to under 0.00001″.

## What gets drawn for each lens

| Value | How it's chosen |
|---|---|
| Lens mass | One of 221 models of real training lenses: Einstein radius, axis ratio, shear strength and centre. The mass angle follows the galaxy's light, give or take about 20°. The shear direction is random. |
| Inner source position | Random, within 0.3× the inner ring's radius of the lens centre |
| Outer source distance `d` | Uniform between 1.1 and 1.8 |
| Outer source position | Random, within 0.3× the outer ring's radius of the lens centre |
| Source size | Half-light radius with a median of 0.05″, kept between 0.03″ and 0.4″ |
| Source shape | A smooth body (Sérsic index 0.7–2.5, axis ratio 0.4–1, random angle) plus 0–6 bright clumps, each 0.02–0.06″ wide |
| Inner source's own mass | Round, with an Einstein radius of 0.02–0.2″ |
| Decoy neighbour | In 25% of lenses, an extra galaxy at the inner source's distance, 0.3–1.0″ away from it. It makes extra arcs at the same radius, so extra light alone doesn't mean compound. |
| Inner ring brightness | Matched to how bright the chosen real lens's ring is, in noise units |
| Outer source brightness `b` | Bright part of the source at 0.5–6× the image's noise, spread evenly on a log scale |
| Blur | Two parts: 0.15″ and 0.36″ wide (FWHM), with half the light in each |

**Host galaxies.** These are real non-lens galaxies from the same data split. Broken cutouts, spirals and clumpy galaxies are left out. Every real training lens galaxy then votes for the 5 remaining galaxies most like it, by brightness, concentration, roundness, lopsidedness and noise, so the picks resemble real lens galaxies.

**Pairs.** Every double has a matching single made from the same galaxy, mass and inner source. The outer source is the only difference, so a model can only tell them apart by noticing it.

**Data sets.** 3,000 training, 600 validation and 600 test pairs (`sim/out/v2`). Galaxies stay in their own split, and lens models come from training lenses only, so nothing about the validation or test lenses reaches the simulator, including the 4 compound lenses. For the second training round, doubles with a total signal under 6× the noise were left out of training (2,011 of 3,000 kept).

## How realistic it is

`realism_check.py` trains a classifier to tell real ordinary lenses from simulated singles. Lower means harder to tell apart:

| Images | Retrained Zoobot AUC |
|---|---|
| Painted rings | 0.996 |
| Version 1 | 0.919, 0.900, 0.932 |
| **Version 2 (used)** | **0.909, 0.904, 0.900** |
| Version 3 (real galaxy shapes as sources) | 0.930, 0.928, 0.930 |

The pass mark was below 0.90 on every seed. Version 2 missed it after the 2 allowed fix rounds, but it is far harder to tell from real lenses than painted rings. The ring zone is the main giveaway: 0.864 with only that zone visible, 0.703 with it hidden.

Some settings were tuned to look like real training lenses, so they aren't independent of the real data: inner ring brightness, source size and source placement.

## Limits

- **Simple sources:** the background galaxies are simple shapes. Real galaxy shapes were tried (version 3) and were easier to spot.
- **No arc noise:** the extra noise from the arcs' own light isn't modelled. It's small next to the sky noise for faint arcs.
- **Approximate blur:** the two-part blur only approximates Euclid's real blur.
- **Borrowed mass:** the photo galaxy isn't a real lens, and its mass comes from a different lens.
- **Distance as a ratio:** distances are set by the ratio `d`, not by redshifts. The range 1.1–1.8 is assumed.
- **Realism mark missed:** a classifier can still tell these from real lenses about 90% of the time.

## Files

| File | What it does |
|---|---|
| `simulate.py` | Makes the simulated pairs |
| `prepare_lens_models.py` | Turns Euclid's published lens models into `lens_models.csv`, training lenses only |
| `measure_blur.py` | Measures the blur in our real cutouts |
| `check_sim.py` | Checks the equations against lenstronomy, ring radius, brightness and thickness, and host galaxies |
| `tune_realism.py` | Tries different settings to make the simulations harder to tell from real lenses |
| `realism_check.py` | The realism check above |
| `score_hosts.py` | An alternative way to pick host galaxies, not used by default |
