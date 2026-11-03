# Compound Lens Detection: How It Works

A classical signal-processing method that takes a gravitational lens image
already verified to contain at least one Einstein ring/arc, and checks
whether it shows evidence of actually being a **compound** (double-ring)
lens. No training data, no model -- this is deliberately NOT machine
learning, which is what makes it useful here: every other approach in this
project needs real labeled data we don't have much of; this needs none.

All code: `src/radial_profile.py` (the method), `src/preview_radial_profiles.py`
(the validation grids), `src/double_ring_sensitivity.py` (the sensitivity sweep).

## The physical idea

Picture the brightness of a lens image averaged over circles of increasing
radius, centered on the lens galaxy. A normal single lens shows one bump in
that curve -- the Einstein ring -- sitting on top of the galaxy's own smooth,
monotonically-declining light. A compound lens, with two source planes, should
show a **second** bump at a different radius. "Make a graph and see where it
dips" is literally what this does.

## The pipeline, function by function

1. **`radial_profile(image, center)`** -- sweeps outward in thin circular
   rings (annuli) from the lens center and averages flux at each radius.
   Collapses the 2D image into a 1D curve: flux vs. radius.

2. **`detrend(profile)`** -- a ring is a *local* bump on top of a *smooth*
   decline. This subtracts a heavily-smoothed copy of the profile from
   itself, zeroing out the smooth part and leaving only the bumps (the
   residual).

3. **`count_rings(image)`** -- finds peaks in that residual, with two
   calibrated safeguards, both added after empirical testing exposed real
   failure modes (not guessed in advance):
   - **Robust noise threshold.** A peak only counts if it clears a multiple
     of the residual's own noise floor, estimated via median absolute
     deviation (MAD) rather than plain standard deviation. Plain std is
     skewed by the very peak you're trying to detect (a strong ring inflates
     its own detection threshold); MAD mostly ignores that outlier.
   - **Minimum peak separation (9px).** A single *asymmetric* arc (common
     when the source isn't perfectly centered) isn't a perfect circle, so a
     naive radial average can smear its own clumpiness into two nearby
     sub-peaks that look like "two rings" but are really one. Calibrated by
     measuring the actual gaps this produces on 12 known-single simulated
     lenses (5-7.5px) and setting the floor above that. This single fix took
     the false-positive rate from 42% down to 0%.

4. **`angular_profile` / `angular_coverage` / `classify_peak`** -- the
   second-stage filter, and the one that matters most. A detected radial
   peak could be a genuine ring, OR an unrelated companion galaxy / artifact
   that happens to sit at some radius from the lens center. The
   distinguishing signal: a ring spreads its flux across a wide range of
   *angles* around that radius; a compact companion object concentrates its
   flux in one narrow angular wedge, however bright. `angular_coverage`
   slices a thin ring at the peak's radius and measures what fraction of the
   full circle (72 angle bins) exceeds half the local peak brightness.
   `classify_peak` labels each peak `ring-like` or `point-like (likely
   companion)` against a threshold, and separately flags `near_edge` if the
   peak sits close to the image boundary (a genuine second ring from the
   same lens system should be well within the cutout; something clipped at
   the edge is more likely an unrelated field object).

   The threshold itself went through two calibration passes, both logged in
   the code's docstrings: the first (36 angle bins, threshold 0.05) worked
   on the original 12-lens check but was later shown, by the sensitivity
   sweep below, to sit too close to its own companion-signature floor,
   misclassifying some genuine (but compact) rings. Re-calibrated at 72 bins
   with threshold 0.02 on a larger 30-image sample -- real margin on both
   sides this time, not a coincidence of a small sample.

## Validation -- both directions, not just one

A method that simply never says "compound" would score perfectly on a
false-positive check alone. Both were measured:

**Specificity** (`preview_radial_profiles.py`): 0/12 false positives on
simulated single lenses with known ground truth (exactly one ring). 1/12 real
verified lenses flagged, with that one case's ambiguity (a peak right at the
threshold, also near the image edge) reported explicitly rather than forced
into a confident answer -- see `outputs/radial_profile_real_lenses.png`.

**Sensitivity** (`double_ring_sensitivity.py`): synthetic double rings built
by combining two independently-rendered lensed arcs (via
`lens_sim.sample_lensed_arc_only(rng, theta_e=...)`) at controlled radius
separation and brightness ratio, on a shared lens-light + noise background
(`lens_sim.sample_background_only`). This is a proxy for a true two-deflector
compound lens, not a full physical multi-plane simulation -- it tests the
detection *pipeline* directly (does it correctly count 2 angularly-broad
bumps when 2 exist) without the added complexity of multi-plane lens
modeling.

Result, swept across 5 separation ratios x 4 brightness ratios x 15 trials
each (`outputs/double_ring_sensitivity_heatmap.png`):

| Separation (outer/inner radius) | Detection rate |
|---|---|
| 1.3x-1.5x (rings close together) | 7-20% |
| 2.0x-4.0x (rings clearly separated) | 47-80% |

The close-separation result isn't a bug -- it reproduces something the team
already expected from direct image inspection: rings that close together
visually blend, and a radius-based method genuinely struggles to resolve
them. That the quantified result matches the qualitative expectation is a
useful sanity check that the method is measuring something real.

## Update: a real prominence-measurement bug, found via a new real-world test

Running this detector on a new, independent 79-image real Euclid test set
(see [`REAL_WORLD_TEST.md`](REAL_WORLD_TEST.md)) flagged 38/79 (48%) as
compound -- far above the 0/12 and 1/12 false-positive rates measured here.
Inspecting the flagged profiles directly (not just the number) showed why:
`scipy.signal.find_peaks`'s default prominence measurement reuses the
deepest valley it can reach without crossing a *higher* peak as the
reference for every peak, including distant, unrelated ones -- so one real,
deep dip right after the genuine ring was inflating the apparent
significance of tiny noise ripples all the way out to the image edge.
Fixed via `find_peaks`'s `wlen` parameter, now exposed as
`count_rings(..., local_window_px=10.0)`, which caps how far outward a
peak's prominence is allowed to be measured. This fix cut the new set's
false-positive rate to 30/79 with **no change at all** to this page's 0/12
and 1/12 results and no material change to the sensitivity sweep table
above (re-run after the fix; same shape, same close-vs-wide pattern). A
tighter window suppresses more false positives but starts costing real
sensitivity at wide (3-4x) separations -- 10px was chosen as the point that
fixes this specific failure mode without touching that tradeoff. Full
reasoning: `local_window_px` in the `count_rings()` docstring in
`src/radial_profile.py`.

This is a reminder that **specificity measured on 12 images doesn't
guarantee specificity on a different real dataset** -- the fix came from
actually running the method on new data and looking at what it found, not
from more theorizing about the existing calibration set.

## Known limitations (stated plainly)

- **Close-separation blind spot.** Below ~1.5x radius ratio, detection is
  weak. This may be a fundamental limit of a radius-only method, not
  something more threshold tuning fixes.
- **Angular coverage alone can't fully separate "very compact genuine ring"
  from "companion object right at the image edge."** The `near_edge` flag is
  a second, independent signal for this specific ambiguity, not a complete
  resolution of it -- deliberately not auto-excluding on one example, since
  that's too little data to safely calibrate a hard cutoff against.
- **The sensitivity sweep uses a synthetic proxy**, not real compound lenses
  or a true multi-plane simulation. Real validation against the 4 known
  compound lenses named in Wilde et al. 2022 (archival HST/HSC imaging) is
  the natural next step.
- Calibration throughout is based on small samples (12-30 images) -- typical
  for this project given how little real labeled data exists, but worth
  remembering when trusting any single threshold.

## Reproducing

```bash
cd src
python preview_radial_profiles.py    # specificity check + the main result grids
python double_ring_sensitivity.py    # sensitivity sweep (a few minutes)
```
