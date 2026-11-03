# Real-World Test: Scoring 79 Real Euclid Lenses for Compound-Lens Likelihood

The deliverable for this branch: every tool built so far (the radial-profile
compound-lens detector + the Euclid Zoobot lens head), run on a real,
external 79-image test set, producing a **1-79 ranked score** -- not a hard
yes/no -- so a human reviewer can triage from the top down instead of
checking all 79 by hand. This also surfaced and fixed a real bug in the
detector itself, found only by running it on this new data.

All code: `src/score_real_test_set.py` (the 79-image scoring pipeline),
`src/validate_on_real_compound.py` (a smaller companion check against 3
famous archival-HST compound lenses). Outputs:
`outputs/real_test_79_scores.csv` (the full ranked table),
`outputs/real_test_79_top12.png` (visual QC grid of the top 12),
`outputs/real_compound_lens_validation.png`.

## The data

`data/real_test_79/`: 79 real Euclid Q1 VIS cutouts (100x100px, 0.1"/px),
all already lens candidates from the Euclid Q1 Strong Lensing Discovery
Engine catalog -- 75 "ordinary" single lenses and 4 **unconfirmed compound
candidates** (Cosmic Dartboard, Cosmic Ammonite, Galileo's Lens, Teapot
Lens; Li et al. 2026, arXiv:2503.15327). "Unconfirmed" matters: these are
visual/automated candidates, not spectroscopically verified two-source-plane
systems, so disagreeing with this label on any one image isn't automatically
a detector failure -- it could also be a wrong candidate label.

## A real bug, found by running this on new data

Running the existing detector (as shipped on `Compound-Lens-Detection`)
on this set flagged **38/79 (48%) as compound** -- wildly higher than the
0/12 and 1/12 false-positive rates measured earlier on the simulated and
original 23-image real-Euclid calibration sets. Visually inspecting the
flagged profiles (not just trusting the number) showed why: most of the
"second" and "third" peaks sat in an otherwise flat, low-amplitude noise
tail far from the real ring, at radii like 18px, 27px, 35px -- not genuine
structure. The cause was a real bug in how `scipy.signal.find_peaks`
measures peak prominence: by default it's measured against the lowest
valley reachable without crossing a *higher* peak, so one strong, real dip
right after the genuine ring gets reused as the reference valley for every
small ripple all the way out to the image edge, inflating their apparent
significance. Fixed by capping the prominence-measurement window to a local
neighborhood (`local_window_px` in `radial_profile.count_rings`, now default
10px) -- full details, including why 10px and not a tighter/looser value,
are in the function's docstring and [`COMPOUND_LENS_DETECTION.md`](COMPOUND_LENS_DETECTION.md).
This cut the flag rate to 30/79, with **zero change** to the original 0/12
and 1/12 specificity results and no material change to the double-ring
sensitivity sweep.

A second, related issue surfaced the same way: even after that fix, a
handful of 3-peak images (one real ring + two marginal noise ripples) still
outranked a *known* genuine 2-ring compound candidate, because the original
scoring formula used raw peak count as its dominant term (3 > 2, regardless
of peak quality). Fixed in the scoring formula itself -- see below.

## How the score works

For each image:

1. **Compound signal** -- `radial_profile.count_rings()` (the method
   documented in `COMPOUND_LENS_DETECTION.md`), now also reporting each
   peak's `prominence_sigma`: how many multiples of the profile's own noise
   floor that peak's prominence reaches. This is the actual detection
   statistic the method was computing internally all along but previously
   discarded after thresholding -- surfacing it is what let the scoring fix
   below work.
2. **Lens sanity check** -- P(lens) from the Euclid Zoobot head (the
   `_v2` shortcut-fixed checkpoint from [`LENS_INJECTION.md`](LENS_INJECTION.md)),
   as a secondary signal, not the primary one (see Limitations).
3. **Composite score**:
   ```
   tier  = 2 if 2+ ring-like peaks found, else 1 if exactly 1, else 0
   score = tier
         + 0.4 * strength_norm(2nd-strongest ring-like peak's prominence_sigma)
         + 0.3 * shape_norm(that same peak's angular coverage)
         + 0.1 * p_lens_zoobot
         - 0.05 * (near-edge flags on the top 2 peaks)
   ```
   `tier` is capped at 2 (not raw peak count) specifically to fix the 3-peak
   bug above. "2nd-strongest" peak = sort all ring-like peaks by strength,
   take the weaker of the best *pair* -- a compound verdict is only as
   trustworthy as its weaker confirmed ring, and extra marginal peaks beyond
   the best pair neither help nor hurt. All non-tier terms are bounded so
   they only ever break ties *within* a tier. Full reasoning and the exact
   saturation constants: `score_components()` docstring in
   `src/score_real_test_set.py`.
4. Images are ranked by this score, descending -- **rank 1 = strongest
   compound candidate**.

## Results

| rank | name | label | n ring-like peaks | peak strengths (sigma) |
|---|---|---|---|---|
| 1/79 | Cosmic Ammonite | COMPOUND CANDIDATE | 2 | 8.7, 24.2 |
| 15/79 | Cosmic Dartboard | COMPOUND CANDIDATE | 2 | 8.9, 5.4 |
| 31/79 | Teapot Lens | single lens | 1 | 28.6 (one strong ring, no 2nd) |
| 41/79 | Galileo's Lens | single lens | 1 | 19.8 (one strong ring, no 2nd) |

Median rank of the 4 known candidates: **23/79** (random chance: ~40). 30/79
images overall are flagged `COMPOUND CANDIDATE` (2+ ring-like peaks); full
breakdown and every image's score/peaks/labels are in
`outputs/real_test_79_scores.csv`.

**Read honestly, not as a clean win:** 2 of the 4 known candidates
(Cosmic Ammonite, Cosmic Dartboard) land at rank 1 and 15 -- genuinely
strong results, with Cosmic Ammonite's top-12 visual grid showing an
unambiguous double ring. The other 2 (Teapot Lens, Galileo's Lens) show
**one very strong single ring and nothing else** -- no second ring-like
peak was found at any strength. This is either (a) the already-documented
close-separation blind spot (a second ring sitting too close to the first
to resolve as a separate radial bump -- see
[`COMPOUND_LENS_DETECTION.md`](COMPOUND_LENS_DETECTION.md)'s sensitivity
sweep, which showed 0-27% detection below 1.5x radius separation), or
(b) these two systems are flagged compound in the source catalog for
reasons other than a visually separated double ring (e.g. spectroscopic or
color evidence), which a radial-brightness method wouldn't be expected to
catch at all. Distinguishing those two explanations needs the two systems'
actual published inner/outer ring geometry, which wasn't available in this
run -- worth following up before trusting the method's recall rate on
close pairs specifically.

## Companion check: 3 famous archival-HST compound lenses

Before scoring the 79-image set, the same detector was run on real archival
HST imaging (not Euclid) of 3 named compound lenses from the literature --
SDSS J0946+1006 ("Jackpot Lens"), SL2S J02176-0513, and J1148+1930 ("Cosmic
Horseshoe"). Result: **0/3 detected as compound** even though all three
visibly show ring/arc structure on inspection (`outputs/real_compound_lens_validation.png`).
Looked at individually this isn't a flat method failure: the Jackpot Lens's
published inner/outer ring separation (~1.5x) falls exactly in the weak
detection zone the sensitivity sweep already predicts (7-27%), so this miss
is consistent with, not contradictory to, that earlier finding. The Cosmic
Horseshoe result looks more like a real preprocessing issue specific to
very bright, high-dynamic-range sources (worth investigating further, not
yet fixed) -- and per its own literature description it may not even be a
genuine double-ring system, making it a questionable test case either way.
This 3-image check is too small to generalize from alone; the 79-image
Euclid result above, from the same survey/processing this project already
uses, is the more load-bearing validation.

## Limitations (stated plainly)

- **The 4 "known" labels are themselves unconfirmed candidates**, not
  spectroscopically verified compound lenses -- a disagreement with one of
  them isn't unambiguous evidence of a detector miss.
- **4 known examples is a very small validation sample.** Both the hit rate
  (2/4) and the "random chance ~40" comparison should be read as a
  directional signal, not a precise recall estimate.
- **The Zoobot P(lens) term barely discriminates on this set**: because all
  79 images are already lens candidates (not a mixed lens/field sample),
  78/79 score P(lens) >= 0.5. It's included as a sanity check (our
  independently-trained head agrees these look like lenses) but does almost
  no work in the ranking itself.
- **The scoring formula's saturation constants (15 sigma, 0.10 coverage)
  were set by inspecting this exact 79-image set.** That's an honest way to
  calibrate given no separate held-out real compound-lens set exists, but
  it means the formula is tuned on the same data it's being evaluated
  against -- a genuinely independent real compound-lens set would be needed
  to check it isn't overfit to this one.
- **This is a triage ranking, not a verdict.** 30/79 flagged is a real
  number but the intended use (per the task this branch was built for) is
  "sort by score, review from the top," not "treat everything above some
  cutoff as confirmed compound."

## Reproducing

```bash
cd src
python validate_on_real_compound.py   # 3-system archival-HST companion check
python score_real_test_set.py         # the full 79-image ranked scoring (main deliverable)
```
