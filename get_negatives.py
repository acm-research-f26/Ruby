# get_negatives.py: collect ~2,000 random Euclid Q1 galaxies (non-lenses)
import random
import numpy as np
from astropy.table import Table, vstack, unique
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.ipac.irsa import Irsa

TARGET = 2000          # how many non-lenses we want
PER_PATCH = 20         # max galaxies to keep from each patch (spreads them out)
random.seed(42)        # makes the random choices repeatable

# 1. Load the known lens candidates (all grades A, B, C)
lenses = Table.read("q1_discovery_engine_lens_catalog.csv", format="csv")
lens_coords = SkyCoord(lenses["right_ascension"], lenses["declination"], unit="deg")

# Only use lenses inside Q1 as "anchor" spots (gz_euclid ones are outside Q1)
anchors = lenses[lenses["subset"] != "gz_euclid"]

def query_patch(ra, dec, radius=0.05):
    q = f"""
    SELECT object_id, ra, dec, flux_detection_total, point_like_prob
    FROM euclid_q1_mer_catalogue
    WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius})) = 1
      AND flux_detection_total > 3.63
      AND spurious_flag = 0
      AND point_like_prob < 0.5
    """
    return Irsa.query_tap(q).to_table()

collected = []
total = 0
tries = 0
while total < TARGET and tries < 400:
    tries += 1
    # 2. Pick a random known lens, then jump a random distance away from it
    a = anchors[random.randrange(len(anchors))]
    ra = a["right_ascension"] + random.uniform(-0.3, 0.3)
    dec = a["declination"] + random.uniform(-0.3, 0.3)
    try:
        patch = query_patch(ra, dec)
    except Exception as e:
        print(f"  query failed ({e}), skipping")
        continue
    if len(patch) == 0:
        continue    # landed outside Q1's coverage, try again

    # 3. Remove anything within 10 arcseconds of ANY known lens candidate
    coords = SkyCoord(patch["ra"], patch["dec"], unit="deg")
    _, sep, _ = coords.match_to_catalog_sky(lens_coords)
    patch = patch[sep > 10 * u.arcsec]

    # 4. Keep a random handful from this patch
    idx = random.sample(range(len(patch)), min(PER_PATCH, len(patch)))
    collected.append(patch[idx])
    total += len(idx)
    print(f"patch {tries}: +{len(idx)}  (total {total})")

negatives = unique(vstack(collected), keys="object_id")   # drop duplicates
negatives.write("negatives.csv", format="csv", overwrite=True)
print(f"Done! Saved {len(negatives)} non-lens galaxies to negatives.csv")