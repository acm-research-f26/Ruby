# make_cutouts.py: cut 100x100-pixel Euclid VIS images for lenses AND non-lenses
#
# Both sets go through exactly the same code, so the model can't learn
# "which pipeline made this image" instead of "is this a lens".
#
# Output:
#   data/cutouts/lenses/<id>.npy      raw VIS pixels (float32), no scaling yet
#   data/cutouts/negatives/<id>.npy
#   data/manifest.csv                 one row per saved cutout (id, label, grade, ...)
#
# Safe to stop and re-run: cutouts that already exist are skipped.

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS, FITSFixedWarning
from astroquery.ipac.irsa import Irsa

warnings.filterwarnings("ignore", category=FITSFixedWarning)

SIZE = 100            # pixels; Euclid VIS is 0.1"/pixel, so 100 px = 10", same as the paper
MAX_NAN_FRAC = 0.05   # skip cutouts where >5% of pixels are missing (edge of the survey)
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None   # e.g. `python make_cutouts.py 5` for a quick test

os.makedirs("data/cutouts/lenses", exist_ok=True)
os.makedirs("data/cutouts/negatives", exist_ok=True)

# ---- 1. Build the list of objects to cut --------------------------------------
cat = pd.read_csv("q1_discovery_engine_lens_catalog.csv")
cat = cat[cat["subset"] != "gz_euclid"]              # gz_euclid lenses are outside Q1
lenses = cat[cat["grade"].isin(["A", "B"])].copy()   # grade C is too uncertain to call a lens
lenses = pd.DataFrame({
    "id": lenses["id_str"].astype(str),
    "ra": lenses["right_ascension"],
    "dec": lenses["declination"],
    "label": 1,
    "grade": lenses["grade"],
    "notes": lenses["notes"].fillna(""),
})

neg = pd.read_csv("negatives.csv")
neg = pd.DataFrame({
    "id": neg["object_id"].astype(str),
    "ra": neg["ra"],
    "dec": neg["dec"],
    "label": 0,
    "grade": "",
    "notes": "",
})

targets = pd.concat([lenses, neg], ignore_index=True)
if LIMIT:  # quick test: a few of each
    targets = pd.concat([lenses.head(LIMIT), neg.head(LIMIT)], ignore_index=True)
print(f"{(targets.label == 1).sum()} lenses + {(targets.label == 0).sum()} non-lenses to cut")

# ---- 2. Find which image tile each object is in ------------------------------
# Each Euclid VIS tile is 19,200 x 19,200 pixels. We only read each tile's header
# (to know which sky area it covers), then ask IRSA's server to crop the small
# square for us. One small download per object instead of hundreds of reads.
tile_cache = []   # list of (url, wcs, shape)

def find_tile(coord):
    """Return (url, wcs) of a VIS tile containing coord."""
    for url, wcs, shape in tile_cache:
        x, y = wcs.world_to_pixel(coord)
        if SIZE < x < shape[1] - SIZE and SIZE < y < shape[0] - SIZE:
            return url, wcs
    imgs = Irsa.query_sia(pos=(coord, 5 * u.arcsec), collection="euclid_DpdMerBksMosaic")
    vis = imgs[(imgs["dataproduct_subtype"] == "science") & (imgs["energy_bandpassname"] == "VIS")]
    if len(vis) == 0:
        return None, None
    url = vis["access_url"][0]
    header = fits.getheader(url, use_fsspec=True)       # header only, not the pixels
    wcs = WCS(header)
    tile_cache.append((url, wcs, (header["NAXIS2"], header["NAXIS1"])))
    return url, wcs

def fetch_cutout(url, coord):
    """Ask IRSA to crop an 11-arcsec square around coord; returns an HDU."""
    crop_url = f"{url}?center={coord.ra.deg},{coord.dec.deg}&size=11arcsec&gzip=false"
    with fits.open(crop_url, cache=False) as h:
        return h[0].data.astype(np.float32), WCS(h[0].header)

# ---- 3. Cut, check, save ------------------------------------------------------
rows, skipped = [], 0
start = time.time()
for i, t in enumerate(targets.itertuples(), 1):
    folder = "lenses" if t.label == 1 else "negatives"
    path = f"data/cutouts/{folder}/{t.id}.npy"
    if not os.path.exists(path):
        try:
            coord = SkyCoord(t.ra, t.dec, unit="deg")
            url, _ = find_tile(coord)
            if url is None:
                skipped += 1
                continue
            crop, crop_wcs = fetch_cutout(url, coord)
            # trim the 111x111 crop to exactly 100x100, centred on the object
            cut = Cutout2D(crop, position=coord, size=(SIZE, SIZE), wcs=crop_wcs,
                           mode="partial", fill_value=np.nan)
            data = np.asarray(cut.data, dtype=np.float32)
            if data.shape != (SIZE, SIZE) or np.isnan(data).mean() > MAX_NAN_FRAC:
                skipped += 1
                continue
            np.save(path, data)
        except Exception as e:
            print(f"  error on {t.id}: {e}")
            skipped += 1
            continue
    rows.append({"id": t.id, "label": t.label, "grade": t.grade, "notes": t.notes,
                 "ra": t.ra, "dec": t.dec, "path": path})
    if i % 50 == 0 or i == len(targets):
        rate = (time.time() - start) / i
        print(f"{i}/{len(targets)} done, {skipped} skipped, ~{rate * (len(targets) - i) / 60:.0f} min left")

pd.DataFrame(rows).to_csv("data/manifest.csv", index=False)
print(f"Done! {len(rows)} cutouts saved, {skipped} skipped. List in data/manifest.csv")
