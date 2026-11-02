# get_more_images.py: more Q1 images for sim/bigger_test.py, picked and cut the same way as get_negatives.py and make_cutouts.py.
#
# Usage (from ~/ruby, with lens_env311):
#   python -I sim/get_more_images.py
#   python -I sim/get_more_images.py --target 20 --limit-c 20 --out data/expand_try     (checks the script)
#
# New non-lenses: same catalog query and same patch method as get_negatives.py, with a new random seed. Galaxies already in
# negatives.csv are left out, and so is anything within 10 arcsec of a known lens candidate of any grade.
# Grade C candidates: every grade C row of the lens catalog except gz_euclid, which is outside Q1.
# Cutouts: same tile lookup, 11-arcsec crop, 100x100 trim and missing-pixel limit as make_cutouts.py.
#
# Safe to stop and re-run: finished patches and saved cutouts are kept.
#
# Output (in --out): non_lenses.csv, targets.csv, manifest.csv, non_lenses/<id>.npy, grade_c/<id>.npy

import argparse
import random
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS, FITSFixedWarning
from astroquery.ipac.irsa import Irsa

warnings.filterwarnings("ignore", category=FITSFixedWarning)

ROOT = Path.home() / "ruby"
SIZE = 100
MAX_NAN_FRAC = 0.05
PER_PATCH = 20
RETRIES = 5
MAX_FAILURES = 30    # catalog queries failing in a row before the script stops and says so
tile_cache = []


def limited(work, seconds=300):
    """Runs an IRSA call with a time limit, because a connection can hang for good if the Mac sleeps during the call."""
    box = {}
    def run():
        try:
            box["result"] = work()
        except Exception as e:
            box["error"] = e
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    if "result" in box:
        return box["result"]
    raise box.get("error", TimeoutError(f"no answer in {seconds} s"))


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


def pick_non_lenses(target, out):
    path = out / "non_lenses.csv"
    done = pd.read_csv(path, dtype={"object_id": str}) if path.exists() else pd.DataFrame()
    if len(done) >= target:
        return done
    random.seed(2026 + len(done))    # a re-run keeps what it has and doesn't repeat the same patches
    lenses = Table.read(ROOT / "q1_discovery_engine_lens_catalog.csv", format="csv")
    lens_ra = np.array(lenses["right_ascension"], dtype=float)
    lens_dec = np.array(lenses["declination"], dtype=float)
    anchors = lenses[lenses["subset"] != "gz_euclid"]
    used = set(pd.read_csv(ROOT / "negatives.csv", dtype={"object_id": str}).object_id)

    collected, seen, tries, failures = [done], set(done.object_id) if len(done) else set(), 0, 0
    save = lambda: pd.concat(collected, ignore_index=True).to_csv(path, index=False)
    while len(seen) < target and tries < 3 * target // PER_PATCH + 100:
        a = anchors[random.randrange(len(anchors))]
        ra = a["right_ascension"] + random.uniform(-0.3, 0.3)
        dec = a["declination"] + random.uniform(-0.3, 0.3)
        try:
            patch = limited(lambda: query_patch(ra, dec))
        except Exception as e:
            failures += 1
            print(f"  query failed ({type(e).__name__}), {failures} in a row", flush=True)
            if failures >= MAX_FAILURES:
                save()
                raise SystemExit(f"IRSA stopped answering. Kept {len(seen)} galaxies; re-run to continue.")
            time.sleep(min(2 ** failures, 60))
            continue
        failures = 0
        tries += 1
        if len(patch) == 0:
            continue

        ra1, dec1 = np.radians(np.array(patch["ra"]))[:, None], np.radians(np.array(patch["dec"]))[:, None]
        ra2, dec2 = np.radians(lens_ra)[None, :], np.radians(lens_dec)[None, :]
        h = np.sin((dec2 - dec1) / 2) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2) ** 2
        patch = patch[np.degrees(2 * np.arcsin(np.sqrt(h))).min(axis=1) * 3600 > 10]
        patch = patch[[str(i) not in used and str(i) not in seen for i in patch["object_id"]]]
        if len(patch) == 0:
            continue

        idx = random.sample(range(len(patch)), min(PER_PATCH, len(patch)))
        collected.append(patch[idx].to_pandas().astype({"object_id": str}))
        seen.update(str(i) for i in patch[idx]["object_id"])
        if tries % 25 == 0:
            save()
            print(f"patch {tries}: {len(seen)} galaxies so far", flush=True)

    save()
    print(f"{len(seen)} new non-lenses, {tries} more patches", flush=True)
    return pd.read_csv(path, dtype={"object_id": str})


def find_tile(coord):
    for url, wcs, shape in tile_cache:
        x, y = wcs.world_to_pixel(coord)
        if SIZE < x < shape[1] - SIZE and SIZE < y < shape[0] - SIZE:
            return url, wcs
    imgs = limited(lambda: Irsa.query_sia(pos=(coord, 5 * u.arcsec), collection="euclid_DpdMerBksMosaic"))
    vis = imgs[(imgs["dataproduct_subtype"] == "science") & (imgs["energy_bandpassname"] == "VIS")]
    if len(vis) == 0:
        return None, None
    url = vis["access_url"][0]
    header = fits.getheader(url, use_fsspec=True)
    wcs = WCS(header)
    tile_cache.append((url, wcs, (header["NAXIS2"], header["NAXIS1"])))
    return url, wcs


def fetch_cutout(url, coord):
    crop_url = f"{url}?center={coord.ra.deg},{coord.dec.deg}&size=11arcsec&gzip=false"
    with fits.open(crop_url, cache=False) as h:
        return h[0].data.astype(np.float32), WCS(h[0].header)


def cut(row):
    """Returns (row, what happened): saved, no tile, too many missing pixels, or the error after every retry."""
    path = ROOT / row["path"]
    if path.exists():
        return row, "saved"
    for attempt in range(RETRIES):
        try:
            coord = SkyCoord(row["ra"], row["dec"], unit="deg")
            url, _ = find_tile(coord)
            if url is None:
                return row, "no tile"
            crop, crop_wcs = fetch_cutout(url, coord)
            data = np.asarray(Cutout2D(crop, position=coord, size=(SIZE, SIZE), wcs=crop_wcs, mode="partial", fill_value=np.nan).data, dtype=np.float32)
            if data.shape != (SIZE, SIZE) or np.isnan(data).mean() > MAX_NAN_FRAC:
                return row, "too many missing pixels"
            np.save(path, data)
            return row, "saved"
        except Exception as e:
            problem = f"error: {type(e).__name__}"
            time.sleep(2 ** attempt)
    return row, problem


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=5000)
    parser.add_argument("--limit-c", type=int, default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", default="data/expand")
    args = parser.parse_args()
    out = ROOT / args.out
    (out / "non_lenses").mkdir(parents=True, exist_ok=True)
    (out / "grade_c").mkdir(parents=True, exist_ok=True)

    non = pick_non_lenses(args.target, out)
    cat = pd.read_csv(ROOT / "q1_discovery_engine_lens_catalog.csv", dtype={"id_str": str}, low_memory=False)
    cat = cat[(cat.subset != "gz_euclid") & (cat.grade == "C")]
    if args.limit_c:
        cat = cat.head(args.limit_c)
    targets = pd.concat([
        pd.DataFrame(dict(id=non.object_id, kind="non_lenses", grade="", notes="", ra=non.ra, dec=non.dec)),
        pd.DataFrame(dict(id=cat.id_str, kind="grade_c", grade="C", notes=cat.notes.fillna(""), ra=cat.right_ascension, dec=cat.declination)),
    ], ignore_index=True)
    targets["path"] = [f"{args.out}/{k}/{i}.npy" for k, i in zip(targets.kind, targets.id)]
    targets.to_csv(out / "targets.csv", index=False)
    print(f"{(targets.kind == 'non_lenses').sum()} non-lenses + {(targets.kind == 'grade_c').sum()} grade C to cut", flush=True)

    rows, counts = [], {}
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (row, result) in enumerate(pool.map(cut, targets.to_dict("records"), chunksize=8), 1):
            counts[result] = counts.get(result, 0) + 1
            if result == "saved":
                rows.append(row)
            if i % 250 == 0 or i == len(targets):
                print(f"{i}/{len(targets)} done: {counts}", flush=True)
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    print(f"Done. {len(rows)} cutouts saved. {counts}", flush=True)
