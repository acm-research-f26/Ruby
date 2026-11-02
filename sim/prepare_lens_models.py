# prepare_lens_models.py: turn Euclid's published lens models into the table simulate.py draws from.
#
# Usage (from ~/ruby, with lens_env311):
#   python -I sim/prepare_lens_models.py data/zenodo_15003116/modeling_lens_mass.csv
#
# Input: modeling_lens_mass.csv from the Euclid Q1 Strong Lensing Discovery Engine release
# (Zenodo 10.5281/zenodo.15003116), one mass model per lens.
#
# Only lenses in our training split are kept, so nothing about the validation or test lenses
# (including the 4 compound lenses) reaches the simulator.
#
# Output: sim/lens_models.csv
#   theta_e    Einstein radius (arcsec), the "effective" one, which matches lenstronomy's SIE
#   q          axis ratio of the mass (1 = round)
#   gamma      strength of the outside shear
#   offset_x/y mass centre relative to the cutout centre (arcsec)
#   ring_level brightness of this real lens's ring in noise units, from the image itself

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate import PIX, ROOT, ring_level

models = pd.read_csv(sys.argv[1])
split = pd.read_csv(ROOT / "data/split.csv")
key = lambda ids: ids.astype(str).str.split("_").str[-1].str.replace("NEG", "")
models["key"], split["key"] = key(models.id_str), key(split.id)
train = split[(split.split == "train") & (split.label == 1)]
rows = train.merge(models, on="key").dropna(subset=["einstein_radius_effective_median_pdf"])

e = np.hypot(rows.ell_comps_0_median_pdf.astype(float), rows.ell_comps_1_median_pdf.astype(float))
out = pd.DataFrame(dict(
    id=rows.id,
    theta_e=rows.einstein_radius_effective_median_pdf.astype(float),
    q=(1 - e) / (1 + e),
    gamma=np.hypot(rows.shear_gamma_1_median_pdf.astype(float), rows.shear_gamma_2_median_pdf.astype(float)),
    offset_x=rows.centre_1_median_pdf.astype(float).clip(-0.3, 0.3),
    offset_y=rows.centre_0_median_pdf.astype(float).clip(-0.3, 0.3)))
out["ring_level"] = [ring_level(np.nan_to_num(np.load(ROOT / p).astype(np.float32)), t / PIX)
                     for p, t in zip(rows.path, out.theta_e)]
out = out[out.theta_e.between(0.2, 3)]                 # a failed model, or rings too big for a 10" cutout
out.to_csv(Path(__file__).resolve().parent / "lens_models.csv", index=False)

pct = [5, 25, 50, 75, 95]
print(f"{len(out)} training lenses")
for c in ["theta_e", "q", "gamma", "offset_x", "offset_y", "ring_level"]:
    print(f"  {c:11s}", np.round(np.percentile(out[c], pct), 3))
