"""Generate and persist the simulated single-lens / non-lens dataset."""
from __future__ import annotations

import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm

from lens_sim import LensSimulator, calibrate_global_scale, fixed_linear_scale

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = ROOT / "configs" / "sim_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate_dataset(config: dict, out_path: str | Path = ROOT / "data" / "lens_dataset.npz"):
    rng = np.random.default_rng(config["seed"])
    sim = LensSimulator(config)
    n_lens = config["dataset"]["n_lens"]
    n_nonlens = config["dataset"]["n_nonlens"]

    scale_low = config["image"].get("scale_low")
    scale_high = config["image"].get("scale_high")
    if scale_low is None or scale_high is None:
        # Deterministic (fixed calibration seed, independent of `config["seed"]`)
        # so every batch generated from physically-identical config sections
        # gets the exact same intensity scale -- images stay comparable across
        # separate generate_dataset() runs/batches without manual bookkeeping.
        scale_low, scale_high = calibrate_global_scale(config)
        print(f"Calibrated fixed intensity scale: low={scale_low:.5f} high={scale_high:.5f}")

    images, labels, metas = [], [], []

    for _ in tqdm(range(n_lens), desc="simulating lenses"):
        img, meta = sim.sample_lens(rng)
        images.append(fixed_linear_scale(img, scale_low, scale_high).astype(np.float32))
        labels.append(1)
        metas.append(meta)

    for _ in tqdm(range(n_nonlens), desc="simulating non-lenses"):
        img, meta = sim.sample_nonlens(rng)
        images.append(fixed_linear_scale(img, scale_low, scale_high).astype(np.float32))
        labels.append(0)
        metas.append(meta)

    images = np.stack(images)
    labels = np.array(labels, dtype=np.int64)

    perm = rng.permutation(len(images))
    images, labels = images[perm], labels[perm]
    metas = [metas[i] for i in perm]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, images=images, labels=labels)
    return images, labels, metas


if __name__ == "__main__":
    cfg = load_config()
    images, labels, metas = generate_dataset(cfg)
    print(f"Saved {len(images)} images ({labels.sum()} lenses, {(labels == 0).sum()} non-lenses) "
          f"to data/lens_dataset.npz")
