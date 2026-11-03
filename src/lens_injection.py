"""Lens injection (HOLISMOKES-style): paint simulated lensed arcs onto REAL
Euclid non-lens backgrounds, so the resulting training image keeps real
noise/PSF/background statistics -- only the arc itself is synthetic.

Split (to keep the evaluation honest -- no data leakage): the first 6 of the
11 real non-lens images are used as injection donors for TRAINING; the
remaining 5 non-lens + all 12 real lens images are held out, untouched, for
evaluation. This shrinks the real eval set from 23 to 17, which is the honest
cost of using some real images for training instead of only evaluation.

Known simplification (time-boxed): the injected arc's flux is scaled into the
real background's fixed [0,1] intensity range rather than a rigorous
photometric flux-calibrated match between simulated and real physical units.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from dataset import load_config
from lens_sim import LensSimulator

ROOT = Path(__file__).resolve().parent.parent

N_INJECTION_DONORS = 6     # of 11 real non-lens images -> training backgrounds
N_VARIANTS_PER_DONOR = 25  # injected images generated per donor background
BRIGHTNESS_FACTOR_RANGE = (0.5, 2.5)

# Fixed scale used when the real Euclid images were originally prepared
# (prepare_real_euclid.py) -- real images live in THIS range, not the
# simulated dataset's scale_low/scale_high, so the injected arc's flux must
# be expressed relative to this one instead.
REAL_SCALE_LOW = -0.00484
REAL_SCALE_HIGH = 2.64156


def build_injected_set(real_images, real_scale_low: float, real_scale_high: float,
                        cfg: dict, seed: int = 42):
    sim = LensSimulator(cfg)
    rng = np.random.default_rng(seed)
    real_span = real_scale_high - real_scale_low

    donors = real_images[:N_INJECTION_DONORS]
    injected = []
    for donor in donors:
        for _ in range(N_VARIANTS_PER_DONOR):
            arc, _ = sim.sample_lensed_arc_only(rng)
            arc_in_real_units = arc / real_span
            brightness = rng.uniform(*BRIGHTNESS_FACTOR_RANGE)
            img = np.clip(donor + arc_in_real_units * brightness, 0.0, 1.0)
            injected.append(img.astype(np.float32))
    return np.stack(injected)


def _augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random flip/rotation -- a real image is still a valid non-lens example
    under these (no new information invented, just viewing angle)."""
    out = img
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    out = np.rot90(out, k=int(rng.integers(0, 4)))
    return out.astype(np.float32)


def build_plain_negative_set(real_images, n_variants_per_donor: int, seed: int = 99):
    """The SAME donor backgrounds used for injection, but with no arc added --
    plain negatives. Without these, every one of these real-noise/PSF
    signatures appears in training ONLY as a positive example, so the model
    can learn "this real-texture signature = lens" as a shortcut instead of
    actually keying on the arc. Including both breaks that shortcut: the only
    thing left that correlates with the label is the arc itself."""
    rng = np.random.default_rng(seed)
    out = []
    for donor in real_images:
        for _ in range(n_variants_per_donor):
            out.append(_augment(donor, rng))
    return np.stack(out)


def main():
    cfg = load_config(ROOT / "configs" / "sim_config_large.yaml")
    real = np.load(ROOT / "data" / "real_euclid_dataset.npz")
    real_images, real_labels = real["images"], real["labels"]
    nonlens_images = real_images[real_labels == 0]

    print(f"Building {N_INJECTION_DONORS * N_VARIANTS_PER_DONOR} lens-injected training images "
          f"from {N_INJECTION_DONORS} real non-lens backgrounds...")
    injected = build_injected_set(nonlens_images, REAL_SCALE_LOW, REAL_SCALE_HIGH, cfg)
    injected_labels = np.ones(len(injected), dtype=np.int64)

    print(f"Building {N_INJECTION_DONORS * N_VARIANTS_PER_DONOR} plain-negative real images "
          f"from the SAME {N_INJECTION_DONORS} backgrounds (breaks the real-texture shortcut)...")
    donors = nonlens_images[:N_INJECTION_DONORS]
    plain_negatives = build_plain_negative_set(donors, N_VARIANTS_PER_DONOR)
    plain_negative_labels = np.zeros(len(plain_negatives), dtype=np.int64)

    # Combine with the existing large simulated training set, then shuffle so
    # the real examples land throughout train AND val, not just at the tail
    # (train() below takes a sequential train/val split).
    sim_data = np.load(ROOT / "data" / "lens_dataset_large.npz")
    all_images = np.concatenate([sim_data["images"], injected, plain_negatives], axis=0)
    all_labels = np.concatenate([sim_data["labels"], injected_labels, plain_negative_labels], axis=0)

    rng = np.random.default_rng(7)
    perm = rng.permutation(len(all_images))
    all_images, all_labels = all_images[perm], all_labels[perm]

    out_path = ROOT / "data" / "lens_dataset_injected_v2.npz"
    np.savez_compressed(out_path, images=all_images, labels=all_labels)
    print(f"Saved {len(all_images)} images ({int(all_labels.sum())} lens / "
          f"{int((all_labels == 0).sum())} nonlens) to {out_path}")

    # Save the held-out real eval set (5 nonlens + 12 lens, untouched by injection)
    held_out_nonlens_idx = np.where(real_labels == 0)[0][N_INJECTION_DONORS:]
    held_out_lens_idx = np.where(real_labels == 1)[0]
    held_out_idx = np.concatenate([held_out_nonlens_idx, held_out_lens_idx])
    np.savez_compressed(
        ROOT / "data" / "real_euclid_heldout.npz",
        images=real_images[held_out_idx], labels=real_labels[held_out_idx],
    )
    print(f"Saved {len(held_out_idx)} held-out real images "
          f"({held_out_lens_idx.size} lens / {held_out_nonlens_idx.size} nonlens) for clean evaluation")


if __name__ == "__main__":
    main()
