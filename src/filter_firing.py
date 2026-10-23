"""Step 3: which Zoobot (Euclid encoder) filters fire for small-radius arcs
(theta_E in [0.70, 1.00] arcsec) vs. easily-detected larger arcs, and does the
answer change across ConvNeXt stages (80/160/320/640 filters, 56/28/14/7 px)?

For each stage: run both image sets through the frozen encoder, take each
filter's mean activation per image, average across the set to get one
activation value per filter per config, then rank and compare the top 10% of
filters ("fires") between configs via Jaccard overlap.
"""
from __future__ import annotations

import json
import numpy as np
import torch
from pathlib import Path

from dataset import load_config
from lens_sim import LensSimulator, fixed_linear_scale
from zoobot_classifier import ZoobotLensClassifier
from train_cnn import pick_device

ROOT = Path(__file__).resolve().parent.parent

STAGES = ["stages.0", "stages.1", "stages.2", "stages.3"]
TOP_FRACTION = 0.10  # compare the top 10% most-active filters at each stage

SMALL_RING_RANGE = (0.70, 1.00)   # arcsec -- this week's Friday deliverable
BASELINE_RING_RANGE = (1.60, 2.40)  # arcsec -- clearly larger, for contrast

N_PER_CONFIG = 100


def generate_ring_size_set(sim: LensSimulator, rng: np.random.Generator,
                            theta_e_range: tuple[float, float], n: int,
                            scale_low: float, scale_high: float) -> torch.Tensor:
    images = []
    for _ in range(n):
        theta_e = rng.uniform(*theta_e_range)
        img, _ = sim.sample_lens_with_ring_size(rng, theta_e)
        images.append(fixed_linear_scale(img, scale_low, scale_high).astype(np.float32))
    arr = np.stack(images)
    return torch.from_numpy(arr).unsqueeze(1)  # (N, 1, H, W)


@torch.no_grad()
def mean_filter_activations(model: ZoobotLensClassifier, images: torch.Tensor,
                             device, batch_size: int = 32) -> dict[str, np.ndarray]:
    """Returns {stage_name: (C,) array}, each filter's mean activation across
    every image and every spatial location in that stage's feature map."""
    model.eval()
    captured = {}

    def hook(name):
        def fn(module, inp, out):
            # out: (B, C, H, W) -> mean over batch and spatial dims -> (C,)
            captured.setdefault(name, []).append(out.mean(dim=(0, 2, 3)).cpu())
        return fn

    handles = [model.encoder.get_submodule(s).register_forward_hook(hook(s)) for s in STAGES]
    try:
        for i in range(0, len(images), batch_size):
            xb = images[i:i + batch_size].to(device)
            x = model.preprocess(xb)
            model.encoder(x)
    finally:
        for h in handles:
            h.remove()

    # Each hook call recorded one (C,) vector per batch; average those,
    # weighted by batch size, to get one (C,) vector per stage overall.
    out = {}
    for name, chunks in captured.items():
        weights = torch.tensor([min(batch_size, len(images) - i)
                                 for i in range(0, len(images), batch_size)], dtype=torch.float32)
        stacked = torch.stack(chunks)  # (n_batches, C)
        out[name] = (stacked * weights[:, None]).sum(0) / weights.sum()
        out[name] = out[name].numpy()
    return out


def top_filters(activations: np.ndarray, fraction: float) -> set[int]:
    k = max(1, int(round(len(activations) * fraction)))
    return set(np.argsort(activations)[-k:].tolist())


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    cfg = load_config(ROOT / "configs" / "sim_config_large.yaml")
    scale_low = cfg["image"]["scale_low"]
    scale_high = cfg["image"]["scale_high"]

    sim = LensSimulator(cfg)
    rng = np.random.default_rng(2026)

    print(f"Generating {N_PER_CONFIG} images with theta_E in {SMALL_RING_RANGE}\" (small/Friday config)...")
    small_images = generate_ring_size_set(sim, rng, SMALL_RING_RANGE, N_PER_CONFIG, scale_low, scale_high)
    print(f"Generating {N_PER_CONFIG} images with theta_E in {BASELINE_RING_RANGE}\" (baseline config)...")
    baseline_images = generate_ring_size_set(sim, rng, BASELINE_RING_RANGE, N_PER_CONFIG, scale_low, scale_high)

    device = pick_device("cpu")  # consistent, avoids MPS odd-size issues seen before
    model = ZoobotLensClassifier(freeze_encoder=True).to(device)
    model.head.load_state_dict(torch.load(
        ROOT / "outputs" / "zoobot_euclid_lens_classifier.pt", map_location="cpu"))

    print("Extracting per-filter mean activations (small-radius set)...")
    small_acts = mean_filter_activations(model, small_images, device)
    print("Extracting per-filter mean activations (baseline set)...")
    baseline_acts = mean_filter_activations(model, baseline_images, device)

    report = {}
    print("\n=== Filter-firing comparison by stage ===")
    for stage in STAGES:
        small_top = top_filters(small_acts[stage], TOP_FRACTION)
        baseline_top = top_filters(baseline_acts[stage], TOP_FRACTION)
        overlap = small_top & baseline_top
        jac = jaccard(small_top, baseline_top)
        n_filters = len(small_acts[stage])
        print(f"\n{stage} ({n_filters} filters, top {int(round(n_filters*TOP_FRACTION))} compared):")
        print(f"  small-radius-only fires : {sorted(small_top - baseline_top)}")
        print(f"  baseline-only fires     : {sorted(baseline_top - small_top)}")
        print(f"  overlap (fires in both) : {sorted(overlap)}")
        print(f"  Jaccard overlap         : {jac:.3f}")
        report[stage] = {
            "n_filters": n_filters,
            "top_k": int(round(n_filters * TOP_FRACTION)),
            "small_radius_top_filters": sorted(small_top),
            "baseline_top_filters": sorted(baseline_top),
            "overlap_filters": sorted(overlap),
            "small_radius_only": sorted(small_top - baseline_top),
            "baseline_only": sorted(baseline_top - small_top),
            "jaccard_overlap": jac,
        }

    out_path = ROOT / "outputs" / "filter_firing_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved full report to {out_path}")


if __name__ == "__main__":
    main()
