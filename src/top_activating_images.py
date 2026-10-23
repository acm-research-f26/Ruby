"""For each target filter (the 28 that diverge between small-radius and
baseline rings at stages.3), find the real images -- from our full simulated
pool plus the 23 real Euclid images -- that activate it most strongly. This
is the grounded alternative to per-filter DeepDream (which collapsed into
indistinguishable noise): instead of synthesizing a "what this filter wants"
image, we look at which actual images it responds to most.
"""
from __future__ import annotations

import json
import numpy as np
import torch
from pathlib import Path

from zoobot_classifier import ZoobotLensClassifier
from train_cnn import pick_device

ROOT = Path(__file__).resolve().parent.parent
STAGE = "encoder.stages.3"
TOP_N = 5


def load_pool():
    """Combine the large simulated set + real Euclid images into one pool,
    keeping track of where each image came from and its label (1=lens)."""
    sim = np.load(ROOT / "data" / "lens_dataset_large.npz")
    real = np.load(ROOT / "data" / "real_euclid_dataset.npz")

    images = np.concatenate([sim["images"], real["images"]], axis=0)
    labels = np.concatenate([sim["labels"], real["labels"]], axis=0)
    sources = ["sim"] * len(sim["images"]) + ["real"] * len(real["images"])
    return images.astype(np.float32), labels, sources


@torch.no_grad()
def all_filter_activations(model: ZoobotLensClassifier, images: torch.Tensor,
                            device, batch_size: int = 64) -> np.ndarray:
    """Returns (N, C) -- every image's per-filter mean activation at STAGE."""
    model.eval()
    captured = []

    def hook(module, inp, out):
        captured.append(out.mean(dim=(2, 3)).cpu())  # (B, C), per-image (not pooled over batch)

    handle = model.get_submodule(STAGE).register_forward_hook(hook)
    try:
        for i in range(0, len(images), batch_size):
            xb = images[i:i + batch_size].to(device)
            x = model.preprocess(xb)
            model.encoder(x)
    finally:
        handle.remove()
    return torch.cat(captured).numpy()  # (N, C)


def main():
    report = json.loads((ROOT / "outputs" / "filter_firing_report.json").read_text())
    small_only = report["stages.3"]["small_radius_only"]
    baseline_only = report["stages.3"]["baseline_only"]
    target_filters = small_only + baseline_only

    print("Loading image pool (6000 sim + 23 real)...")
    images_np, labels, sources = load_pool()
    images = torch.from_numpy(images_np).unsqueeze(1)

    device = pick_device("cpu")
    model = ZoobotLensClassifier(freeze_encoder=True).to(device)

    print(f"Computing activations for all {images.shape[0]} images at {STAGE}...")
    acts = all_filter_activations(model, images, device)  # (N, 640)

    results = {}
    for filt in target_filters:
        col = acts[:, filt]
        top_idx = np.argsort(col)[-TOP_N:][::-1]
        results[str(filt)] = [
            {"index": int(i), "source": sources[i], "label": int(labels[i]), "activation": float(col[i])}
            for i in top_idx
        ]

    out_path = ROOT / "outputs" / "top_activating_images.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    # No separate image cache saved -- the preview script just reloads
    # load_pool() directly (cheap, and avoids duplicating ~240MB already on disk).


if __name__ == "__main__":
    main()
