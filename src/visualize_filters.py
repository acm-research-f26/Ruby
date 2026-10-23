"""Per-filter DeepDream: for each of the filters that diverged between small-
radius and baseline rings (from filter_firing.py's stages.3 results), run
gradient ascent from noise targeting ONLY that one channel, to see the
pattern it actually responds to. This is what lets us replace "filter 614"
with a real, descriptive name.
"""
from __future__ import annotations

import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from zoobot_classifier import ZoobotLensClassifier
from train_cnn import pick_device
from deepdream import DeepDreamer

ROOT = Path(__file__).resolve().parent.parent
STAGE = "encoder.stages.3"

REG_KWARGS = dict(tv_weight=0.02, grad_blur_sigma=0.6, image_blur_every=10, image_blur_sigma=0.4)


def save_gray(tensor: torch.Tensor, path: Path):
    arr = tensor.squeeze().numpy()
    arr = np.flipud(arr)
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L").save(path)


def main():
    report = json.loads((ROOT / "outputs" / "filter_firing_report.json").read_text())
    small_only = report["stages.3"]["small_radius_only"]
    baseline_only = report["stages.3"]["baseline_only"]

    device = pick_device("cpu")
    model = ZoobotLensClassifier(freeze_encoder=True).to(device)
    # Probing the encoder's own learned features -- the task head isn't involved.

    dreamer = DeepDreamer(model, layer_name=STAGE, device=device)

    out_dir = ROOT / "outputs" / "filter_portraits"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [("small_only", f) for f in small_only] + [("baseline_only", f) for f in baseline_only]
    for group, filt in targets:
        torch.manual_seed(filt)  # reproducible per-filter seed
        seed = torch.rand(1, 1, 100, 100)
        result = dreamer.dream(seed, channel=filt, iterations=40, lr=0.08,
                                octaves=3, octave_scale=1.4, jitter=4, **REG_KWARGS)
        save_gray(result, out_dir / f"{group}_filter{filt:03d}.png")
        print(f"saved {group}_filter{filt:03d}.png")


if __name__ == "__main__":
    main()
