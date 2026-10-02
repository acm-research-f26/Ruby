"""DeepDream on the Zoobot-based classifier: 100x100 grayscale seed -> gradient
ascent through the classifier's own preprocessing (upsample to 224, replicate
to 3ch, ImageNet-normalize) and into the frozen pretrained ConvNeXt encoder,
back out to pixel space. Since the wrapper always resizes to 224 internally
regardless of the input's own size, multi-octave DeepDream (which varies the
100x100-space seed's resolution) works fine here, unlike the fixed-size ViT.

Two targets:
  - "head": the classifier's own lens-logit (what makes Zoobot's head call
    something a lens, on top of general galaxy-morphology features)
  - "encoder.stages.2": a mid-level ConvNeXt stage (320ch, 14x14 at 224 input)
    -- class-agnostic, shows what Zoobot's *pretrained* features (learned
    from millions of real galaxies, not our sims) respond to.
"""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from PIL import Image

from zoobot_classifier import ZoobotLensClassifier
from train_cnn import pick_device, load_dataset
from deepdream import DeepDreamer

ROOT = Path(__file__).resolve().parent.parent


def load_zoobot(ckpt_path=ROOT / "outputs" / "zoobot_lens_classifier.pt"):
    model = ZoobotLensClassifier(freeze_encoder=True)
    model.head.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model


def save_gray(tensor: torch.Tensor, path: Path):
    arr = tensor.squeeze().numpy()
    arr = np.flipud(arr)
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L").save(path)


def main():
    device = pick_device("cpu")  # avoid MPS odd-size issues seen with octave resizing
    model = load_zoobot()

    images, labels = load_dataset(ROOT / "data" / "lens_dataset.npz")
    lens_img = torch.from_numpy(images[labels == 1][0]).view(1, 1, *images.shape[1:])
    nonlens_img = torch.from_numpy(images[labels == 0][0]).view(1, 1, *images.shape[1:])
    noise_img = torch.rand(1, 1, *images.shape[1:])
    seeds = {"noise": noise_img, "real_lens": lens_img, "real_nonlens": nonlens_img}

    out_dir = ROOT / "outputs" / "deepdream_zoobot"
    out_dir.mkdir(parents=True, exist_ok=True)

    reg_kwargs = dict(tv_weight=0.02, grad_blur_sigma=0.6,
                       image_blur_every=10, image_blur_sigma=0.4)

    # 1) maximize the classifier's lens logit
    dreamer_head = DeepDreamer(model, layer_name="head.1", device=device)
    for name, seed in seeds.items():
        result = dreamer_head.dream(seed, channel=None, iterations=40, lr=0.08,
                                     octaves=3, octave_scale=1.4, jitter=4, **reg_kwargs)
        save_gray(result, out_dir / f"head_maximize_{name}.png")
        print(f"saved head_maximize_{name}.png")

    # 2) class-agnostic texture on a mid-level pretrained ConvNeXt stage
    dreamer_stage = DeepDreamer(model, layer_name="encoder.stages.2", device=device)
    for name, seed in seeds.items():
        result = dreamer_stage.dream(seed, channel=None, iterations=40, lr=0.08,
                                      octaves=3, octave_scale=1.4, jitter=4, **reg_kwargs)
        save_gray(result, out_dir / f"stage2_texture_{name}.png")
        print(f"saved stage2_texture_{name}.png")


if __name__ == "__main__":
    main()
