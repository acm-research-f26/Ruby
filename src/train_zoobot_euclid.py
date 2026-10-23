"""Step 2: train the lens head on top of the Euclid-pretrained Zoobot encoder
(mwalmsley/zoobot-encoder-euclid, wired in via zoobot_classifier.ENCODER_NAME).

Saved separately from the earlier general-purpose-Zoobot checkpoint
(outputs/zoobot_lens_classifier.pt) so we can still compare the two encoders
head-to-head later, rather than silently overwriting that result.
"""
from __future__ import annotations

from pathlib import Path

from dataset import load_config
from train_zoobot import train

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    cfg = load_config(ROOT / "configs" / "sim_config_large.yaml")
    model, history = train(
        cfg,
        data_path=ROOT / "data" / "lens_dataset_large.npz",
        ckpt_path=ROOT / "outputs" / "zoobot_euclid_lens_classifier.pt",
        epochs=40,
    )
