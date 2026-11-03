"""Retrain on the v2 training set: lens-injected positives PLUS the same
donor backgrounds as plain negatives, to break the real-texture shortcut
diagnosed from v1's results (see evaluate_lens_injection.py's findings)."""
from __future__ import annotations

from pathlib import Path

from dataset import load_config
from train_zoobot import train

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    cfg = load_config(ROOT / "configs" / "sim_config_large.yaml")
    model, history = train(
        cfg,
        data_path=ROOT / "data" / "lens_dataset_injected_v2.npz",
        ckpt_path=ROOT / "outputs" / "zoobot_euclid_lens_injected_v2.pt",
        epochs=40,
    )
