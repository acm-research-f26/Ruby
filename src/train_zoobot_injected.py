"""Retrain the Euclid-backed Zoobot lens head on the lens-injection-augmented
training set (data/lens_dataset_injected.npz), for before/after comparison
against the original zoobot_euclid_lens_classifier.pt on held-out real images."""
from __future__ import annotations

from pathlib import Path

from dataset import load_config
from train_zoobot import train

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    cfg = load_config(ROOT / "configs" / "sim_config_large.yaml")
    model, history = train(
        cfg,
        data_path=ROOT / "data" / "lens_dataset_injected.npz",
        ckpt_path=ROOT / "outputs" / "zoobot_euclid_lens_injected.pt",
        epochs=40,
    )
