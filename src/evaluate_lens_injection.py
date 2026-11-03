"""Before/after: does lens injection actually help on real Euclid images?
Evaluates the original (sim-only-trained) Zoobot-Euclid head against the
lens-injection-trained one, on the 17 held-out real images neither model's
training data has ever touched."""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

from zoobot_classifier import ZoobotLensClassifier
from train_cnn import pick_device
from evaluate_all import metrics, predict

ROOT = Path(__file__).resolve().parent.parent


def load_zoobot(ckpt_name: str, device):
    model = ZoobotLensClassifier(freeze_encoder=True)
    model.head.load_state_dict(torch.load(ROOT / "outputs" / ckpt_name, map_location="cpu"))
    return model.to(device)


def main():
    device = pick_device("cpu")
    held_out = np.load(ROOT / "data" / "real_euclid_heldout.npz")
    images = torch.from_numpy(held_out["images"]).unsqueeze(1)
    y_true = held_out["labels"]

    print(f"Held-out real set: {len(y_true)} images ({int(y_true.sum())} lens / "
          f"{int((y_true == 0).sum())} nonlens) -- untouched by either model's training\n")

    for label, ckpt in [("v0 BEFORE (sim-only)", "zoobot_euclid_lens_classifier.pt"),
                         ("v1 AFTER (injection only)", "zoobot_euclid_lens_injected.pt"),
                         ("v2 AFTER (injection + plain-neg)", "zoobot_euclid_lens_injected_v2.pt")]:
        model = load_zoobot(ckpt, device)
        y_pred = predict(model, images, device)
        m = metrics(y_true, y_pred)
        print(f"{label:34s} acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  "
              f"recall={m['recall']:.3f}  f1={m['f1']:.3f}  "
              f"(tp={m['tp']} tn={m['tn']} fp={m['fp']} fn={m['fn']})")


if __name__ == "__main__":
    main()
