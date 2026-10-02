"""Evaluate CNN, ViT, and Zoobot-head classifiers on every test set we have:
the two lenstronomy batches (balanced + 10/90 imbalanced) and the real Euclid
Q1 cutouts. Reports accuracy/precision/recall/F1 + confusion matrix per
(model, dataset) pair."""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

from models import SimpleLensCNN
from vit_model import build_lens_vit
from zoobot_classifier import ZoobotLensClassifier
from train_cnn import pick_device

ROOT = Path(__file__).resolve().parent.parent


def load_npz(path):
    d = np.load(path)
    images = torch.from_numpy(d["images"]).unsqueeze(1)
    labels = torch.from_numpy(d["labels"]).long()
    return images, labels


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return dict(n=len(y_true), tp=tp, tn=tn, fp=fp, fn=fn,
                accuracy=acc, precision=precision, recall=recall, f1=f1)


@torch.no_grad()
def predict(model, images: torch.Tensor, device, batch_size: int = 64) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(images), batch_size):
        xb = images[i:i + batch_size].to(device)
        logits = model(xb)
        preds.append((logits.sigmoid() > 0.5).long().cpu())
    return torch.cat(preds).squeeze(-1).numpy()


def load_models(device):
    cnn = SimpleLensCNN(in_channels=1)
    cnn.load_state_dict(torch.load(ROOT / "outputs" / "simple_lens_cnn.pt", map_location="cpu"))
    cnn.to(device)

    vit_ckpt = torch.load(ROOT / "outputs" / "lens_vit.pt", map_location="cpu", weights_only=False)
    vit = build_lens_vit(img_size=vit_ckpt["img_size"])
    vit.load_state_dict(vit_ckpt["state_dict"])
    vit.to(device)

    zoobot = ZoobotLensClassifier(freeze_encoder=True)
    zoobot.head.load_state_dict(torch.load(ROOT / "outputs" / "zoobot_lens_classifier.pt", map_location="cpu"))
    zoobot.to(device)

    return {"CNN": cnn, "ViT": vit, "Zoobot-head": zoobot}


def main():
    device = pick_device("cpu")  # avoids MPS adaptive-pool/odd-size issues seen earlier
    models = load_models(device)

    datasets = {
        "sim_batch1 (balanced 400/400)": ROOT / "data" / "lens_dataset.npz",
        "sim_batch2 (10%/90% imbalanced)": ROOT / "data" / "lens_dataset_batch2.npz",
        "sim_large_heldout (last 10%, unseen in training)": ROOT / "data" / "lens_dataset_large.npz",
        "real_euclid_q1 (23 real cutouts)": ROOT / "data" / "real_euclid_dataset.npz",
    }

    lines = []
    for ds_name, path in datasets.items():
        if not path.exists():
            continue
        images, labels = load_npz(path)
        if ds_name.startswith("sim_large_heldout"):
            # All 3 models trained on sim_config_large.yaml with train_fraction=0.9
            # and no shuffling at train time -- slice the same held-out tail here.
            n_train = int(len(images) * 0.9)
            images, labels = images[n_train:], labels[n_train:]
        y_true = labels.numpy()
        lines.append(f"\n=== {ds_name}  (n={len(y_true)}, {int(y_true.sum())} lens / {int((y_true==0).sum())} nonlens) ===")
        for model_name, model in models.items():
            y_pred = predict(model, images, device)
            m = metrics(y_true, y_pred)
            lines.append(
                f"  {model_name:12s}  acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  "
                f"recall={m['recall']:.3f}  f1={m['f1']:.3f}  "
                f"(tp={m['tp']} tn={m['tn']} fp={m['fp']} fn={m['fn']})"
            )

    report = "\n".join(lines)
    print(report)
    out_path = ROOT / "outputs" / "evaluation_summary.txt"
    out_path.write_text(report)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
