"""Train the linear head on top of the frozen Zoobot encoder."""
from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path

from zoobot_classifier import ZoobotLensClassifier
from train_cnn import pick_device, load_dataset

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def _embed_all(model: ZoobotLensClassifier, images: torch.Tensor, device, batch_size: int = 64):
    """One forward pass through the frozen encoder per image, cached -- the
    head is the only trainable part, so there's no need to re-run the (much
    more expensive) encoder on every epoch."""
    model.eval()
    feats = []
    for i in range(0, len(images), batch_size):
        xb = images[i:i + batch_size].to(device)
        x = model.preprocess(xb)
        feats.append(model.encoder(x).cpu())
    return torch.cat(feats)


def train(config: dict, data_path=ROOT / "data" / "lens_dataset.npz",
          ckpt_path=ROOT / "outputs" / "zoobot_lens_classifier.pt",
          epochs: int = 40, lr: float = 1e-3, seed: int = 0):
    torch.manual_seed(seed)
    images, labels = load_dataset(data_path)
    images = torch.from_numpy(images).unsqueeze(1)
    labels = torch.from_numpy(labels).float().unsqueeze(1)

    n_train = int(len(images) * config["dataset"]["train_fraction"])
    device = pick_device(config["training"]["device"])
    model = ZoobotLensClassifier(freeze_encoder=True).to(device)

    print(f"Embedding {len(images)} images through the frozen Zoobot encoder...")
    embeddings = _embed_all(model, images, device)
    x_train, x_val = embeddings[:n_train].to(device), embeddings[n_train:].to(device)
    y_train, y_val = labels[:n_train].to(device), labels[n_train:].to(device)

    opt = torch.optim.Adam(model.head.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    batch_size = 32
    n = len(x_train)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        model.head.train()
        perm = torch.randperm(n)
        epoch_loss, correct = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x_train[idx], y_train[idx]
            opt.zero_grad()
            logits = model.head(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
            correct += ((logits.sigmoid() > 0.5).float() == yb).sum().item()
        train_loss = epoch_loss / n
        train_acc = correct / n

        model.head.eval()
        with torch.no_grad():
            logits = model.head(x_val)
            val_loss = loss_fn(logits, y_val).item()
            val_acc = ((logits.sigmoid() > 0.5).float() == y_val).float().mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"epoch {epoch + 1:3d}/{epochs}  train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.3f}  val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.head.state_dict(), ckpt_path)
    return model, history


if __name__ == "__main__":
    from dataset import load_config
    cfg = load_config()
    train(cfg)
