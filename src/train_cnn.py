"""Train SimpleLensCNN on the simulated lens / non-lens dataset."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from models import SimpleLensCNN

ROOT = Path(__file__).resolve().parent.parent


def pick_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dataset(path: str | Path = ROOT / "data" / "lens_dataset.npz"):
    data = np.load(path)
    return data["images"], data["labels"]


def train(config: dict, data_path=ROOT / "data" / "lens_dataset.npz",
          ckpt_path=ROOT / "outputs" / "simple_lens_cnn.pt", seed: int = 0):
    torch.manual_seed(seed)  # reproducible init -- see BatchNorm comment in models.py
    images, labels = load_dataset(data_path)
    images = torch.from_numpy(images).unsqueeze(1)          # (N, 1, H, W)
    labels = torch.from_numpy(labels).float().unsqueeze(1)  # (N, 1)

    n_train = int(len(images) * config["dataset"]["train_fraction"])
    x_train, x_val = images[:n_train], images[n_train:]
    y_train, y_val = labels[:n_train], labels[n_train:]

    device = pick_device(config["training"]["device"])
    model = SimpleLensCNN(in_channels=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    loss_fn = nn.BCEWithLogitsLoss()

    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    n = len(x_train)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss, correct = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x_train[idx].to(device), y_train[idx].to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
            correct += ((logits.sigmoid() > 0.5).float() == yb).sum().item()
        train_loss = epoch_loss / n
        train_acc = correct / n

        model.eval()
        with torch.no_grad():
            xb, yb = x_val.to(device), y_val.to(device)
            logits = model(xb)
            val_loss = loss_fn(logits, yb).item()
            val_acc = ((logits.sigmoid() > 0.5).float() == yb).float().mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"epoch {epoch + 1:3d}/{epochs}  train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.3f}  val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    return model, history


if __name__ == "__main__":
    from dataset import load_config
    cfg = load_config()
    train(cfg)
