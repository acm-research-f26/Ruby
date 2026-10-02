"""Train the small ViT classifier on the simulated lens / non-lens dataset.

ViTs have weaker inductive biases than CNNs and want more data than our ~800
images provide, so this adds light flip/rotation augmentation (lens images
have no preferred orientation) and stronger weight decay to compensate.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path

from vit_model import build_lens_vit
from train_cnn import pick_device, load_dataset

ROOT = Path(__file__).resolve().parent.parent


def augment(batch: torch.Tensor) -> torch.Tensor:
    if torch.rand(1).item() < 0.5:
        batch = batch.flip(-1)
    if torch.rand(1).item() < 0.5:
        batch = batch.flip(-2)
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        batch = torch.rot90(batch, k, dims=(-2, -1))
    return batch


def train(config: dict, data_path=ROOT / "data" / "lens_dataset.npz",
          ckpt_path=ROOT / "outputs" / "lens_vit.pt",
          epochs: int | None = None, weight_decay: float = 0.05, seed: int = 0):
    torch.manual_seed(seed)
    images, labels = load_dataset(data_path)
    img_size = images.shape[-1]
    images = torch.from_numpy(images).unsqueeze(1)          # (N, 1, H, W)
    labels = torch.from_numpy(labels).float().unsqueeze(1)  # (N, 1)

    n_train = int(len(images) * config["dataset"]["train_fraction"])
    x_train, x_val = images[:n_train], images[n_train:]
    y_train, y_val = labels[:n_train], labels[n_train:]

    device = pick_device(config["training"]["device"])
    model = build_lens_vit(img_size=img_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                             weight_decay=weight_decay)
    epochs = epochs or config["training"]["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    batch_size = config["training"]["batch_size"]
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    n = len(x_train)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss, correct = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x_train[idx].to(device), y_train[idx].to(device)
            xb = augment(xb)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
            correct += ((logits.sigmoid() > 0.5).float() == yb).sum().item()
        sched.step()
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
    torch.save({"state_dict": model.state_dict(), "img_size": img_size}, ckpt_path)
    return model, history


if __name__ == "__main__":
    from dataset import load_config
    cfg = load_config()
    train(cfg)
