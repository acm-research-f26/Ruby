#!/usr/bin/env python3
"""Train a head-only Zoobot lens classifier on Ruby's labeled image CSV."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("synthetic_lenses.csv"))
    parser.add_argument("--save-dir", type=Path, default=Path("model_checkpoints/ruby_synthetic_lens_head"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--encoder",
        default="hf_hub:mwalmsley/zoobot-encoder-convnext_nano",
        help="Pretrained Zoobot encoder identifier used by the existing notebook.",
    )
    args = parser.parse_args()

    try:
        import lightning as L
        import pandas as pd
        from galaxy_datasets.pytorch.galaxy_datamodule import CatalogDataModule
        from galaxy_datasets.transforms import (
            default_view_config,
            get_galaxy_transform,
            minimal_view_config,
        )
        from sklearn.model_selection import train_test_split
        from zoobot.pytorch.training import finetune
    except ImportError as error:
        parser.error(
            "Zoobot training dependencies are missing. Set up the local checkout with "
            "`python -m pip install -r requirements-zoobot.txt`. "
            f"Import error: {error}"
        )

    dataset_path = args.dataset.resolve()
    if not dataset_path.is_file():
        parser.error(f"Training CSV not found: {dataset_path}")
    catalog = pd.read_csv(dataset_path)
    required_columns = {"id", "label", "image_path"}
    missing_columns = required_columns - set(catalog.columns)
    if missing_columns:
        parser.error(f"Training CSV is missing columns: {', '.join(sorted(missing_columns))}")
    if set(catalog["label"].unique()) != {0, 1}:
        parser.error("Training labels must contain both classes encoded as 0 and 1")

    file_paths = []
    for image_path in catalog["image_path"]:
        path = Path(str(image_path))
        if not path.is_absolute():
            path = dataset_path.parent / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Training image listed in CSV is missing: {path}")
        file_paths.append(str(path))
    catalog["file_loc"] = file_paths
    catalog["id_str"] = catalog["id"].astype(str)

    train_catalog, validation_catalog = train_test_split(
        catalog,
        test_size=0.2,
        random_state=args.seed,
        stratify=catalog["label"],
    )

    train_config = default_view_config()
    train_config.output_size = 224
    train_config.greyscale = True
    train_transform = get_galaxy_transform(train_config)
    validation_config = minimal_view_config()
    validation_config.output_size = 224
    validation_config.greyscale = True
    validation_transform = get_galaxy_transform(validation_config)

    datamodule = CatalogDataModule(
        label_cols=["label"],
        train_catalog=train_catalog,
        val_catalog=validation_catalog,
        train_transform=train_transform,
        test_transform=validation_transform,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
    )
    model = finetune.FinetuneableZoobotClassifier(
        name=args.encoder,
        num_classes=2,
        label_col="label",
        training_mode="head_only",
        greyscale=True,
        seed=args.seed,
    )
    trainer = finetune.get_trainer(
        save_dir=str(args.save_dir.resolve()),
        max_epochs=args.epochs,
        patience=max(2, args.epochs),
        accelerator="cpu",
        devices=1,
    )
    L.seed_everything(args.seed, workers=True)
    print(
        f"Training on {len(train_catalog)} images; validating on {len(validation_catalog)}. "
        "These Ruby training images are synthetic."
    )
    trainer.fit(model, datamodule=datamodule)
    checkpoint_path = trainer.checkpoint_callback.best_model_path
    if not checkpoint_path:
        raise RuntimeError("Zoobot training finished without producing a best-model checkpoint")
    print(f"Best validation checkpoint: {checkpoint_path}")
    print(
        "This checkpoint is trained on synthetic Ruby images; scores are a pipeline baseline, "
        "not real-survey performance."
    )


if __name__ == "__main__":
    main()
