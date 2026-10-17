#!/usr/bin/env python3
"""Run the project's Zoobot classifier on a DSPL simulation manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Manifest contains no images: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("dspl_experiment/simulation_manifest.csv"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dspl_experiment/zoobot_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(
            f"Trained Zoobot Lightning checkpoint not found: {args.checkpoint}. "
            "The notebook names a pretrained encoder, not a trained lens classifier. "
            "Train/save the notebook classifier or provide its existing .ckpt; this script will not substitute an untrained head."
        )

    missing = []
    try:
        import lightning as L
        import pandas as pd
        import torch
        from galaxy_datasets.pytorch.galaxy_datamodule import CatalogDataModule
        from galaxy_datasets.transforms import get_galaxy_transform, minimal_view_config
        from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier
    except ImportError as error:
        missing.append(str(error))
        parser.error(
            "Zoobot inference dependencies are missing. Install them with "
            "`python -m pip install -r requirements-zoobot.txt`. "
            f"Import error: {missing[0]}"
        )

    rows = read_manifest(args.manifest)
    manifest_root = args.manifest.resolve().parent
    for row in rows:
        image_path = (manifest_root / row["image_path"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image listed in manifest is missing: {image_path}")
        row["file_loc"] = str(image_path)
        row["id_str"] = row["image_id"]

    catalog = pd.DataFrame(rows)
    transform_config = minimal_view_config()
    transform_config.output_size = 224
    transform_config.greyscale = True
    inference_transform = get_galaxy_transform(transform_config)
    datamodule = CatalogDataModule(
        label_cols=["label"],
        predict_catalog=catalog,
        test_transform=inference_transform,
        batch_size=args.batch_size,
        num_workers=0,
    )
    model = FinetuneableZoobotClassifier.load_from_checkpoint(
        str(args.checkpoint.resolve()),
        map_location="cpu",
    )
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    batches = trainer.predict(model, datamodule=datamodule)
    probabilities = torch.cat([batch.detach().cpu() for batch in batches], dim=0)
    if probabilities.ndim != 2 or probabilities.shape != (len(rows), 2):
        raise ValueError(
            f"Expected two class probabilities per image; got {tuple(probabilities.shape)}"
        )
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("Zoobot returned non-finite class scores")

    lens_scores = probabilities[:, 1].tolist()
    class0_scores = probabilities[:, 0].tolist()
    predictions = probabilities.argmax(dim=1).tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [*rows[0].keys(), "class0_probability", "lens_score", "predicted_label", "checkpoint"]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row, class0_score, lens_score, prediction in zip(
            rows, class0_scores, lens_scores, predictions
        ):
            writer.writerow(
                {
                    **row,
                    "class0_probability": class0_score,
                    "lens_score": lens_score,
                    "predicted_label": prediction,
                    "checkpoint": str(args.checkpoint.resolve()),
                }
            )

    print(f"Scored {len(rows)} images with checkpoint {args.checkpoint}")
    print(f"Wrote {args.output}")
    print("lens_score is Zoobot's softmax probability for class 1 (lens), matching notebook labels.")


if __name__ == "__main__":
    main()