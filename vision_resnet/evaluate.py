"""Evaluate a saved checkpoint on train or validation data."""

from __future__ import annotations

import argparse
from pathlib import Path

from vision.datasets import build_dataloaders
from vision.engine import evaluate_model
from vision.runtime import (
    build_data_loader_config_from_args,
    build_model_from_checkpoint_metadata,
    build_transform_config_from_checkpoint,
)
from vision.transforms import build_eval_transform, build_train_transform
from vision.utils import get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--disable-pin-memory", action="store_true")
    parser.add_argument("--disable-persistent-workers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    checkpoint_metadata = checkpoint["metadata"]
    cached_input_stats = checkpoint.get("input_stats")
    device = get_device(args.device)
    transform_config = build_transform_config_from_checkpoint(
        checkpoint_metadata=checkpoint_metadata,
        cached_input_stats=cached_input_stats,
    )
    train_transform = build_train_transform(transform_config)
    eval_transform = build_eval_transform(transform_config)
    data_config = build_data_loader_config_from_args(args, device)
    data_bundle = build_dataloaders(
        data_root=args.data_root,
        train_transform=train_transform,
        eval_transform=eval_transform,
        config=data_config,
    )

    model = build_model_from_checkpoint_metadata(checkpoint_metadata)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    evaluation_loader = data_bundle["train_loader"] if args.split == "train" else data_bundle["val_loader"]
    evaluation_metrics = evaluate_model(
        model=model,
        data_loader=evaluation_loader,
        device=device,
        num_classes=int(checkpoint_metadata["num_classes"]),
    )
    print(
        f"{args.split} | loss={evaluation_metrics['loss']:.4f} "
        f"acc={evaluation_metrics['accuracy']:.4f} micro_f1={evaluation_metrics['micro_f1']:.4f} "
        f"macro_f1={evaluation_metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
