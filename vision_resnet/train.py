"""Train ResNet-18 image classification variants."""

from __future__ import annotations

import argparse
from pathlib import Path

from vision.datasets import DataLoaderConfig, build_dataloaders
from vision.engine import (
    fit,
)
from vision.resnet import build_resnet18
from vision.runtime import (
    build_data_loader_config_from_args,
    build_optimizer_config_from_args,
    build_scheduler_config_from_args,
    build_training_config_from_args,
    build_transform_config_from_training_args,
)
from vision.transforms import TransformConfig, build_eval_transform, build_train_transform
from vision.utils import get_device, load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root containing train/ and val/.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints and logs.")
    parser.add_argument(
        "--norm",
        type=str,
        default="baseline",
        choices=["baseline", "nn", "bn", "in", "bin", "ln", "gn"],
        help="Normalization variant.",
    )
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam", "adamw", "rmsprop"],
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "multistep", "none"],
    )
    parser.add_argument("--scheduler-gamma", type=float, default=0.1)
    parser.add_argument("--scheduler-milestones", type=str, default="60,80,90")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--dropout-prob", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA.")
    parser.add_argument("--disable-pin-memory", action="store_true")
    parser.add_argument("--disable-persistent-workers", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--train-resize-max", type=int, default=480)
    parser.add_argument(
        "--train-resize-mode",
        type=str,
        default="alexnet_square",
        choices=["alexnet_square", "resnet_paper", "resnet_strict"],
        help="Training resize recipe. `resnet_strict` also enables PCA lighting and per-pixel mean-image subtraction.",
    )
    parser.add_argument(
        "--input-normalization",
        type=str,
        default="imagenet",
        choices=["imagenet", "mean_only", "per_pixel_mean", "none"],
    )
    parser.add_argument("--use-cutout", action="store_true")
    parser.add_argument("--cutout-size", type=int, default=56)
    parser.add_argument(
        "--autoaugment-policy",
        type=str,
        default="none",
        choices=["none", "imagenet"],
        help="Apply an AutoAugment policy during training. Defaults to off.",
    )
    parser.add_argument("--group-count", type=int, default=32)
    parser.add_argument(
        "--shortcut-type",
        type=str,
        default="projection",
        choices=["projection", "identity_pad", "always_projection"],
        help="Shortcut type. Defaults to learned 1x1 projection only when shapes change.",
    )
    parser.add_argument("--bn-eps", type=float, default=1e-5)
    parser.add_argument("--bn-momentum", type=float, default=0.1)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument(
        "--early-stop-metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "loss"],
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    transform_config, input_stats_cache = build_transform_config_from_training_args(args)
    train_transform = build_train_transform(transform_config)
    eval_transform = build_eval_transform(transform_config)

    data_config = build_data_loader_config_from_args(args, device)
    data_bundle = build_dataloaders(
        data_root=args.data_root,
        train_transform=train_transform,
        eval_transform=eval_transform,
        config=data_config,
    )

    if args.num_classes != len(data_bundle["classes"]):
        print(
            f"Adjusting num_classes from {args.num_classes} to "
            f"{len(data_bundle['classes'])} based on dataset folders."
        )
        args.num_classes = len(data_bundle["classes"])

    model = build_resnet18(
        num_classes=args.num_classes,
        norm_kind=args.norm,
        norm_eps=args.bn_eps,
        bn_momentum=args.bn_momentum,
        group_count=args.group_count,
        shortcut_type=args.shortcut_type,
        dropout_prob=args.dropout_prob,
        input_size=args.image_size,
    ).to(device)

    optimizer_config = build_optimizer_config_from_args(args)
    scheduler_config = build_scheduler_config_from_args(args)
    training_config = build_training_config_from_args(args)

    checkpoint_metadata = {
        "norm": args.norm,
        "num_classes": args.num_classes,
        "class_names": data_bundle["classes"],
        "data_root": str(args.data_root),
        "image_size": args.image_size,
        "resize_size": args.resize_size,
        "train_resize_max": args.train_resize_max,
        "train_resize_mode": args.train_resize_mode,
        "input_normalization": transform_config.normalization,
        "use_cutout": args.use_cutout,
        "cutout_size": args.cutout_size,
        "autoaugment_policy": args.autoaugment_policy,
        "use_pca_color_augmentation": transform_config.use_pca_color_augmentation,
        "dropout_prob": args.dropout_prob,
        "group_count": args.group_count,
        "shortcut_type": args.shortcut_type,
        "bn_eps": args.bn_eps,
        "bn_momentum": args.bn_momentum,
    }

    start_epoch = 0
    best_val_accuracy = float("-inf")
    optimizer_state_dict = None
    scheduler_state_dict = None
    history = None

    if args.resume is not None:
        resume_checkpoint = load_checkpoint(args.resume, map_location="cpu")
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        start_epoch = int(resume_checkpoint.get("epoch", 0))
        best_val_accuracy = float(resume_checkpoint.get("best_val_accuracy", float("-inf")))
        optimizer_state_dict = resume_checkpoint.get("optimizer_state_dict")
        scheduler_state_dict = resume_checkpoint.get("scheduler_state_dict")
        history = resume_checkpoint.get("history")
        print(f"Resuming from {args.resume} at epoch {start_epoch}.")

    fit(
        model=model,
        train_loader=data_bundle["train_loader"],
        val_loader=data_bundle["val_loader"],
        device=device,
        output_dir=args.output_dir,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        training_config=training_config,
        metadata=checkpoint_metadata,
        start_epoch=start_epoch,
        best_val_accuracy=best_val_accuracy,
        optimizer_state_dict=optimizer_state_dict,
        scheduler_state_dict=scheduler_state_dict,
        history=history,
        extra_checkpoint_state={"input_stats": input_stats_cache} if input_stats_cache is not None else None,
    )


if __name__ == "__main__":
    main()
