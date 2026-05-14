"""Shared runtime helpers for CLI entry points."""

from __future__ import annotations

from argparse import Namespace

from .datasets import DataLoaderConfig
from .engine import OptimizerConfig, SchedulerConfig, TrainingConfig
from .input_stats import load_or_compute_resnet_paper_input_stats
from .resnet import build_resnet18
from .transforms import TransformConfig


def _load_cached_input_stats(
    data_root,
    image_size: int,
    resize_size: int,
) -> dict:
    return load_or_compute_resnet_paper_input_stats(
        train_root=data_root / "train",
        image_size=image_size,
        resize_size=resize_size,
        cache_dir=data_root / ".col775_cache",
    )


def build_transform_config_from_training_args(
    args: Namespace,
) -> tuple[TransformConfig, dict | None]:
    cached_input_stats = None
    normalization_mode = args.input_normalization
    enable_pca_color_augmentation = False

    if args.train_resize_mode == "resnet_strict":
        if args.input_normalization != "imagenet":
            print(
                "ResNet strict preprocessing overrides --input-normalization and uses per-pixel mean-image subtraction."
            )
        cached_input_stats = _load_cached_input_stats(
            data_root=args.data_root,
            image_size=args.image_size,
            resize_size=args.resize_size,
        )
        normalization_mode = "per_pixel_mean"
        enable_pca_color_augmentation = True
    elif args.input_normalization == "per_pixel_mean":
        cached_input_stats = _load_cached_input_stats(
            data_root=args.data_root,
            image_size=args.image_size,
            resize_size=args.resize_size,
        )

    transform_config = TransformConfig(
        image_size=args.image_size,
        resize_size=args.resize_size,
        train_resize_max=args.train_resize_max,
        normalization=normalization_mode,
        use_cutout=args.use_cutout,
        cutout_size=args.cutout_size,
        autoaugment_policy=args.autoaugment_policy,
        train_resize_mode=args.train_resize_mode,
        mean_image=None if cached_input_stats is None else cached_input_stats["mean_image"],
        pca_eigvals=None if cached_input_stats is None else cached_input_stats["pca_eigvals"],
        pca_eigvecs=None if cached_input_stats is None else cached_input_stats["pca_eigvecs"],
        pca_alpha_std=0.1 if cached_input_stats is None else float(cached_input_stats["pca_alpha_std"]),
        use_pca_color_augmentation=enable_pca_color_augmentation,
    )
    return transform_config, cached_input_stats


def build_transform_config_from_checkpoint(
    checkpoint_metadata: dict,
    cached_input_stats: dict | None,
) -> TransformConfig:
    return TransformConfig(
        image_size=int(checkpoint_metadata["image_size"]),
        resize_size=int(checkpoint_metadata["resize_size"]),
        train_resize_max=int(checkpoint_metadata.get("train_resize_max", 480)),
        normalization=str(checkpoint_metadata["input_normalization"]),
        train_resize_mode=str(checkpoint_metadata.get("train_resize_mode", "alexnet_square")),
        mean_image=None if cached_input_stats is None else cached_input_stats.get("mean_image"),
        pca_eigvals=None if cached_input_stats is None else cached_input_stats.get("pca_eigvals"),
        pca_eigvecs=None if cached_input_stats is None else cached_input_stats.get("pca_eigvecs"),
        pca_alpha_std=0.1 if cached_input_stats is None else float(cached_input_stats.get("pca_alpha_std", 0.1)),
        use_pca_color_augmentation=bool(checkpoint_metadata.get("use_pca_color_augmentation", False)),
    )


def build_data_loader_config_from_args(args: Namespace, device) -> DataLoaderConfig:
    eval_batch_size = getattr(args, "eval_batch_size", args.batch_size)
    return DataLoaderConfig(
        batch_size=args.batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=(not args.disable_pin_memory) and device.type == "cuda",
        persistent_workers=not args.disable_persistent_workers,
    )


def build_optimizer_config_from_args(args: Namespace) -> OptimizerConfig:
    return OptimizerConfig(
        name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )


def build_scheduler_config_from_args(args: Namespace) -> SchedulerConfig:
    return SchedulerConfig(
        name=args.scheduler,
        min_lr=args.min_lr,
        gamma=args.scheduler_gamma,
        milestones=tuple(
            int(item)
            for item in args.scheduler_milestones.split(",")
            if item.strip()
        ),
    )


def build_training_config_from_args(args: Namespace) -> TrainingConfig:
    return TrainingConfig(
        epochs=args.epochs,
        amp=args.amp,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        max_grad_norm=args.max_grad_norm,
        early_stop_patience=args.early_stop_patience,
        early_stop_metric=args.early_stop_metric,
    )


def build_model_from_checkpoint_metadata(checkpoint_metadata: dict):
    return build_resnet18(
        num_classes=int(checkpoint_metadata["num_classes"]),
        norm_kind=str(checkpoint_metadata["norm"]),
        norm_eps=float(checkpoint_metadata.get("bn_eps", 1e-5)),
        bn_momentum=float(checkpoint_metadata.get("bn_momentum", 0.1)),
        group_count=int(checkpoint_metadata.get("group_count", 32)),
        shortcut_type=str(checkpoint_metadata.get("shortcut_type", "projection")),
        dropout_prob=float(checkpoint_metadata.get("dropout_prob", 0.0)),
        input_size=int(checkpoint_metadata.get("image_size", 224)),
    )
