"""Training and evaluation engine for image classification experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .checkpoint import save_checkpoint
from .metrics import AverageMeter, ClassificationMeter, safe_float
from .normalization import clamp_bin_parameters
from .plotting import plot_history
from .utils import ensure_dir, save_json, save_yaml


@dataclass
class OptimizerConfig:
    name: str = "sgd"
    lr: float = 0.05
    weight_decay: float = 1e-4
    momentum: float = 0.9
    nesterov: bool = False
    bin_rho_lr_scale: float = 10.0


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    min_lr: float = 1e-6
    gamma: float = 0.1
    milestones: tuple[int, ...] = (60, 80, 90)


@dataclass
class TrainingConfig:
    epochs: int = 100
    amp: bool = True
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    max_grad_norm: float = 0.0
    early_stop_patience: int = 0
    early_stop_metric: str = "accuracy"


class EarlyStopper:
    def __init__(self, patience: int, mode: str = "max"):
        self.patience = max(int(patience), 0)
        self.mode = str(mode).lower()
        if self.mode not in {"max", "min"}:
            raise ValueError(f"Unsupported early stopping mode: {mode}")
        self.best_score = -math.inf if self.mode == "max" else math.inf
        self.bad_epochs = 0

    def step(self, score: float) -> bool:
        if self.patience <= 0:
            return False

        improved = score > self.best_score if self.mode == "max" else score < self.best_score
        if improved:
            self.best_score = float(score)
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    rho_parameters: list[nn.Parameter] = []
    non_rho_parameters: list[nn.Parameter] = []

    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter_name.endswith(".rho"):
            rho_parameters.append(parameter)
        else:
            non_rho_parameters.append(parameter)

    parameter_groups: list[dict[str, Any]] = [
        {"params": non_rho_parameters, "lr": config.lr, "weight_decay": config.weight_decay}
    ]
    if rho_parameters:
        parameter_groups.append(
            {
                "params": rho_parameters,
                "lr": config.lr * config.bin_rho_lr_scale,
                "weight_decay": 0.0,
            }
        )

    optimizer_name = config.name.lower()
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameter_groups,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov,
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(parameter_groups, lr=config.lr, weight_decay=config.weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameter_groups,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    if optimizer_name == "rmsprop":
        return torch.optim.RMSprop(
            parameter_groups,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
    epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    scheduler_name = config.name.lower()
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(epochs), 1),
            eta_min=config.min_lr,
        )
    if scheduler_name == "multistep":
        milestones = sorted(set(int(m) for m in config.milestones if 0 < int(m) < epochs))
        if not milestones:
            milestones = [max(1, int(0.6 * epochs)), max(1, int(0.8 * epochs))]
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=config.gamma,
        )
    raise ValueError(f"Unsupported scheduler: {config.name}")


def mixup_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0.0:
        return images, targets, targets, 1.0

    mix_ratio = float(np.random.beta(alpha, alpha))
    shuffled_indices = torch.randperm(images.size(0), device=images.device)
    mixed_images = mix_ratio * images + (1.0 - mix_ratio) * images[shuffled_indices]
    return mixed_images, targets, targets[shuffled_indices], mix_ratio


def mixed_cross_entropy(
    logits: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    return lam * F.cross_entropy(
        logits, targets_a, label_smoothing=label_smoothing
    ) + (1.0 - lam) * F.cross_entropy(
        logits, targets_b, label_smoothing=label_smoothing
    )


def run_epoch(
    model: nn.Module,
    data_loader,
    device: torch.device,
    num_classes: int,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp: bool = False,
    label_smoothing: float = 0.0,
    mixup_alpha: float = 0.0,
    max_grad_norm: float = 0.0,
) -> dict[str, float]:
    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()

    loss_tracker = AverageMeter()
    classification_tracker = ClassificationMeter(num_classes=num_classes)
    autocast_enabled = bool(amp and device.type == "cuda")
    autocast_dtype = torch.float16

    for images, targets, _ in data_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = targets.size(0)

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            images, primary_targets, secondary_targets, mix_ratio = mixup_batch(images, targets, mixup_alpha)
        else:
            primary_targets = secondary_targets = targets
            mix_ratio = 1.0

        with torch.set_grad_enabled(is_training):
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                logits = model(images)
                if is_training and mixup_alpha > 0.0:
                    loss = mixed_cross_entropy(
                        logits,
                        primary_targets,
                        secondary_targets,
                        mix_ratio,
                        label_smoothing=label_smoothing,
                    )
                else:
                    loss = F.cross_entropy(
                        logits,
                        targets,
                        label_smoothing=label_smoothing,
                    )

            if not torch.isfinite(loss):
                phase = "training" if is_training else "evaluation"
                raise RuntimeError(
                    f"Encountered non-finite loss during {phase}. "
                    "This usually indicates optimizer divergence or invalid inputs."
                )

            if is_training:
                assert optimizer is not None
                assert scaler is not None
                scaler.scale(loss).backward()
                if max_grad_norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                clamp_bin_parameters(model)

        predictions = logits.argmax(dim=1)
        classification_tracker.update(predictions, targets)
        loss_tracker.update(loss.item(), batch_size)

    epoch_metrics = classification_tracker.compute()
    epoch_metrics["loss"] = safe_float(loss_tracker.average)
    return epoch_metrics


def evaluate_model(
    model: nn.Module,
    data_loader,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    return run_epoch(
        model=model,
        data_loader=data_loader,
        device=device,
        num_classes=num_classes,
        optimizer=None,
        scaler=None,
        amp=False,
    )


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    output_dir: str | Path,
    optimizer_config: OptimizerConfig,
    scheduler_config: SchedulerConfig,
    training_config: TrainingConfig,
    metadata: dict[str, Any],
    start_epoch: int = 0,
    best_val_accuracy: float = -math.inf,
    optimizer_state_dict: dict[str, Any] | None = None,
    scheduler_state_dict: dict[str, Any] | None = None,
    history: dict[str, list[dict[str, float]]] | None = None,
    extra_checkpoint_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    optimizer = build_optimizer(model, optimizer_config)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    scheduler = build_scheduler(
        optimizer=optimizer,
        config=scheduler_config,
        epochs=training_config.epochs,
    )
    if scheduler is not None and scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=bool(training_config.amp and device.type == "cuda"),
    )
    history = history or {"train": [], "val": []}
    early_stop_metric = str(training_config.early_stop_metric).lower()
    if early_stop_metric not in {"accuracy", "loss"}:
        raise ValueError(
            f"Unsupported early stopping metric: {training_config.early_stop_metric}"
        )
    early_stop_mode = "max" if early_stop_metric == "accuracy" else "min"
    early_stopper = EarlyStopper(training_config.early_stop_patience, mode=early_stop_mode)
    if early_stop_metric == "accuracy":
        if math.isfinite(best_val_accuracy):
            early_stopper.best_score = float(best_val_accuracy)
        elif history and history.get("val"):
            early_stopper.best_score = max(
                float(entry["accuracy"]) for entry in history["val"]
            )
    elif history and history.get("val"):
        early_stopper.best_score = min(float(entry["loss"]) for entry in history["val"])

    config_payload = {
        "optimizer": asdict(optimizer_config),
        "scheduler": {
            **asdict(scheduler_config),
            "milestones": list(scheduler_config.milestones),
        },
        "training": asdict(training_config),
        "metadata": metadata,
    }
    save_yaml(output_dir / "config.yaml", config_payload)

    num_classes = int(metadata["num_classes"])

    for epoch in range(start_epoch, training_config.epochs):
        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            device=device,
            num_classes=num_classes,
            optimizer=optimizer,
            scaler=scaler,
            amp=training_config.amp,
            label_smoothing=training_config.label_smoothing,
            mixup_alpha=training_config.mixup_alpha,
            max_grad_norm=training_config.max_grad_norm,
        )
        val_metrics = evaluate_model(
            model=model,
            data_loader=val_loader,
            device=device,
            num_classes=num_classes,
        )

        if scheduler is not None:
            scheduler.step()

        train_entry = {"epoch": epoch + 1, **train_metrics}
        val_entry = {"epoch": epoch + 1, **val_metrics}
        history["train"].append(train_entry)
        history["val"].append(val_entry)

        val_accuracy = val_metrics["accuracy"]
        is_best = val_accuracy > best_val_accuracy
        best_val_accuracy = max(best_val_accuracy, val_accuracy)

        checkpoint_state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_val_accuracy": best_val_accuracy,
            "history": history,
            "metadata": metadata,
            "config": config_payload,
        }
        if extra_checkpoint_state is not None:
            checkpoint_state.update(extra_checkpoint_state)
        save_checkpoint(checkpoint_state, output_dir=output_dir, is_best=is_best)
        save_json(output_dir / "history.json", history)
        plot_history(history, output_dir / "curves.png")

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:03d}/{training_config.epochs:03d} | "
            f"lr={current_lr:.6f} | "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} "
            f"micro_f1={train_metrics['micro_f1']:.4f} macro_f1={train_metrics['macro_f1']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
            f"micro_f1={val_metrics['micro_f1']:.4f} macro_f1={val_metrics['macro_f1']:.4f}"
        )

        early_stop_score = val_accuracy if early_stop_metric == "accuracy" else val_metrics["loss"]
        if early_stopper.step(float(early_stop_score)):
            print(
                f"Early stopping triggered after epoch {epoch + 1} based on validation {early_stop_metric}."
            )
            break

    summary = {
        "best_val_accuracy": best_val_accuracy,
        "final_train": history["train"][-1] if history["train"] else None,
        "final_val": history["val"][-1] if history["val"] else None,
        "history": history,
    }
    save_json(output_dir / "summary.json", summary)
    return summary
