"""Classification metrics for model evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


class ClassificationMeter:
    """Accumulate a confusion matrix and derive classification metrics."""

    def __init__(self, num_classes: int):
        self.num_classes = int(num_classes)
        self.confusion = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64
        )

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        predictions = predictions.detach().view(-1).to(torch.int64).cpu()
        targets = targets.detach().view(-1).to(torch.int64).cpu()
        valid = (targets >= 0) & (targets < self.num_classes)
        targets = targets[valid]
        predictions = predictions[valid]

        indices = targets * self.num_classes + predictions
        batch_confusion = torch.bincount(
            indices, minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)
        self.confusion += batch_confusion

    def compute(self) -> dict[str, float]:
        confusion = self.confusion.to(torch.float64)
        true_positives = confusion.diag()
        support = confusion.sum(dim=1)
        predicted = confusion.sum(dim=0)

        total = confusion.sum().item()
        correct = true_positives.sum().item()
        accuracy = correct / total if total > 0 else 0.0

        false_positives = predicted - true_positives
        false_negatives = support - true_positives

        precision_macro = torch.where(
            predicted > 0,
            true_positives / predicted.clamp_min(1.0),
            torch.zeros_like(true_positives),
        )
        recall_macro = torch.where(
            support > 0,
            true_positives / support.clamp_min(1.0),
            torch.zeros_like(true_positives),
        )
        macro_f1_per_class = torch.where(
            (precision_macro + recall_macro) > 0,
            2.0 * precision_macro * recall_macro / (precision_macro + recall_macro),
            torch.zeros_like(true_positives),
        )
        macro_f1 = macro_f1_per_class.mean().item()

        tp_micro = true_positives.sum()
        fp_micro = false_positives.sum()
        fn_micro = false_negatives.sum()
        denominator = 2.0 * tp_micro + fp_micro + fn_micro
        micro_f1 = (2.0 * tp_micro / denominator).item() if denominator > 0 else 0.0

        return {
            "accuracy": float(accuracy),
            "micro_f1": float(micro_f1),
            "macro_f1": float(macro_f1),
        }


def round_metrics(metrics: dict[str, float], digits: int = 4) -> dict[str, float]:
    return {key: round(value, digits) for key, value in metrics.items()}


def safe_float(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)
