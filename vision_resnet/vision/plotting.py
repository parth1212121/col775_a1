"""Training curve plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_history(history: dict[str, list[dict[str, float]]], output_path: str | Path) -> None:
    epochs = [entry["epoch"] for entry in history["train"]]
    metrics = ["loss", "accuracy", "micro_f1", "macro_f1"]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, metric in zip(axes.flat, metrics):
        train_values = [entry[metric] for entry in history["train"]]
        val_values = [entry[metric] for entry in history["val"]]
        axis.plot(epochs, train_values, label="train")
        axis.plot(epochs, val_values, label="val")
        axis.set_title(metric.replace("_", " ").title())
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
