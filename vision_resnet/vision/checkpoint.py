"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    state: dict[str, Any],
    output_dir: str | Path,
    is_best: bool = False,
    filename: str = "checkpoint_last.pt",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / filename
    torch.save(state, checkpoint_path)

    if is_best:
        best_path = output_dir / "checkpoint_best.pt"
        torch.save(state, best_path)

    return checkpoint_path
