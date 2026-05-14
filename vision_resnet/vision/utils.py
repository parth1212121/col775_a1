"""General utilities for training scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)
