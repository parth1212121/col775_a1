"""Dataset helpers for ImageNet-style directory layouts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .constants import VALID_IMAGE_EXTENSIONS


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in VALID_IMAGE_EXTENSIONS


def discover_classes(root: str | os.PathLike[str]) -> tuple[list[str], dict[str, int]]:
    root_path = Path(root)
    classes = sorted(entry.name for entry in root_path.iterdir() if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class directories found under {root_path}.")
    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
    return classes, class_to_idx


def collect_image_samples(
    root: str | os.PathLike[str],
    class_to_idx: dict[str, int],
) -> list[tuple[str, int]]:
    root_path = Path(root)
    samples: list[tuple[str, int]] = []
    for class_name in sorted(class_to_idx):
        class_dir = root_path / class_name
        if not class_dir.is_dir():
            continue
        for file_path in sorted(class_dir.rglob("*")):
            if file_path.is_file() and _is_image_file(file_path):
                samples.append((str(file_path), class_to_idx[class_name]))
    if not samples:
        raise FileNotFoundError(f"No image files found under {root_path}.")
    return samples


class ImageFolderDataset(Dataset):
    """Small torchvision-free ImageFolder replacement."""

    def __init__(self, root: str | os.PathLike[str], transform: Callable | None = None):
        self.root = Path(root)
        self.transform = transform
        self.classes, self.class_to_idx = discover_classes(self.root)
        self.samples = collect_image_samples(self.root, self.class_to_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, target, path


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = 128
    eval_batch_size: int | None = None
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


def _build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    config: DataLoaderConfig,
    num_workers: int,
    persistent_workers: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=persistent_workers,
    )


def build_dataloaders(
    data_root: str | os.PathLike[str],
    train_transform: Callable,
    eval_transform: Callable,
    config: DataLoaderConfig,
) -> dict[str, object]:
    data_root = Path(data_root)
    train_root = data_root / "train"
    val_root = data_root / "val"
    if not train_root.is_dir() or not val_root.is_dir():
        raise FileNotFoundError(
            f"Expected ImageNet-style 'train' and 'val' directories under {data_root}."
        )

    train_dataset = ImageFolderDataset(train_root, transform=train_transform)
    val_dataset = ImageFolderDataset(val_root, transform=eval_transform)

    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Train/validation class folders do not match.")

    eval_batch_size = config.eval_batch_size or config.batch_size
    num_workers = max(int(config.num_workers), 0)
    persistent_workers = bool(config.persistent_workers and num_workers > 0)

    train_loader = _build_loader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        config=config,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    val_loader = _build_loader(
        dataset=val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        config=config,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
    }
