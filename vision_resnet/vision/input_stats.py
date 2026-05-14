"""Dataset statistics for paper-style ImageNet preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .constants import VALID_IMAGE_EXTENSIONS
from .transforms import CenterCrop, ResizeShorterSide, ToTensor
from .utils import ensure_dir

DEFAULT_PCA_ALPHA_STD = 0.1


def _iter_image_paths(root: str | Path):
    root_path = Path(root)
    for path in sorted(root_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
            yield path


def compute_rgb_pca(train_root: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    rgb_sum = np.zeros(3, dtype=np.float64)
    rgb_outer = np.zeros((3, 3), dtype=np.float64)
    pixel_count = 0

    for image_path in _iter_image_paths(train_root):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            pixels = np.asarray(image, dtype=np.float64).reshape(-1, 3) / 255.0

        rgb_sum += pixels.sum(axis=0)
        rgb_outer += pixels.T @ pixels
        pixel_count += pixels.shape[0]

    if pixel_count == 0:
        raise FileNotFoundError(f"No image files found under {train_root}.")

    mean = rgb_sum / float(pixel_count)
    covariance = rgb_outer / float(pixel_count) - np.outer(mean, mean)
    eigvals, eigvecs = np.linalg.eigh(covariance)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    return (
        torch.tensor(eigvals, dtype=torch.float32),
        torch.tensor(eigvecs, dtype=torch.float32),
    )


def compute_mean_image(
    train_root: str | Path,
    image_size: int,
    resize_size: int,
) -> torch.Tensor:
    resize = ResizeShorterSide(resize_size)
    crop = CenterCrop(image_size)
    to_tensor = ToTensor()

    total = torch.zeros(3, int(image_size), int(image_size), dtype=torch.float64)
    image_count = 0

    for image_path in _iter_image_paths(train_root):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = crop(resize(image))
            tensor = to_tensor(image).to(dtype=torch.float64)

        total += tensor
        image_count += 1

    if image_count == 0:
        raise FileNotFoundError(f"No image files found under {train_root}.")

    return (total / float(image_count)).to(dtype=torch.float32)


def load_or_compute_resnet_paper_input_stats(
    train_root: str | Path,
    image_size: int,
    resize_size: int,
    cache_dir: str | Path | None = None,
) -> dict[str, torch.Tensor | float]:
    cache_root = ensure_dir(cache_dir or (Path(train_root).parent / ".col775_cache"))
    cache_path = cache_root / f"resnet_paper_stats_resize{resize_size}_image{image_size}.pt"

    if cache_path.is_file():
        return torch.load(cache_path, map_location="cpu")

    stats = {
        "mean_image": compute_mean_image(
            train_root=train_root,
            image_size=image_size,
            resize_size=resize_size,
        ),
        "pca_eigvals": None,
        "pca_eigvecs": None,
        "pca_alpha_std": float(DEFAULT_PCA_ALPHA_STD),
    }
    pca_eigvals, pca_eigvecs = compute_rgb_pca(train_root)
    stats["pca_eigvals"] = pca_eigvals
    stats["pca_eigvecs"] = pca_eigvecs

    torch.save(stats, cache_path)
    return stats
