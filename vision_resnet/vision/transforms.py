"""Minimal image transforms implemented without torchvision."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps

from .constants import IMAGENET_MEAN, IMAGENET_STD

Interpolation = Image.Resampling


def _resolve_hw_pair(size: int | Sequence[int], op_name: str) -> tuple[int, int]:
    if isinstance(size, Iterable) and not isinstance(size, (str, bytes)):
        size = tuple(size)
        if len(size) != 2:
            raise ValueError(f"{op_name} expects a single int or a pair of ints.")
        return int(size[0]), int(size[1])
    scalar_size = int(size)
    return scalar_size, scalar_size


def _pad_image_to_minimum_size(
    image: Image.Image,
    min_width: int,
    min_height: int,
) -> Image.Image:
    width, height = image.size
    if width >= min_width and height >= min_height:
        return image
    pad_width = max(min_width - width, 0)
    pad_height = max(min_height - height, 0)
    return ImageOps.expand(
        image,
        border=(0, 0, pad_width, pad_height),
        fill=0,
    )


class Compose:
    """Compose image transforms."""

    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)

    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image


class ResizeShorterSide:
    """Resize the shorter image side to a fixed length."""

    def __init__(self, size: int, interpolation: int = Interpolation.BILINEAR):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width == 0 or height == 0:
            raise ValueError("Encountered image with zero-sized dimension.")

        if width <= height:
            new_width = self.size
            new_height = int(round(height * self.size / width))
        else:
            new_height = self.size
            new_width = int(round(width * self.size / height))

        return image.resize((new_width, new_height), self.interpolation)


class RandomResizeShorterSide:
    """Resize the shorter image side to a random length in a fixed range."""

    def __init__(
        self,
        min_size: int,
        max_size: int,
        interpolation: int = Interpolation.BILINEAR,
    ):
        self.min_size = int(min_size)
        self.max_size = int(max_size)
        if self.max_size < self.min_size:
            raise ValueError("max_size must be >= min_size for RandomResizeShorterSide.")
        self.interpolation = interpolation

    def __call__(self, image: Image.Image) -> Image.Image:
        size = random.randint(self.min_size, self.max_size)
        return ResizeShorterSide(size=size, interpolation=self.interpolation)(image)


class CenterCrop:
    """Crop the central patch from an image."""

    def __init__(self, size: int | Sequence[int]):
        self.height, self.width = _resolve_hw_pair(size, "CenterCrop")

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        crop_width = min(self.width, width)
        crop_height = min(self.height, height)
        left = max((width - crop_width) // 2, 0)
        top = max((height - crop_height) // 2, 0)
        return image.crop((left, top, left + crop_width, top + crop_height))


class RandomCrop:
    """Randomly crop an image to a fixed size."""

    def __init__(self, size: int | Sequence[int], padding: int = 0):
        self.height, self.width = _resolve_hw_pair(size, "RandomCrop")
        self.padding = int(padding)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.padding > 0:
            image = ImageOps.expand(image, border=self.padding, fill=0)

        image = _pad_image_to_minimum_size(image, self.width, self.height)
        width, height = image.size

        left = random.randint(0, width - self.width)
        top = random.randint(0, height - self.height)
        return image.crop((left, top, left + self.width, top + self.height))


class RandomHorizontalFlip:
    """Flip an image horizontally with probability p."""

    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image


class ToTensor:
    """Convert a PIL image into a float tensor in CHW format."""

    def __call__(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        if array.ndim == 2:
            array = np.expand_dims(array, axis=-1)
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array)


class Normalize:
    """Normalize a float tensor image channel-wise."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std


class SubtractMeanImage:
    """Subtract a cached per-pixel mean image from a CHW tensor."""

    def __init__(self, mean_image: torch.Tensor):
        if mean_image.ndim != 3:
            raise ValueError("SubtractMeanImage expects a CHW mean image tensor.")
        self.mean_image = mean_image.to(dtype=torch.float32)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape != self.mean_image.shape:
            raise ValueError(
                f"Mean image shape {tuple(self.mean_image.shape)} does not match input {tuple(tensor.shape)}."
            )
        return tensor - self.mean_image


class PCALighting:
    """AlexNet-style PCA color augmentation on RGB tensors."""

    def __init__(
        self,
        eigvals: torch.Tensor,
        eigvecs: torch.Tensor,
        alpha_std: float = 0.1,
    ):
        if eigvals.shape != (3,) or eigvecs.shape != (3, 3):
            raise ValueError("PCALighting expects eigvals shape (3,) and eigvecs shape (3, 3).")
        self.eigvals = eigvals.to(dtype=torch.float32)
        self.eigvecs = eigvecs.to(dtype=torch.float32)
        self.alpha_std = float(alpha_std)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.alpha_std <= 0.0:
            return tensor
        alpha = torch.normal(mean=0.0, std=self.alpha_std, size=(3,), dtype=tensor.dtype)
        rgb = self.eigvecs.to(dtype=tensor.dtype) @ (
            self.eigvals.to(dtype=tensor.dtype) * alpha
        )
        return tensor + rgb.view(3, 1, 1)


class Cutout:
    """Randomly mask out a square region in the tensor image."""

    def __init__(self, size: int, fill_value: float = 0.0):
        self.size = int(size)
        self.fill_value = float(fill_value)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 3:
            raise ValueError("Cutout expects a CHW tensor.")

        _, height, width = tensor.shape
        center_y = random.randint(0, height - 1)
        center_x = random.randint(0, width - 1)
        half = self.size // 2
        top = max(center_y - half, 0)
        bottom = min(center_y + half, height)
        left = max(center_x - half, 0)
        right = min(center_x + half, width)
        tensor = tensor.clone()
        tensor[:, top:bottom, left:right] = self.fill_value
        return tensor


AUTOAUGMENT_MAGNITUDE_BINS = 10
# ImageNet policy from Table 9 of the AutoAugment supplementary material.
IMAGENET_AUTOAUGMENT_POLICY: tuple[
    tuple[tuple[str, float, int], tuple[str, float, int]],
    ...,
] = (
    (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),
    (("Solarize", 0.6, 5), ("AutoContrast", 0.6, 5)),
    (("Equalize", 0.8, 8), ("Equalize", 0.6, 3)),
    (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
    (("Equalize", 0.4, 7), ("Solarize", 0.2, 4)),
    (("Equalize", 0.4, 4), ("Rotate", 0.8, 8)),
    (("Solarize", 0.6, 3), ("Equalize", 0.6, 7)),
    (("Posterize", 0.8, 5), ("Equalize", 1.0, 2)),
    (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),
    (("Equalize", 0.6, 8), ("Posterize", 0.4, 6)),
    (("Rotate", 0.8, 8), ("Color", 0.4, 0)),
    (("Rotate", 0.4, 9), ("Equalize", 0.6, 2)),
    (("Equalize", 0.0, 7), ("Equalize", 0.8, 8)),
    (("Invert", 0.6, 4), ("Equalize", 1.0, 8)),
    (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
    (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
    (("Color", 0.8, 8), ("Solarize", 0.8, 7)),
    (("Sharpness", 0.4, 7), ("Invert", 0.6, 8)),
    (("ShearX", 0.6, 5), ("Equalize", 1.0, 9)),
    (("Color", 0.4, 0), ("Equalize", 0.6, 3)),
    (("Equalize", 0.4, 7), ("Solarize", 0.2, 4)),
    (("Solarize", 0.6, 5), ("AutoContrast", 0.6, 5)),
    (("Invert", 0.6, 4), ("Equalize", 1.0, 8)),
    (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
    (("Equalize", 0.8, 8), ("Equalize", 0.6, 3)),
)


def _fill_for_mode(image: Image.Image, value: int = 128) -> int | tuple[int, ...]:
    if image.mode == "RGB":
        return (value, value, value)
    if image.mode in {"L", "1"}:
        return value
    return tuple(value for _ in image.getbands())


def _level_to_magnitude(level: int, lower: float, upper: float) -> float:
    if not 0 <= int(level) < AUTOAUGMENT_MAGNITUDE_BINS:
        raise ValueError(
            f"AutoAugment magnitude level must be in [0, {AUTOAUGMENT_MAGNITUDE_BINS - 1}]."
        )
    if AUTOAUGMENT_MAGNITUDE_BINS == 1:
        return float(upper)
    return float(lower) + (float(upper) - float(lower)) * (
        float(level) / float(AUTOAUGMENT_MAGNITUDE_BINS - 1)
    )


def _maybe_negate(value: float) -> float:
    return -value if random.random() < 0.5 else value


def _apply_autoaugment_op(image: Image.Image, name: str, level: int) -> Image.Image:
    fill = _fill_for_mode(image)
    op_name = name.lower()

    if op_name == "autocontrast":
        return ImageOps.autocontrast(image)
    if op_name == "equalize":
        return ImageOps.equalize(image)
    if op_name == "invert":
        return ImageOps.invert(image)
    if op_name == "posterize":
        bits = int(round(_level_to_magnitude(level, 8.0, 4.0)))
        return ImageOps.posterize(image, bits)
    if op_name == "solarize":
        threshold = int(round(_level_to_magnitude(level, 256.0, 0.0)))
        return ImageOps.solarize(image, threshold)
    if op_name == "rotate":
        degrees = _maybe_negate(_level_to_magnitude(level, 0.0, 30.0))
        return image.rotate(degrees, resample=Interpolation.BILINEAR, fillcolor=fill)
    if op_name == "shearx":
        shear = _maybe_negate(_level_to_magnitude(level, 0.0, 0.3))
        return image.transform(
            image.size,
            Image.Transform.AFFINE,
            (1.0, shear, 0.0, 0.0, 1.0, 0.0),
            resample=Interpolation.BILINEAR,
            fillcolor=fill,
        )
    if op_name == "color":
        factor = 1.0 + _maybe_negate(_level_to_magnitude(level, 0.0, 0.9))
        return ImageEnhance.Color(image).enhance(factor)
    if op_name == "contrast":
        factor = 1.0 + _maybe_negate(_level_to_magnitude(level, 0.0, 0.9))
        return ImageEnhance.Contrast(image).enhance(factor)
    if op_name == "sharpness":
        factor = 1.0 + _maybe_negate(_level_to_magnitude(level, 0.0, 0.9))
        return ImageEnhance.Sharpness(image).enhance(factor)

    raise ValueError(f"Unsupported AutoAugment operation: {name}")


class ImageNetAutoAugment:
    """Apply one random ImageNet AutoAugment sub-policy per image."""

    def __init__(
        self,
        policy: Sequence[tuple[tuple[str, float, int], tuple[str, float, int]]] | None = None,
    ):
        self.policy = tuple(policy or IMAGENET_AUTOAUGMENT_POLICY)

    def __call__(self, image: Image.Image) -> Image.Image:
        for op_name, probability, level in random.choice(self.policy):
            if random.random() <= probability:
                image = _apply_autoaugment_op(image, op_name, level)
        return image


@dataclass(frozen=True)
class TransformConfig:
    image_size: int = 224
    resize_size: int = 256
    train_resize_max: int = 480
    hflip_prob: float = 0.5
    normalization: str = "imagenet"
    use_cutout: bool = False
    cutout_size: int = 56
    autoaugment_policy: str = "none"
    train_resize_mode: str = "alexnet_square"
    mean_image: torch.Tensor | None = None
    pca_eigvals: torch.Tensor | None = None
    pca_eigvecs: torch.Tensor | None = None
    pca_alpha_std: float = 0.1
    use_pca_color_augmentation: bool = False


def build_square_resize_transform(config: TransformConfig) -> Compose:
    square_resize_ops = [
        ResizeShorterSide(config.resize_size),
        CenterCrop(config.resize_size),
    ]
    return Compose(square_resize_ops)


def build_visual_transform(config: TransformConfig) -> Compose:
    visual_ops = build_square_resize_transform(config).transforms + [
        CenterCrop(config.image_size)
    ]
    return Compose(visual_ops)


def _build_train_image_ops(config: TransformConfig) -> list[Callable]:
    if config.train_resize_mode == "alexnet_square":
        return build_square_resize_transform(config).transforms + [
            RandomCrop(config.image_size),
            RandomHorizontalFlip(config.hflip_prob),
        ]
    if config.train_resize_mode in {"resnet_paper", "resnet_strict"}:
        return [
            RandomResizeShorterSide(config.resize_size, config.train_resize_max),
            RandomCrop(config.image_size),
            RandomHorizontalFlip(config.hflip_prob),
        ]
    raise ValueError(f"Unsupported train resize mode: {config.train_resize_mode}")


def _build_optional_image_augmentation_ops(config: TransformConfig) -> list[Callable]:
    if config.autoaugment_policy == "none":
        return []
    if config.autoaugment_policy == "imagenet":
        return [ImageNetAutoAugment()]
    raise ValueError(f"Unsupported AutoAugment policy: {config.autoaugment_policy}")


def _build_optional_tensor_augmentation_ops(config: TransformConfig) -> list[Callable]:
    tensor_augmentations: list[Callable] = []
    if config.use_pca_color_augmentation:
        if config.pca_eigvals is None or config.pca_eigvecs is None:
            raise ValueError("PCA color augmentation requested without cached PCA statistics.")
        tensor_augmentations.append(
            PCALighting(
                eigvals=config.pca_eigvals,
                eigvecs=config.pca_eigvecs,
                alpha_std=config.pca_alpha_std,
            )
        )
    return tensor_augmentations


def build_train_transform(config: TransformConfig) -> Compose:
    train_transform_ops = _build_train_image_ops(config)
    train_transform_ops.extend(_build_optional_image_augmentation_ops(config))
    train_transform_ops.append(ToTensor())
    train_transform_ops.extend(_build_optional_tensor_augmentation_ops(config))
    train_transform_ops.extend(build_normalization_transform(config).transforms)
    if config.use_cutout:
        train_transform_ops.append(Cutout(config.cutout_size))
    return Compose(train_transform_ops)


def build_normalization_transform(config: TransformConfig) -> Compose:
    normalization_ops: list[Callable] = []
    if config.normalization == "imagenet":
        normalization_ops.append(Normalize(IMAGENET_MEAN, IMAGENET_STD))
    elif config.normalization == "mean_only":
        normalization_ops.append(Normalize(IMAGENET_MEAN, (1.0, 1.0, 1.0)))
    elif config.normalization == "per_pixel_mean":
        if config.mean_image is None:
            raise ValueError("Per-pixel mean normalization requested without a cached mean image.")
        normalization_ops.append(SubtractMeanImage(config.mean_image))
    elif config.normalization != "none":
        raise ValueError(f"Unsupported normalization mode: {config.normalization}")

    return Compose(normalization_ops)


def build_tensor_transform(config: TransformConfig) -> Compose:
    tensor_ops = [ToTensor()]
    tensor_ops.extend(build_normalization_transform(config).transforms)
    return Compose(tensor_ops)


def build_eval_transform(config: TransformConfig) -> Compose:
    eval_ops = build_square_resize_transform(config).transforms + [
        CenterCrop(config.image_size),
    ]
    eval_ops.extend(build_tensor_transform(config).transforms)
    return Compose(eval_ops)
