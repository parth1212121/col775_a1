"""Custom normalization layers used in ResNet experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

NormKind = Literal["baseline", "nn", "bn", "in", "bin", "ln", "gn"]


def _reshape_channel_vector(values: torch.Tensor) -> torch.Tensor:
    return values.view(1, -1, 1, 1)


def _cast_stat_input(input_tensor: torch.Tensor) -> torch.Tensor:
    if input_tensor.dtype in (torch.float16, torch.bfloat16):
        return input_tensor.float()
    return input_tensor


def _compute_batch_channel_stats(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    stats_input = _cast_stat_input(input_tensor)
    batch_mean = stats_input.mean(dim=(0, 2, 3))
    batch_var = stats_input.var(dim=(0, 2, 3), unbiased=False)
    return batch_mean, batch_var


def _compute_unbiased_channel_var(batch_var: torch.Tensor, elements_per_channel: int) -> torch.Tensor:
    if elements_per_channel > 1:
        return batch_var * elements_per_channel / (elements_per_channel - 1)
    return batch_var


def _resolve_running_stat_momentum(momentum: float | None, num_batches_tracked: torch.Tensor) -> float:
    if momentum is None:
        return 1.0 / float(num_batches_tracked.item())
    return float(momentum)


def _update_running_channel_stats(
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    num_batches_tracked: torch.Tensor,
    batch_mean: torch.Tensor,
    batch_var: torch.Tensor,
    momentum: float | None,
    elements_per_channel: int,
) -> None:
    num_batches_tracked.add_(1)
    update_factor = _resolve_running_stat_momentum(momentum, num_batches_tracked)
    unbiased_batch_var = _compute_unbiased_channel_var(batch_var, elements_per_channel)
    with torch.no_grad():
        running_mean.mul_(1.0 - update_factor).add_(update_factor * batch_mean.detach())
        running_var.mul_(1.0 - update_factor).add_(update_factor * unbiased_batch_var.detach())


class IdentityNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _AffineNormBase(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine = bool(affine)

        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.num_features))
            self.bias = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def _apply_affine(self, x: torch.Tensor) -> torch.Tensor:
        if not self.affine:
            return x
        return x * _reshape_channel_vector(self.weight) + _reshape_channel_vector(self.bias)

    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"{self.__class__.__name__} expects 4D NCHW inputs.")


class CustomBatchNorm2d(_AffineNormBase):
    """Manual BatchNorm matching PyTorch semantics closely."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__(num_features=num_features, eps=eps, affine=affine)
        self.momentum = momentum
        self.track_running_stats = bool(track_running_stats)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(self.num_features))
            self.register_buffer("running_var", torch.ones(self.num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)
        stats_input = _cast_stat_input(x)

        if self.training or not self.track_running_stats:
            batch_mean, batch_var = _compute_batch_channel_stats(x)

            if self.training and self.track_running_stats:
                _update_running_channel_stats(
                    running_mean=self.running_mean,
                    running_var=self.running_var,
                    num_batches_tracked=self.num_batches_tracked,
                    batch_mean=batch_mean,
                    batch_var=batch_var,
                    momentum=self.momentum,
                    elements_per_channel=x.numel() // x.shape[1],
                )
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var

        x = (stats_input - _reshape_channel_vector(batch_mean)) / torch.sqrt(
            _reshape_channel_vector(batch_var) + self.eps
        )
        return self._apply_affine(x)


class CustomInstanceNorm2d(_AffineNormBase):
    """InstanceNorm over each sample-channel pair."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self._apply_affine(x)


class CustomLayerNorm2d(_AffineNormBase):
    """LayerNorm for convolutional tensors over C, H, and W per sample."""

    def __init__(
        self,
        num_features: int,
        spatial_shape: tuple[int, int],
        eps: float = 1e-5,
        affine: bool = True,
    ):
        nn.Module.__init__(self)
        self.num_features = int(num_features)
        self.spatial_shape = (int(spatial_shape[0]), int(spatial_shape[1]))
        self.eps = float(eps)
        self.affine = bool(affine)

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones(self.num_features, self.spatial_shape[0], self.spatial_shape[1])
            )
            self.bias = nn.Parameter(
                torch.zeros(self.num_features, self.spatial_shape[0], self.spatial_shape[1])
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)
        if tuple(x.shape[1:]) != (self.num_features, *self.spatial_shape):
            raise ValueError(
                f"{self.__class__.__name__} expected input shape (*, {self.num_features}, "
                f"{self.spatial_shape[0]}, {self.spatial_shape[1]}), got {tuple(x.shape)}."
            )
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if not self.affine:
            return x
        return x * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class CustomGroupNorm2d(_AffineNormBase):
    """GroupNorm with a fixed number of channel groups."""

    def __init__(
        self,
        num_features: int,
        num_groups: int = 32,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__(num_features=num_features, eps=eps, affine=affine)
        self.num_groups = int(num_groups)
        if self.num_features % self.num_groups != 0:
            raise ValueError(
                f"num_features={self.num_features} must be divisible by num_groups={self.num_groups}."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)
        batch, channels, height, width = x.shape
        reshaped = x.view(batch, self.num_groups, channels // self.num_groups, height, width)
        mean = reshaped.mean(dim=(2, 3, 4), keepdim=True)
        var = reshaped.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        normalized = (reshaped - mean) / torch.sqrt(var + self.eps)
        x = normalized.view(batch, channels, height, width)
        return self._apply_affine(x)


class BatchInstanceNorm2d(_AffineNormBase):
    """Batch-Instance Normalization from Nam and Kim (2019)."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        rho_init: float = 1.0,
    ):
        super().__init__(num_features=num_features, eps=eps, affine=affine)
        self.momentum = momentum
        self.track_running_stats = bool(track_running_stats)
        self.rho = nn.Parameter(torch.full((self.num_features,), float(rho_init)))

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(self.num_features))
            self.register_buffer("running_var", torch.ones(self.num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)
        stats_input = _cast_stat_input(x)

        if self.training or not self.track_running_stats:
            batch_mean, batch_var = _compute_batch_channel_stats(x)

            if self.training and self.track_running_stats:
                _update_running_channel_stats(
                    running_mean=self.running_mean,
                    running_var=self.running_var,
                    num_batches_tracked=self.num_batches_tracked,
                    batch_mean=batch_mean,
                    batch_var=batch_var,
                    momentum=self.momentum,
                    elements_per_channel=x.numel() // x.shape[1],
                )
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var

        instance_mean = stats_input.mean(dim=(2, 3), keepdim=True)
        instance_var = stats_input.var(dim=(2, 3), keepdim=True, unbiased=False)

        batch_normed = (stats_input - _reshape_channel_vector(batch_mean)) / torch.sqrt(
            _reshape_channel_vector(batch_var) + self.eps
        )
        instance_normed = (stats_input - instance_mean) / torch.sqrt(instance_var + self.eps)

        rho = _reshape_channel_vector(self.rho.clamp(0.0, 1.0))
        mixed = rho * batch_normed + (1.0 - rho) * instance_normed
        return self._apply_affine(mixed)


def clamp_bin_parameters(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, BatchInstanceNorm2d):
            child.rho.data.clamp_(0.0, 1.0)


def resolve_group_count(num_features: int, preferred_groups: int) -> int:
    preferred_groups = min(int(preferred_groups), int(num_features))
    for candidate in range(preferred_groups, 0, -1):
        if num_features % candidate == 0:
            return candidate
    raise ValueError(f"Unable to find a valid group count for {num_features} channels.")


@dataclass(frozen=True)
class NormConfig:
    kind: NormKind = "baseline"
    eps: float = 1e-5
    momentum: float = 0.1
    group_count: int = 32

    @property
    def conv_bias(self) -> bool:
        return False

    def create(
        self,
        num_features: int,
        spatial_shape: tuple[int, int] | None = None,
    ) -> nn.Module:
        return build_norm_layer(
            kind=self.kind,
            num_features=num_features,
            eps=self.eps,
            momentum=self.momentum,
            group_count=self.group_count,
            spatial_shape=spatial_shape,
        )


def build_norm_layer(
    kind: NormKind,
    num_features: int,
    eps: float = 1e-5,
    momentum: float = 0.1,
    group_count: int = 32,
    spatial_shape: tuple[int, int] | None = None,
) -> nn.Module:
    kind = kind.lower()
    if kind == "baseline":
        return nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum)
    if kind == "nn":
        return IdentityNorm()
    if kind == "bn":
        return CustomBatchNorm2d(num_features=num_features, eps=eps, momentum=momentum)
    if kind == "in":
        return CustomInstanceNorm2d(num_features=num_features, eps=eps)
    if kind == "bin":
        return BatchInstanceNorm2d(num_features=num_features, eps=eps, momentum=momentum)
    if kind == "ln":
        if spatial_shape is None:
            raise ValueError("LayerNorm requires an explicit spatial_shape.")
        return CustomLayerNorm2d(
            num_features=num_features,
            spatial_shape=spatial_shape,
            eps=eps,
        )
    if kind == "gn":
        groups = resolve_group_count(num_features, group_count)
        return CustomGroupNorm2d(
            num_features=num_features,
            num_groups=groups,
            eps=eps,
        )
    raise ValueError(f"Unsupported normalization kind: {kind}")


CUSTOM_NORM_TYPES = (
    CustomBatchNorm2d,
    CustomInstanceNorm2d,
    BatchInstanceNorm2d,
    CustomLayerNorm2d,
    CustomGroupNorm2d,
)
