"""ResNet-18 implementation with pluggable normalization and shortcut types."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from .normalization import (
    CUSTOM_NORM_TYPES,
    IdentityNorm,
    NormConfig,
)


def conv3x3(in_channels: int, out_channels: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=bias,
    )


def compute_conv_output_size(size: int, kernel_size: int, stride: int, padding: int) -> int:
    return ((int(size) + 2 * int(padding) - int(kernel_size)) // int(stride)) + 1


class ProjectionShortcut(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        norm_config: NormConfig,
        spatial_size: int,
    ):
        super().__init__()
        self.conv = conv1x1(
            in_channels,
            out_channels,
            stride=stride,
            bias=norm_config.conv_bias,
        )
        self.norm = norm_config.create(out_channels, spatial_shape=(spatial_size, spatial_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return x


class IdentityPadShortcut(nn.Module):
    """Identity shortcut with stride-based downsampling and zero channel padding."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        if out_channels < in_channels:
            raise ValueError("IdentityPadShortcut requires out_channels >= in_channels.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride > 1:
            x = x[:, :, :: self.stride, :: self.stride]

        channel_padding = self.out_channels - self.in_channels
        if channel_padding > 0:
            zeros = x.new_zeros(x.size(0), channel_padding, x.size(2), x.size(3))
            x = torch.cat((x, zeros), dim=1)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        norm_config: NormConfig,
        spatial_size: int,
        shortcut_type: Literal["projection", "identity_pad", "always_projection"] = "projection",
    ):
        super().__init__()
        bias = norm_config.conv_bias

        self.conv1 = conv3x3(in_channels, out_channels, stride=stride, bias=bias)
        self.norm1 = norm_config.create(out_channels, spatial_shape=(spatial_size, spatial_size))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, stride=1, bias=bias)
        self.norm2 = norm_config.create(out_channels, spatial_shape=(spatial_size, spatial_size))

        requires_projection = (
            shortcut_type == "always_projection" or stride != 1 or in_channels != out_channels
        )

        if requires_projection:
            if shortcut_type in {"projection", "always_projection"}:
                self.downsample = ProjectionShortcut(
                    in_channels,
                    out_channels,
                    stride=stride,
                    norm_config=norm_config,
                    spatial_size=spatial_size,
                )
            elif shortcut_type == "identity_pad":
                self.downsample = IdentityPadShortcut(
                    in_channels,
                    out_channels,
                    stride=stride,
                )
            else:
                raise ValueError(f"Unsupported shortcut type: {shortcut_type}")
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        norm_config: NormConfig | None = None,
        shortcut_type: Literal["projection", "identity_pad", "always_projection"] = "projection",
        dropout_prob: float = 0.0,
        input_size: int = 224,
    ):
        super().__init__()
        self.norm_config = norm_config or NormConfig()
        self.shortcut_type = shortcut_type
        self.in_channels = 64
        self.feature_map_resolution = int(input_size)

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=self.norm_config.conv_bias,
        )
        self.feature_map_resolution = compute_conv_output_size(self.feature_map_resolution, 7, 2, 3)
        self.norm1 = self.norm_config.create(
            64,
            spatial_shape=(self.feature_map_resolution, self.feature_map_resolution),
        )
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.feature_map_resolution = compute_conv_output_size(self.feature_map_resolution, 3, 2, 1)

        self.layer1 = self._make_layer(out_channels=64, blocks=2, stride=1)
        self.layer2 = self._make_layer(out_channels=128, blocks=2, stride=2)
        self.layer3 = self._make_layer(out_channels=256, blocks=2, stride=2)
        self.layer4 = self._make_layer(out_channels=512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=float(dropout_prob)) if float(dropout_prob) > 0.0 else nn.Identity()
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        next_feature_map_resolution = compute_conv_output_size(self.feature_map_resolution, 3, stride, 1)
        layers = [
            BasicBlock(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride,
                norm_config=self.norm_config,
                spatial_size=next_feature_map_resolution,
                shortcut_type=self.shortcut_type,
            )
        ]
        self.in_channels = out_channels * BasicBlock.expansion
        self.feature_map_resolution = next_feature_map_resolution

        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=1,
                    norm_config=self.norm_config,
                    spatial_size=self.feature_map_resolution,
                    shortcut_type=self.shortcut_type,
                )
            )
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        norm_types = (nn.BatchNorm2d, IdentityNorm, *CUSTOM_NORM_TYPES)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)
            elif isinstance(module, norm_types):
                if getattr(module, "weight", None) is not None:
                    nn.init.ones_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def build_resnet18(
    num_classes: int = 100,
    norm_kind: str = "baseline",
    norm_eps: float = 1e-5,
    bn_momentum: float = 0.1,
    group_count: int = 32,
    shortcut_type: Literal["projection", "identity_pad", "always_projection"] = "projection",
    dropout_prob: float = 0.0,
    input_size: int = 224,
) -> ResNet18:
    norm_config = NormConfig(
        kind=norm_kind,
        eps=norm_eps,
        momentum=bn_momentum,
        group_count=group_count,
    )
    return ResNet18(
        num_classes=num_classes,
        norm_config=norm_config,
        shortcut_type=shortcut_type,
        dropout_prob=dropout_prob,
        input_size=input_size,
    )
