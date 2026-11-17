"""Minimal OSNet-x0.25 feature extractor for mask-aware association.

The architecture follows the official MIT-licensed Torchreid implementation:
https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/models/osnet.py

Only the inference feature path is included. The first convolution optionally
accepts a fourth mask channel; official RGB weights are copied into channels
0..2 and the mask channel is zero-initialized so pretraining is preserved at
the start of fine-tuning.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


OSNET_X0_25_IMAGENET_SHA256 = "f54941a66bad4ddd07f2907f498c810ce639ce7a1abeaf2a151f8da118d84693"
OSNET_SOURCE_URL = "https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/models/osnet.py"
OSNET_WEIGHT_URL = "https://huggingface.co/kaiyangzhou/osnet/blob/main/osnet_x0_25_imagenet.pth"


class ConvLayer(nn.Module):
    """Convolution, batch normalization, and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(ConvLayer):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(in_channels, out_channels, 1)


class Conv1x1Linear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    """Pointwise linear projection followed by a depthwise 3x3 convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    """Channel-wise gate shared by all four omni-scale streams."""

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(in_channels // reduction, 1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, hidden, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, in_channels, 1, bias=True)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = self.global_avgpool(x)
        gates = self.fc2(self.relu(self.fc1(gates)))
        return x * self.gate_activation(gates)


class OSBlock(nn.Module):
    """Official OSNet omni-scale residual block."""

    def __init__(self, in_channels: int, out_channels: int, bottleneck_reduction: int = 4) -> None:
        super().__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = Conv1x1Linear(in_channels, out_channels) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        reduced = self.conv1(x)
        mixed = (
            self.gate(self.conv2a(reduced))
            + self.gate(self.conv2b(reduced))
            + self.gate(self.conv2c(reduced))
            + self.gate(self.conv2d(reduced))
        )
        output = self.conv3(mixed)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(output + identity)


class OSNetX025FeatureExtractor(nn.Module):
    """OSNet-x0.25 backbone with the official 512-dimensional feature layer."""

    output_dim = 512

    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        channels = (16, 64, 96, 128)
        self.conv1 = ConvLayer(in_channels, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(channels[0], channels[1], reduce_spatial_size=True)
        self.conv3 = self._make_layer(channels[1], channels[2], reduce_spatial_size=True)
        self.conv4 = self._make_layer(channels[2], channels[3], reduce_spatial_size=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[3], self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(inplace=True),
        )
        self._init_params()

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, reduce_spatial_size: bool) -> nn.Sequential:
        layers: list[nn.Module] = [OSBlock(in_channels, out_channels), OSBlock(out_channels, out_channels)]
        if reduce_spatial_size:
            layers.append(nn.Sequential(Conv1x1(out_channels, out_channels), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def _init_params(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.conv1(x))
        x = self.conv5(self.conv4(self.conv3(self.conv2(x))))
        x = self.global_avgpool(x).flatten(1)
        return self.fc(x)


def load_osnet_x0_25_imagenet(model: OSNetX025FeatureExtractor, path: str | Path) -> dict[str, Any]:
    """Load official Torchreid ImageNet weights, adapting RGB conv1 to RGB+mask."""
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != OSNET_X0_25_IMAGENET_SHA256:
        raise ValueError(
            f"OSNet-x0.25 ImageNet SHA256 mismatch: expected {OSNET_X0_25_IMAGENET_SHA256}, got {digest}"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"unsupported OSNet checkpoint format: {path}")
    target = model.state_dict()
    accepted: dict[str, torch.Tensor] = {}
    discarded: list[str] = []
    adapted_mask_channel = False
    for raw_name, value in checkpoint.items():
        name = str(raw_name)
        if name.startswith("module."):
            name = name[7:]
        if name == "conv1.conv.weight" and name in target and value.shape[1] == 3 and target[name].shape[1] == 4:
            adapted = torch.zeros_like(target[name])
            adapted[:, :3] = value
            accepted[name] = adapted
            adapted_mask_channel = True
        elif name in target and target[name].shape == value.shape:
            accepted[name] = value
        else:
            discarded.append(name)
    if not accepted:
        raise ValueError(f"no OSNet layers matched the official checkpoint: {path}")
    missing, unexpected = model.load_state_dict(accepted, strict=False)
    return {
        "path": str(path),
        "sha256": digest,
        "source_code": OSNET_SOURCE_URL,
        "weight_source": OSNET_WEIGHT_URL,
        "matched_layers": len(accepted),
        "discarded_layers": len(discarded),
        "missing_layers": list(missing),
        "unexpected_layers": list(unexpected),
        "mask_channel_zero_initialized": adapted_mask_channel,
    }
