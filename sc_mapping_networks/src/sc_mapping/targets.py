"""Flat-parameter target networks used by Mapping Networks and SC-Mapping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ParamSpec:
    name: str
    shape: Tuple[int, ...]

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= int(s)
        return n


class FlatTarget:
    """Base class for target networks parameterized by one flat vector."""

    specs: Sequence[ParamSpec]

    @property
    def num_params(self) -> int:
        return sum(s.numel for s in self.specs)

    def unpack(self, theta: Tensor) -> Dict[str, Tensor]:
        if theta.ndim != 1:
            raise ValueError(f"theta must be 1-D, got shape {tuple(theta.shape)}")
        if theta.numel() != self.num_params:
            raise ValueError(f"expected {self.num_params} parameters, got {theta.numel()}")
        out: Dict[str, Tensor] = {}
        offset = 0
        for spec in self.specs:
            chunk = theta[offset : offset + spec.numel]
            out[spec.name] = chunk.view(spec.shape)
            offset += spec.numel
        return out

    def init_theta(self, seed: int = 0, device: torch.device | str = "cpu") -> Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        tensors: List[Tensor] = []
        for spec in self.specs:
            if spec.name.endswith(".bias") or len(spec.shape) == 1:
                t = torch.zeros(spec.shape)
            else:
                fan_in = spec.shape[1]
                if len(spec.shape) > 2:
                    fan_in *= spec.shape[2] * spec.shape[3]
                t = torch.randn(spec.shape, generator=generator) * (2.0 / fan_in) ** 0.5
            tensors.append(t.reshape(-1))
        return torch.cat(tensors).to(device)

    def __call__(self, x: Tensor, theta: Tensor) -> Tensor:
        return self.forward_from_flat(x, theta)

    def forward_from_flat(self, x: Tensor, theta: Tensor) -> Tensor:  # pragma: no cover
        raise NotImplementedError


class MLPClassifier(FlatTarget):
    """Two-layer MLP for MNIST/FashionMNIST sanity checks."""

    def __init__(self, in_dim: int = 784, hidden_dim: int = 128, num_classes: int = 10):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.specs = [
            ParamSpec("fc1.weight", (hidden_dim, in_dim)),
            ParamSpec("fc1.bias", (hidden_dim,)),
            ParamSpec("fc2.weight", (num_classes, hidden_dim)),
            ParamSpec("fc2.bias", (num_classes,)),
        ]

    def forward_from_flat(self, x: Tensor, theta: Tensor) -> Tensor:
        p = self.unpack(theta)
        x = x.view(x.size(0), -1)
        x = F.linear(x, p["fc1.weight"], p["fc1.bias"])
        x = F.relu(x, inplace=False)
        return F.linear(x, p["fc2.weight"], p["fc2.bias"])


class SmallCNN(FlatTarget):
    """Small CNN similar in spirit to the paper's image-classification target nets."""

    def __init__(self, in_channels: int = 1, width: int = 16, num_classes: int = 10):
        self.in_channels = in_channels
        self.width = width
        self.num_classes = num_classes
        w = width
        self.specs = [
            ParamSpec("conv1.weight", (w, in_channels, 3, 3)),
            ParamSpec("conv1.bias", (w,)),
            ParamSpec("conv2.weight", (2 * w, w, 3, 3)),
            ParamSpec("conv2.bias", (2 * w,)),
            ParamSpec("fc1.weight", (64, 2 * w * 7 * 7)),
            ParamSpec("fc1.bias", (64,)),
            ParamSpec("fc2.weight", (num_classes, 64)),
            ParamSpec("fc2.bias", (num_classes,)),
        ]

    def forward_from_flat(self, x: Tensor, theta: Tensor) -> Tensor:
        p = self.unpack(theta)
        x = F.conv2d(x, p["conv1.weight"], p["conv1.bias"], padding=1)
        x = F.relu(x, inplace=False)
        x = F.max_pool2d(x, 2)
        x = F.conv2d(x, p["conv2.weight"], p["conv2.bias"], padding=1)
        x = F.relu(x, inplace=False)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.linear(x, p["fc1.weight"], p["fc1.bias"])
        x = F.relu(x, inplace=False)
        return F.linear(x, p["fc2.weight"], p["fc2.bias"])


def make_target(name: str, **kwargs) -> FlatTarget:
    name = name.lower()
    if name in {"mlp", "mlpclassifier"}:
        return MLPClassifier(**kwargs)
    if name in {"cnn", "smallcnn"}:
        return SmallCNN(**kwargs)
    raise ValueError(f"Unknown target network: {name}")
