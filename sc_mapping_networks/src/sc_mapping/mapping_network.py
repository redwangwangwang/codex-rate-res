from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .targets import FlatTarget


@dataclass
class MappingLossWeights:
    task: float = 1.0
    stability: float = 0.05
    smoothness: float = 1e-4
    alignment: float = 0.01


class OriginalMappingNetwork(nn.Module):
    """Minimal latent-to-weight Mapping Network reproduction."""

    def __init__(
        self,
        target: FlatTarget,
        latent_dim: int = 512,
        hidden_dim: int = 256,
        seed: int = 0,
        activation: str = "gelu",
        output_scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.target = target
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_scale = float(output_scale)
        self.activation = activation

        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        self.z = nn.Parameter(torch.randn(latent_dim, generator=g) * 0.02)

        w1 = torch.empty(hidden_dim, latent_dim)
        nn.init.orthogonal_(w1)
        b1 = torch.zeros(hidden_dim)
        w2 = torch.empty(target.num_params, hidden_dim)
        nn.init.orthogonal_(w2)
        b2 = torch.zeros(target.num_params)
        self.register_buffer("w1", w1)
        self.register_buffer("b1", b1)
        self.register_buffer("w2", w2)
        self.register_buffer("b2", b2)

    @property
    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_generated(self) -> int:
        return self.target.num_params

    def _act(self, x: Tensor) -> Tensor:
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "tanh":
            return torch.tanh(x)
        return F.gelu(x)

    def theta(self, z: Optional[Tensor] = None) -> Tensor:
        if z is None:
            z = self.z
        h = self._act(F.linear(z, self.w1, self.b1))
        scale = 1.0 + 0.1 * torch.tanh(h)
        theta = F.linear(h * scale, self.w2, self.b2)
        return self.output_scale * theta

    def forward(self, x: Tensor, z: Optional[Tensor] = None) -> Tensor:
        return self.target(x, self.theta(z))


def mapping_regularizers(model: OriginalMappingNetwork, x: Tensor, logits: Tensor, eps_std: float = 1e-2) -> dict[str, Tensor]:
    z = model.z
    noise = torch.randn_like(z) * eps_std
    with torch.enable_grad():
        logits_perturbed = model(x, z + noise)
        theta = model.theta(z)
        theta_perturbed = model.theta(z + noise)
    stability = F.mse_loss(logits_perturbed, logits.detach())
    smoothness = (theta_perturbed - theta).pow(2).mean() / (noise.pow(2).mean() + 1e-12)
    align_anchor = model.w1.mean(dim=0).detach()
    alignment = 1.0 - F.cosine_similarity(z, align_anchor, dim=0)
    return {"stability": stability, "smoothness": smoothness, "alignment": alignment}


def mapping_loss(task_loss: Tensor, regs: dict[str, Tensor], weights: MappingLossWeights) -> Tensor:
    return weights.task * task_loss + weights.stability * regs["stability"] + weights.smoothness * regs["smoothness"] + weights.alignment * regs["alignment"]
