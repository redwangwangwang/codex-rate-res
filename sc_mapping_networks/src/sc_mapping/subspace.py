"""Subspace-Certified Mapping Networks (SC-Mapping)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .targets import FlatTarget


@dataclass
class BasisStats:
    singular_values: Tensor
    energy: Tensor
    retained_energy: float


@torch.no_grad()
def projection_residual(delta_theta: Tensor, basis: Tensor) -> Tensor:
    """Return ||(I - UU^T) delta||^2 / ||delta||^2 for an orthonormal basis U."""
    projected = basis @ (basis.T @ delta_theta)
    residual = delta_theta - projected
    return residual.pow(2).sum() / (delta_theta.pow(2).sum() + 1e-12)


def _batch_gradient(target: FlatTarget, theta0: Tensor, x: Tensor, y: Tensor) -> Tensor:
    theta = theta0.detach().clone().requires_grad_(True)
    logits = target(x, theta)
    loss = F.cross_entropy(logits, y)
    (grad,) = torch.autograd.grad(loss, theta, retain_graph=False, create_graph=False)
    return grad.detach()


def compute_gradient_basis(
    target: FlatTarget,
    theta0: Tensor,
    loader: Iterable[Tuple[Tensor, Tensor]],
    rank: int = 32,
    num_batches: int = 32,
    device: torch.device | str = "cpu",
    normalize_grads: bool = True,
) -> tuple[Tensor, BasisStats]:
    """Estimate a task-aligned tangent basis from mini-batch gradients."""
    theta0 = theta0.to(device)
    grads = []
    for i, (x, y) in enumerate(loader):
        if i >= num_batches:
            break
        x, y = x.to(device), y.to(device)
        g = _batch_gradient(target, theta0, x, y)
        if normalize_grads:
            g = g / (g.norm() + 1e-12)
        grads.append(g.cpu())
    if not grads:
        raise RuntimeError("No gradients collected; check that loader is non-empty.")
    G = torch.stack(grads, dim=1)  # [P, K]
    max_rank = min(rank, G.shape[1], G.shape[0])
    U, S, _ = torch.linalg.svd(G, full_matrices=False)
    U = U[:, :max_rank].contiguous()
    energy = S.pow(2) / (S.pow(2).sum() + 1e-12)
    retained = float(energy[:max_rank].sum().item())
    return U.to(device), BasisStats(S, energy, retained)


class SCMappingNetwork(nn.Module):
    """Certifiable low-dimensional training: theta(z) = theta0 + U z."""

    def __init__(self, target: FlatTarget, theta0: Tensor, basis: Tensor, z_init_std: float = 0.0) -> None:
        super().__init__()
        if basis.ndim != 2:
            raise ValueError("basis must be [num_params, rank]")
        if basis.shape[0] != target.num_params:
            raise ValueError(f"basis first dim {basis.shape[0]} != target params {target.num_params}")
        self.target = target
        self.register_buffer("theta0", theta0.detach().clone())
        self.register_buffer("basis", basis.detach().clone())
        z = torch.zeros(basis.shape[1], dtype=theta0.dtype, device=theta0.device)
        if z_init_std > 0:
            z.normal_(0.0, z_init_std)
        self.z = nn.Parameter(z)

    @property
    def num_trainable(self) -> int:
        return int(self.z.numel())

    @property
    def num_generated(self) -> int:
        return self.target.num_params

    def theta(self) -> Tensor:
        return self.theta0 + self.basis @ self.z

    def forward(self, x: Tensor) -> Tensor:
        return self.target(x, self.theta())


def subspace_weight_decay(model: SCMappingNetwork) -> Tensor:
    return model.z.pow(2).mean()
