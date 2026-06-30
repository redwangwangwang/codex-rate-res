"""Shared training/evaluation loops."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class Metrics:
    loss: float
    acc: float


def evaluate(model: nn.Module, loader: Iterable[Tuple[Tensor, Tensor]], device: torch.device | str) -> Metrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            total_loss += float(loss.item()) * y.numel()
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
    return Metrics(total_loss / max(total, 1), total_correct / max(total, 1))


def train_epoch(
    model: nn.Module,
    loader: Iterable[Tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    loss_fn: Callable[[nn.Module, Tensor, Tensor, Tensor], Tensor] | None = None,
) -> Metrics:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        task_loss = F.cross_entropy(logits, y)
        loss = loss_fn(model, x, y, logits) if loss_fn is not None else task_loss
        loss.backward()
        optimizer.step()
        total_loss += float(task_loss.item()) * y.numel()
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total += int(y.numel())
    return Metrics(total_loss / max(total, 1), total_correct / max(total, 1))
