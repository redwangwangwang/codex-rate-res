#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_mapping.subspace import SCMappingNetwork, compute_gradient_basis, subspace_weight_decay
from sc_mapping.targets import make_target
from sc_mapping.data import make_loaders
from sc_mapping.engine import evaluate, train_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "mnist", "fashionmnist", "fmnist"])
    p.add_argument("--target", default="cnn", choices=["cnn", "mlp"])
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--basis-batches", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--z-weight-decay", type=float, default=1e-4)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--output", default="runs/sc_mapping_metrics.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = make_loaders(args.dataset, args.data_root, args.batch_size, download=not args.no_download)
    target = make_target(args.target)
    theta0 = target.init_theta(seed=args.seed, device=args.device)
    basis, stats = compute_gradient_basis(target, theta0, train_loader, rank=args.rank, num_batches=args.basis_batches, device=args.device)
    model = SCMappingNetwork(target, theta0, basis).to(args.device)
    optimizer = torch.optim.AdamW([model.z], lr=args.lr)

    def loss_fn(m, x, y, logits):
        return F.cross_entropy(logits, y) + args.z_weight_decay * subspace_weight_decay(m)

    history = []
    print(f"target_params={target.num_params:,} trainable_params={model.num_trainable:,} basis_rank={basis.shape[1]} retained_gradient_energy={stats.retained_energy:.4f}")
    for epoch in range(1, args.epochs + 1):
        train = train_epoch(model, train_loader, optimizer, args.device, loss_fn=loss_fn)
        test = evaluate(model, test_loader, args.device)
        row = {"epoch": epoch, "train_loss": train.loss, "train_acc": train.acc, "test_loss": test.loss, "test_acc": test.acc}
        history.append(row)
        print(row)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "target_params": target.num_params, "trainable_params": model.num_trainable, "retained_gradient_energy": stats.retained_energy, "history": history}, f, indent=2)


if __name__ == "__main__":
    main()
