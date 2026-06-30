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

from sc_mapping.mapping_network import MappingLossWeights, OriginalMappingNetwork, mapping_loss, mapping_regularizers
from sc_mapping.targets import make_target
from sc_mapping.data import make_loaders
from sc_mapping.engine import evaluate, train_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "mnist", "fashionmnist", "fmnist"])
    p.add_argument("--target", default="cnn", choices=["cnn", "mlp"])
    p.add_argument("--latent-dim", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--output", default="runs/mapnet_metrics.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = make_loaders(args.dataset, args.data_root, args.batch_size, download=not args.no_download)
    target = make_target(args.target)
    model = OriginalMappingNetwork(target, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(args.device)
    weights = MappingLossWeights()
    optimizer = torch.optim.AdamW([model.z], lr=args.lr, weight_decay=1e-4)

    def loss_fn(m, x, y, logits):
        task = F.cross_entropy(logits, y)
        regs = mapping_regularizers(m, x, logits)
        return mapping_loss(task, regs, weights)

    history = []
    print(f"target_params={target.num_params:,} trainable_params={model.num_trainable:,}")
    for epoch in range(1, args.epochs + 1):
        train = train_epoch(model, train_loader, optimizer, args.device, loss_fn=loss_fn)
        test = evaluate(model, test_loader, args.device)
        row = {"epoch": epoch, "train_loss": train.loss, "train_acc": train.acc, "test_loss": test.loss, "test_acc": test.acc}
        history.append(row)
        print(row)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "history": history}, f, indent=2)


if __name__ == "__main__":
    main()
