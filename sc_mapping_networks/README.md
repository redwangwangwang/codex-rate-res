# SC-Mapping Networks

Independent reproduction scaffold for **Mapping Networks** plus a theoretically cleaner extension: **Subspace-Certified Mapping Networks (SC-Mapping)**.

> Status: I did not find an official author-released GitHub repository for the CVPR 2026 paper. This folder therefore starts from an independent minimal reproduction of the paper's core latent-to-weight idea, then adds the proposed certifiable subspace version.

## Why this exists

The original Mapping Networks paper argues that neural-network weights can be optimized through a compact latent vector `z`, with a mapping network generating all target parameters. The central concern is that its theorem is mostly an existence argument: it assumes a low-dimensional weight manifold and shows that a smooth mapping can exist.

SC-Mapping changes the claim into something measurable and provable:

```text
theta(z) = theta0 + U z
```

where `U` is a frozen data-aligned tangent basis estimated from mini-batch gradients. The excess risk decomposes into projection error, optimization error, and generalization error.

See [`docs/theory.md`](docs/theory.md).

## Repository layout

```text
src/sc_mapping/
  targets.py          # flat-parameter CNN/MLP target networks
  mapping_network.py  # minimal original-style Mapping Network reproduction
  subspace.py         # SC-Mapping: theta = theta0 + U z
  data.py             # MNIST/FashionMNIST/synthetic loaders
  engine.py           # train/eval loops
scripts/
  train_mapnet.py     # original-style Mapping Network training
  train_sc_mapping.py # Subspace-Certified Mapping training
configs/
  *.yaml              # example settings
docs/
  theory.md           # theorem sketch and proof decomposition
  reproduction_notes.md
```

## Installation

```bash
pip install -r sc_mapping_networks/requirements.txt
```

## Quick smoke test, no download needed

```bash
cd sc_mapping_networks
python scripts/train_mapnet.py --dataset synthetic --epochs 2
python scripts/train_sc_mapping.py --dataset synthetic --rank 16 --basis-batches 16 --epochs 2
```

## MNIST/FashionMNIST experiments

```bash
python scripts/train_mapnet.py --dataset mnist --target cnn --latent-dim 512 --hidden-dim 256 --epochs 10
python scripts/train_sc_mapping.py --dataset mnist --target cnn --rank 64 --basis-batches 64 --epochs 10
```

For FashionMNIST:

```bash
python scripts/train_mapnet.py --dataset fashionmnist --target cnn --epochs 10
python scripts/train_sc_mapping.py --dataset fashionmnist --target cnn --rank 64 --basis-batches 64 --epochs 10
```

## Main conceptual difference

| Method | Weight parameterization | Trainable variables | What can be certified? |
|---|---|---:|---|
| Original-style Mapping Network | `theta = g(z)` | `z` | weak existence-style justification |
| SC-Mapping | `theta = theta0 + U z` | `z` | projection + optimization + generalization bound |

## Important limitations

This is a starting scaffold, not a completed exact reproduction of all CVPR 2026 experiments. It currently focuses on image-classification sanity checks and the core algorithmic comparison. The next step is to add full baselines: direct full training, random subspace, LoRA/adapters, and stronger CNN/ResNet backbones.
