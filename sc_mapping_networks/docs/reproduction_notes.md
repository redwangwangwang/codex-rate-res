# Reproduction notes

## Code-release status

As of 2026-06-30, I found the arXiv paper, PDF, HTML, and TeX source for `Mapping Networks`, but I did not find an author-owned code repository linked from the paper page. GitHub search returned one third-party reproduction repository, `ZenoAFfectionate/Mapping-Networks-Reproduction`; this scaffold is independent from that repository.

## What is reproduced here

This folder implements a minimal PyTorch reproduction of the central mechanism:

1. The target CNN/MLP is represented by a flat parameter vector `theta`.
2. The target network is not optimized directly.
3. `OriginalMappingNetwork` optimizes a compact latent vector `z` and generates `theta=g(z)` through frozen orthogonal mapping weights.
4. The Mapping Loss includes task, stability, smoothness, and alignment terms.

The implementation is designed for MNIST/FashionMNIST/synthetic sanity experiments rather than exact reproduction of every table in the paper.

## What is extended

`SCMappingNetwork` implements the proposed Subspace-Certified Mapping Networks:

```text
theta(z) = theta0 + U z
```

where `U` is a frozen data-aligned tangent basis estimated from mini-batch gradient sketches. This gives an explicit projection residual and a clean excess-risk decomposition.
