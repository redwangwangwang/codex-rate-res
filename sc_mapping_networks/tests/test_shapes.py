import pathlib
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_mapping.mapping_network import OriginalMappingNetwork
from sc_mapping.subspace import SCMappingNetwork, compute_gradient_basis
from sc_mapping.targets import make_target


def test_original_mapping_shapes():
    target = make_target("mlp")
    model = OriginalMappingNetwork(target, latent_dim=16, hidden_dim=32)
    x = torch.randn(4, 1, 28, 28)
    logits = model(x)
    assert logits.shape == (4, 10)
    assert model.theta().numel() == target.num_params
    assert model.num_trainable == 16


def test_sc_mapping_shapes():
    target = make_target("mlp")
    theta0 = target.init_theta(seed=0)
    x = torch.randn(16, 1, 28, 28)
    y = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    basis, stats = compute_gradient_basis(target, theta0, loader, rank=2, num_batches=2)
    model = SCMappingNetwork(target, theta0, basis)
    logits = model(x[:4])
    assert logits.shape == (4, 10)
    assert model.num_trainable == 2
    assert 0.0 <= stats.retained_energy <= 1.0
