"""Dataset helpers for quick Mapping Network experiments."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class SyntheticImageDataset(Dataset):
    """Small deterministic image classification dataset for offline smoke tests."""

    def __init__(self, n: int = 4096, num_classes: int = 10, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        y = torch.randint(0, num_classes, (n,), generator=g)
        prototypes = torch.randn(num_classes, 1, 28, 28, generator=g) * 0.7
        x = prototypes[y] + 0.25 * torch.randn(n, 1, 28, 28, generator=g)
        self.x = x.clamp(-3, 3)
        self.y = y

    def __len__(self) -> int:
        return int(self.y.numel())

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def make_loaders(
    dataset: str = "synthetic",
    data_root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    dataset = dataset.lower()
    if dataset in {"mnist", "fashionmnist", "fmnist"}:
        from torchvision import datasets, transforms

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        if dataset == "mnist":
            train_set = datasets.MNIST(data_root, train=True, transform=transform, download=download)
            test_set = datasets.MNIST(data_root, train=False, transform=transform, download=download)
        else:
            train_set = datasets.FashionMNIST(data_root, train=True, transform=transform, download=download)
            test_set = datasets.FashionMNIST(data_root, train=False, transform=transform, download=download)
    elif dataset == "synthetic":
        full = SyntheticImageDataset(n=4096, seed=0)
        train_set, test_set = random_split(full, [3072, 1024], generator=torch.Generator().manual_seed(0))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
