from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def synthetic_embeddings() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic, linearly separable two-class embedding data."""
    rng = np.random.default_rng(2026)
    negative_train = rng.normal(loc=-1.0, scale=0.15, size=(8, 4))
    positive_train = rng.normal(loc=1.0, scale=0.15, size=(8, 4))
    negative_test = rng.normal(loc=-1.0, scale=0.15, size=(4, 4))
    positive_test = rng.normal(loc=1.0, scale=0.15, size=(4, 4))

    train_embeddings = np.vstack([negative_train, positive_train]).astype(np.float32)
    test_embeddings = np.vstack([negative_test, positive_test]).astype(np.float32)
    train_labels = np.array([0] * 8 + [1] * 8, dtype=np.int64)
    test_labels = np.array([0] * 4 + [1] * 4, dtype=np.int64)
    return train_embeddings, test_embeddings, train_labels, test_labels


@pytest.fixture
def tiny_classification_loaders() -> tuple[DataLoader, DataLoader]:
    """Return small in-memory loaders that require neither downloads nor CUDA."""
    generator = torch.Generator().manual_seed(17)
    train_x = torch.randn(12, 4, generator=generator)
    train_y = (train_x.sum(dim=1) > 0).long()
    test_x = torch.randn(6, 4, generator=generator)
    test_y = (test_x.sum(dim=1) > 0).long()
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=4, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=3, shuffle=False)
    return train_loader, test_loader
