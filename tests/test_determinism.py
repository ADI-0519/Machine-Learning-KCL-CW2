from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.data import make_subset_loader
from src.protocol import SeedBundle, derive_seed
from src.seed import seed_worker, set_seed


def _loader_order(seed: int) -> list[int]:
    dataset = TensorDataset(torch.arange(24))
    loader = make_subset_loader(
        dataset=dataset,
        indices=list(range(24)),
        batch_size=5,
        shuffle=True,
        num_workers=0,
        seed=seed,
    )
    return [int(value) for batch in loader for value in batch[0]]


def test_seed_derivation_has_fixed_known_outputs() -> None:
    tpcrp = SeedBundle.for_round(
        replicate=42,
        dataset="cifar10",
        framework="ssl_embedding",
        method="tpcrp",
        round_id=1,
    )
    assert tpcrp == SeedBundle(
        replicate=42,
        clustering=824551655,
        selector=934712551,
        training=1924623876,
        dataloader=200152058,
    )
    assert derive_seed(42, "cifar10", "ssl_embedding", 1, "clustering") == 824551655


def test_typiclust_ablation_family_shares_all_nuisance_randomness() -> None:
    common = {
        "replicate": 42,
        "dataset": "cifar10",
        "framework": "ssl_embedding",
        "round_id": 1,
    }
    tpcrp = SeedBundle.for_round(method="tpcrp", **common)
    ccfl = SeedBundle.for_round(method="tpcrp_ccfl", **common)
    unweighted = SeedBundle.for_round(method="ccfl_unweighted", **common)
    random_method = SeedBundle.for_round(method="random", **common)

    assert tpcrp.clustering == ccfl.clustering
    assert tpcrp.training == ccfl.training
    assert tpcrp.dataloader == ccfl.dataloader
    assert tpcrp.selector == ccfl.selector == unweighted.selector
    assert random_method.selector != tpcrp.selector


def test_subset_loader_order_is_repeatable_and_seed_sensitive() -> None:
    assert _loader_order(101) == _loader_order(101)
    assert _loader_order(101) != _loader_order(102)


def test_set_seed_repeats_python_numpy_and_torch_streams() -> None:
    set_seed(88)
    first = (random.random(), np.random.random(), torch.rand(1).item())
    set_seed(88)
    second = (random.random(), np.random.random(), torch.rand(1).item())
    assert first == second
    assert torch.are_deterministic_algorithms_enabled()


def test_seed_worker_repeats_python_and_numpy_from_torch_seed(monkeypatch) -> None:
    monkeypatch.setattr(torch, "initial_seed", lambda: 123456)
    seed_worker(0)
    first = (random.random(), np.random.random())
    seed_worker(999)
    second = (random.random(), np.random.random())
    assert first == second
