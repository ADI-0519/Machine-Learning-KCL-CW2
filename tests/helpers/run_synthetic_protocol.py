"""Execute one deterministic synthetic protocol run in a fresh Python process."""

from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn as nn

import src.experiment as experiment
from src.protocol import EvaluationMetrics, TrainingOutcome


class TargetsOnlyDataset:
    """Minimal dataset surface needed by the experiment orchestration layer."""

    def __init__(self, size: int) -> None:
        self.targets = [index % 2 for index in range(size)]


def synthetic_train_eval(**kwargs) -> TrainingOutcome:
    """Return deterministic metrics from the cumulative selected-set size."""
    selected_count = len(kwargs["selected_indices"])
    return TrainingOutcome(
        model=nn.Identity(),
        metrics=EvaluationMetrics(
            trained_epochs=2,
            test_loss=selected_count / 100.0,
            test_accuracy=selected_count / 30.0,
        ),
        history=[],
    )


def main() -> None:
    """Patch only external data/training boundaries and run real orchestration."""
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_synthetic_protocol.py CONFIG_PATH")
    experiment.get_device = lambda: torch.device("cpu")
    experiment.get_cifar10_train = lambda **_kwargs: TargetsOnlyDataset(30)
    experiment.get_cifar10_test = lambda **_kwargs: TargetsOnlyDataset(12)
    experiment._train_eval_fully_supervised = synthetic_train_eval
    experiment.load_or_compute_embeddings = lambda **kwargs: (
        np.arange(120, dtype=np.float32).reshape(30, 4)
        if kwargs["split"] == "train"
        else np.arange(48, dtype=np.float32).reshape(12, 4)
    )
    experiment.collect_environment = lambda **_kwargs: {
        "git_commit": "e" * 40,
        "git_dirty": False,
        "python_version": "3.11.5",
        "platform": "synthetic-test-platform",
        "packages": {"torch": torch.__version__},
        "device": "cpu",
        "cuda_runtime": None,
        "deterministic_algorithms": True,
        "cublas_workspace_config": None,
        "checkpoint_path": "unused.pt",
        "checkpoint_sha256": "f" * 64,
    }
    experiment.run_single_experiment(
        config_path=sys.argv[1],
        method="random",
        seed=41,
        framework="fully_supervised",
    )


if __name__ == "__main__":
    main()
