from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

import src.experiment as experiment
import src.train_classifier as classifier_training
from src.protocol import PROTOCOL_VERSION, TrainingOutcome
from src.seed import set_seed


class TinyClassifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(4, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def _assert_training_only_history(outcome: TrainingOutcome, epochs: int) -> None:
    assert outcome.metrics.trained_epochs == epochs
    assert len(outcome.history) == epochs
    assert all("test" not in key.lower() for row in outcome.history for key in row)


def test_supervised_training_evaluates_test_loader_once(
    monkeypatch,
    tmp_path,
    tiny_classification_loaders,
) -> None:
    train_loader, test_loader = tiny_classification_loaders
    monkeypatch.setattr(classifier_training, "CIFARClassifier", TinyClassifier)

    original_evaluate = classifier_training.evaluate_classifier
    calls = 0

    def counted_evaluate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(classifier_training, "evaluate_classifier", counted_evaluate)
    checkpoint_path = tmp_path / "tiny.pt"
    set_seed(7)
    outcome = classifier_training.train_classifier(
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=2,
        epochs=3,
        lr=0.05,
        momentum=0.0,
        weight_decay=0.0,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
        verbose=False,
    )

    assert calls == 1
    _assert_training_only_history(outcome, epochs=3)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["epoch"] == 3
    assert payload["metrics"] == {
        "protocol_version": PROTOCOL_VERSION,
        **outcome.metrics.to_dict(),
    }


def test_frozen_embedding_training_computes_test_metrics_once(
    monkeypatch,
    synthetic_embeddings,
) -> None:
    train_embeddings, test_embeddings, train_labels, test_labels = synthetic_embeddings
    selected = np.array([0, 1, 8, 9], dtype=int)
    accuracy_calls = 0
    loss_calls = 0
    original_accuracy = experiment.accuracy_score
    original_loss = experiment.log_loss

    def counted_accuracy(*args, **kwargs):
        nonlocal accuracy_calls
        accuracy_calls += 1
        return original_accuracy(*args, **kwargs)

    def counted_loss(*args, **kwargs):
        nonlocal loss_calls
        loss_calls += 1
        return original_loss(*args, **kwargs)

    monkeypatch.setattr(experiment, "accuracy_score", counted_accuracy)
    monkeypatch.setattr(experiment, "log_loss", counted_loss)
    set_seed(19)
    outcome = experiment._train_eval_ssl_embedding(
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        train_labels=train_labels,
        test_labels=test_labels,
        selected_indices=selected,
        epochs=4,
        classifier_cfg={
            "lr": 0.01,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "ssl_embedding_lr": 1.0,
            "ssl_embedding_dropout_p": 0.2,
        },
        device=torch.device("cpu"),
        num_classes=2,
    )

    assert accuracy_calls == 1
    assert loss_calls == 1
    _assert_training_only_history(outcome, epochs=4)


def test_label_spreading_proxy_returns_common_outcome_schema(synthetic_embeddings) -> None:
    train_embeddings, test_embeddings, train_labels, test_labels = synthetic_embeddings
    selected = np.array([0, 1, 2, 8, 9, 10], dtype=int)
    outcome = experiment._train_eval_label_spreading_proxy(
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        train_labels=train_labels,
        test_labels=test_labels,
        selected_indices=selected,
        num_classes=2,
    )

    assert isinstance(outcome, TrainingOutcome)
    assert outcome.metrics.trained_epochs == 1
    assert 0.0 <= outcome.metrics.test_accuracy <= 1.0
    assert np.isfinite(outcome.metrics.test_loss)
    assert outcome.history == []


def test_evaluators_reject_nonpositive_epoch_counts(tiny_classification_loaders) -> None:
    train_loader, test_loader = tiny_classification_loaders
    with pytest.raises(ValueError, match="epochs must be positive"):
        classifier_training.train_classifier(
            train_loader=train_loader,
            test_loader=test_loader,
            num_classes=2,
            epochs=0,
            lr=0.1,
            momentum=0.0,
            weight_decay=0.0,
            device=torch.device("cpu"),
            verbose=False,
        )
