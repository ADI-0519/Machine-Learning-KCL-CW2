from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest
import torch.nn as nn

import src.experiment as experiment
from src.artifacts import artifact_path_for, config_digest
from src.config import CCFL_METHODS, load_configurations, validate_protocol_config
from src.diagnostics import selection_diagnostics
from src.protocol import (
    EvaluationMetrics,
    SeedBundle,
    TrainingOutcome,
    validate_query,
    validate_round_query_sizes,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ([10], [10]),
        ([10, 10, 20], [10, 10, 20]),
        ((1, 2, 3), [1, 2, 3]),
        ([np.int64(4), np.int32(5)], [4, 5]),
    ],
)
def test_validate_round_query_sizes_accepts_positive_integers(raw, expected) -> None:
    assert validate_round_query_sizes(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [[], [0], [-1], [10, 0], [True], [1.5], ["10"]],
)
def test_validate_round_query_sizes_rejects_invalid_schedules(raw) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        validate_round_query_sizes(raw)


def test_validate_query_accepts_unique_global_pool_indices() -> None:
    validate_query(
        np.array([11, 4, 19], dtype=int),
        pool_indices=np.array([4, 7, 11, 19], dtype=int),
        query_size=3,
    )


def test_validate_query_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_query(
            np.array([[4, 7]], dtype=int),
            pool_indices=np.array([4, 7, 11], dtype=int),
            query_size=2,
        )


def test_validate_query_rejects_wrong_size() -> None:
    with pytest.raises(ValueError, match="expected 2 query indices, got 1"):
        validate_query(
            np.array([4], dtype=int),
            pool_indices=np.array([4, 7, 11], dtype=int),
            query_size=2,
        )


def test_validate_query_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        validate_query(
            np.array([4, 4], dtype=int),
            pool_indices=np.array([4, 7, 11], dtype=int),
            query_size=2,
        )


@pytest.mark.parametrize("invalid_index", [2, 999])
def test_validate_query_rejects_labeled_or_out_of_range_indices(invalid_index: int) -> None:
    with pytest.raises(ValueError, match="outside the unlabeled pool"):
        validate_query(
            np.array([4, invalid_index], dtype=int),
            pool_indices=np.array([4, 7, 11], dtype=int),
            query_size=2,
        )


def test_pool_local_adapter_returns_global_indices() -> None:
    query = experiment._pool_local_to_global_query(
        np.array([2, 0], dtype=int),
        pool_indices=np.array([4, 11, 19], dtype=int),
        query_size=2,
    )
    assert query.tolist() == [19, 4]


def test_embedding_loader_rejects_unknown_split_without_loading_data() -> None:
    with pytest.raises(ValueError, match="unsupported embedding split"):
        experiment.build_embedding_loader(
            data_root="unused",
            split="validation",
            batch_size=4,
            num_workers=0,
            seed=1,
        )


def test_protocol_run_rejects_dirty_git_before_loading_data(monkeypatch, tmp_path) -> None:
    config = load_configurations("configs/protocol_v2_pilot.yaml")
    config["output"]["root"] = str(tmp_path / "outputs")
    monkeypatch.setattr(experiment, "load_configurations", lambda _path: config)
    monkeypatch.setattr(experiment, "get_device", lambda: experiment.torch.device("cpu"))
    monkeypatch.setattr(
        experiment,
        "collect_environment",
        lambda **_kwargs: {
            "git_dirty": True,
            "checkpoint_sha256": "f" * 64,
        },
    )

    with pytest.raises(RuntimeError, match="clean Git worktree"):
        experiment.run_single_experiment(
            config_path="unused.yaml",
            method="tpcrp",
            seed=42,
            framework="ssl_embedding",
        )


def test_two_round_run_is_nested_and_evaluates_twice(monkeypatch, tmp_path) -> None:
    config = {
        "protocol": {
            "version": "2.0",
            "deterministic": True,
            "test_evaluations_per_round": 1,
        },
        "output": {"root": str(tmp_path / "outputs"), "resume": True},
        "data": {
            "name": "cifar10",
            "root": "unused",
            "num_workers": 0,
            "num_classes": 10,
        },
        "representation": {
            "backend": "simclr",
            "checkpoint_path": "unused.pt",
            "projection_dim": 4,
            "batch_size": 8,
            "embedding_seed": 21,
        },
        "selection": {
            "round_query_sizes": [10, 10],
            "knn_k": 3,
            "max_clusters": 20,
            "min_cluster_size": 2,
        },
        "evaluation": {
            "framework": "fully_supervised",
            "batch_size": 4,
            "epochs": 2,
            "lr": 0.01,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "dropout_p": 0.0,
            "test_policy": "once_after_fixed_epochs",
        },
        "experiment": {
            "methods": ["random"],
            "replicate_seeds": [42],
            "primary_comparison": {
                "method_a": "random",
                "method_b": "random",
                "cumulative_budget": 10,
                "metric": "test_accuracy",
            },
        },
        "ccfl_variants": {
            "tpcrp_ccfl": {
                "candidates_per_cluster": 2,
                "refine_steps": 1,
                "use_cluster_weights": True,
            },
            "ccfl_candidate_only": {
                "candidates_per_cluster": 2,
                "refine_steps": 0,
                "use_cluster_weights": False,
            },
            "ccfl_unweighted": {
                "candidates_per_cluster": 2,
                "refine_steps": 1,
                "use_cluster_weights": False,
            },
            "ccfl_weighted": {
                "candidates_per_cluster": 2,
                "refine_steps": 1,
                "use_cluster_weights": True,
            },
        },
        "probcover": {
            "alpha": 0.95,
            "delta_search": {"minimum": 0.05, "maximum": 0.10, "step": 0.01},
        },
    }

    class TargetsOnlyDataset:
        def __init__(self, size: int) -> None:
            self.targets = [index % 2 for index in range(size)]

    evaluator_inputs: list[list[int]] = []
    trainer_seed_inputs: list[tuple[int, int]] = []
    seed_calls: list[int] = []

    def fake_train_eval(**kwargs) -> TrainingOutcome:
        selected = kwargs["selected_indices"].tolist()
        evaluator_inputs.append(selected)
        trainer_seed_inputs.append((seed_calls[-1], kwargs["dataloader_seed"]))
        return TrainingOutcome(
            model=nn.Identity(),
            metrics=EvaluationMetrics(
                trained_epochs=2,
                test_loss=float(len(selected)) / 100.0,
                test_accuracy=float(len(selected)) / 30.0,
            ),
            history=[],
        )

    monkeypatch.setattr(experiment, "load_configurations", lambda _path: config)
    monkeypatch.setattr(
        experiment,
        "get_cifar10_train",
        lambda **_kwargs: TargetsOnlyDataset(30),
    )
    monkeypatch.setattr(
        experiment,
        "get_cifar10_test",
        lambda **_kwargs: TargetsOnlyDataset(12),
    )
    monkeypatch.setattr(experiment, "_train_eval_fully_supervised", fake_train_eval)
    monkeypatch.setattr(
        experiment,
        "load_or_compute_embeddings",
        lambda **kwargs: (
            np.arange(120, dtype=np.float32).reshape(30, 4)
            if kwargs["split"] == "train"
            else np.arange(48, dtype=np.float32).reshape(12, 4)
        ),
    )
    monkeypatch.setattr(experiment, "set_seed", seed_calls.append)
    monkeypatch.setattr(
        experiment,
        "collect_environment",
        lambda **_kwargs: {
            "git_commit": "c" * 40,
            "git_dirty": False,
            "python_version": "3.11.5",
            "platform": "Windows-test",
            "packages": {"torch": "2.8.0+cpu"},
            "device": "cpu",
            "cuda_runtime": None,
            "deterministic_algorithms": True,
            "cublas_workspace_config": None,
            "checkpoint_path": "unused.pt",
            "checkpoint_sha256": "f" * 64,
        },
    )

    output = experiment.run_single_experiment(
        config_path="unused.yaml",
        method="random",
        seed=42,
        framework="fully_supervised",
    )

    assert len(evaluator_inputs) == 2
    assert [len(indices) for indices in evaluator_inputs] == [10, 20]
    assert evaluator_inputs[1][:10] == evaluator_inputs[0]
    assert len(set(evaluator_inputs[1])) == 20

    expected_round_seeds = [
        SeedBundle.for_round(
            replicate=42,
            dataset="cifar10",
            framework="fully_supervised",
            method="random",
            round_id=round_id,
        )
        for round_id in (1, 2)
    ]
    assert seed_calls == [
        42,
        expected_round_seeds[0].selector,
        expected_round_seeds[0].training,
        expected_round_seeds[1].selector,
        expected_round_seeds[1].training,
    ]
    assert trainer_seed_inputs == [
        (expected.training, expected.dataloader) for expected in expected_round_seeds
    ]

    artifact_path = artifact_path_for(config["output"]["root"], output["run_id"])
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    rounds = payload["rounds"]
    expected_round_keys = {
        "round",
        "query_size",
        "cumulative_budget",
        "new_indices",
        "selected_indices",
        "seeds",
        "trained_epochs",
        "test_loss",
        "test_accuracy",
        "selection_diagnostics",
        "method_metadata",
    }
    assert all(set(record) == expected_round_keys for record in rounds)
    assert [record["query_size"] for record in rounds] == [10, 10]
    assert [record["cumulative_budget"] for record in rounds] == [10, 20]
    assert rounds[1]["selected_indices"][:10] == rounds[0]["selected_indices"]
    assert set(rounds[0]["new_indices"]).isdisjoint(rounds[1]["new_indices"])
    expected_diagnostics = {
        "nearest_distance_mean",
        "nearest_distance_p95",
        "nearest_distance_max",
        "selected_pairwise_cosine_mean",
        "selected_typicality_mean",
    }
    assert all(set(record["selection_diagnostics"]) == expected_diagnostics for record in rounds)
    assert all(record["method_metadata"] == {} for record in rounds)
    assert len(payload["timings"]["selector_seconds"]) == 2

    first_artifact = deepcopy(output)
    artifact_path.unlink()
    second_artifact = experiment.run_single_experiment(
        config_path="unused.yaml",
        method="random",
        seed=42,
        framework="fully_supervised",
    )
    first_artifact.pop("timings")
    second_artifact.pop("timings")
    assert second_artifact == first_artifact


def test_cluster_round_returns_global_indices_and_uses_clustering_seed(monkeypatch) -> None:
    embeddings = np.arange(12, dtype=np.float32).reshape(6, 2)
    pool_indices = np.array([1, 3, 4, 5], dtype=int)
    labeled_indices = np.array([0, 2], dtype=int)
    observed_seeds: list[int] = []

    def fixed_clusters(*, embeddings, n_clusters, random_state):
        observed_seeds.append(random_state)
        assert n_clusters == 4
        labels = np.array([0, 0, 1, 1, 2, 3], dtype=int)
        centroids = np.zeros((4, embeddings.shape[1]), dtype=np.float32)
        return labels, centroids

    monkeypatch.setattr(experiment, "cluster_embeddings", fixed_clusters)
    query = experiment._select_cluster_based_round(
        method="tpcrp",
        full_embeddings=embeddings,
        pool_indices=pool_indices,
        labeled_indices=labeled_indices,
        query_size=2,
        knn_k=2,
        rng=np.random.default_rng(123),
        max_clusters=10,
        min_cluster_size=2,
        ccfl_variant=None,
        clustering_seed=987654,
    )

    assert observed_seeds == [987654]
    assert query.tolist() == [4, 5]
    validate_query(query, pool_indices=pool_indices, query_size=2)


def test_probcover_radius_cache_is_keyed_by_all_inputs(monkeypatch, tmp_path) -> None:
    embeddings = np.array([[0.0], [0.1], [2.0], [2.1]], dtype=np.float64)
    candidates = np.array([0.05, 0.1, 0.2], dtype=np.float64)
    calls: list[dict] = []

    def fake_estimator(values, **kwargs):
        calls.append({"embeddings": values.copy(), **kwargs})
        return 0.1

    monkeypatch.setattr(experiment, "estimate_probcover_delta", fake_estimator)
    first = experiment._load_or_estimate_probcover_delta(
        cache_root=tmp_path,
        embeddings=embeddings,
        num_classes=2,
        candidates=candidates,
        alpha=0.95,
        clustering_seed=123,
    )
    second = experiment._load_or_estimate_probcover_delta(
        cache_root=tmp_path,
        embeddings=embeddings.copy(),
        num_classes=2,
        candidates=candidates.copy(),
        alpha=0.95,
        clustering_seed=123,
    )

    assert first == second
    assert first[0] == 0.1
    assert len(first[1]) == 64
    assert len(calls) == 1

    changed = experiment._load_or_estimate_probcover_delta(
        cache_root=tmp_path,
        embeddings=embeddings,
        num_classes=2,
        candidates=candidates,
        alpha=0.90,
        clustering_seed=123,
    )
    assert changed[1] != first[1]
    assert len(calls) == 2


def test_synthetic_benchmark_reports_diagnostics_for_typiclust_and_ccfl_variants() -> None:
    embeddings = np.array(
        [
            [-3.0, 0.0],
            [-3.1, 0.1],
            [-2.9, -0.1],
            [-3.0, 0.2],
            [0.0, 3.0],
            [0.1, 3.1],
            [-0.1, 2.9],
            [0.2, 3.0],
            [3.0, 0.0],
            [3.1, 0.1],
            [2.9, -0.1],
            [3.0, 0.2],
        ],
        dtype=np.float64,
    )
    pool = np.arange(len(embeddings), dtype=int)
    typicality = np.linspace(1.0, 2.0, len(embeddings))
    variants = {
        "tpcrp_ccfl": {
            "candidates_per_cluster": 2,
            "refine_steps": 1,
            "use_cluster_weights": True,
        },
        "ccfl_candidate_only": {
            "candidates_per_cluster": 2,
            "refine_steps": 0,
            "use_cluster_weights": False,
        },
        "ccfl_unweighted": {
            "candidates_per_cluster": 2,
            "refine_steps": 1,
            "use_cluster_weights": False,
        },
        "ccfl_weighted": {
            "candidates_per_cluster": 2,
            "refine_steps": 1,
            "use_cluster_weights": True,
        },
    }

    for method in ["tpcrp", *sorted(CCFL_METHODS)]:
        kwargs = {
            "method": method,
            "full_embeddings": embeddings,
            "pool_indices": pool,
            "labeled_indices": np.array([], dtype=int),
            "query_size": 3,
            "knn_k": 2,
            "max_clusters": 20,
            "min_cluster_size": 2,
            "ccfl_variant": variants.get(method),
            "probcover_delta": None,
            "clustering_seed": 991,
        }
        first = experiment._select_protocol_round(rng=np.random.default_rng(123), **kwargs)
        second = experiment._select_protocol_round(rng=np.random.default_rng(123), **kwargs)
        np.testing.assert_array_equal(first, second)
        validate_query(first, pool_indices=pool, query_size=3)

        remaining = np.setdiff1d(pool, first)
        diagnostics = selection_diagnostics(
            embeddings,
            first,
            evaluation_indices=remaining,
            selected_typicality_scores=typicality[first],
        )
        assert set(diagnostics) == {
            "nearest_distance_mean",
            "nearest_distance_p95",
            "nearest_distance_max",
            "selected_pairwise_cosine_mean",
            "selected_typicality_mean",
        }
        assert all(np.isfinite(value) for value in diagnostics.values())


def test_protocol_v2_pilot_config_locks_initial_comparison() -> None:
    config = load_configurations("configs/protocol_v2_pilot.yaml")
    assert set(config) == {
        "protocol",
        "output",
        "data",
        "representation",
        "selection",
        "evaluation",
        "experiment",
        "ccfl_variants",
        "probcover",
    }
    assert config["protocol"]["version"] == "2.0"
    assert config["selection"]["round_query_sizes"] == [10, 10, 10, 10, 10]
    assert config["experiment"]["methods"] == [
        "random",
        "tpcrp",
        "kcenter",
        "probcover",
        "tpcrp_ccfl",
        "ccfl_candidate_only",
        "ccfl_unweighted",
    ]
    assert config["experiment"]["primary_comparison"] == {
        "method_a": "tpcrp_ccfl",
        "method_b": "tpcrp",
        "cumulative_budget": 10,
        "metric": "test_accuracy",
    }
    validate_protocol_config(config)


@pytest.mark.parametrize(
    ("path", "backend", "methods", "expected_digest"),
    [
        (
            "configs/protocol_v2_pilot.yaml",
            "simclr",
            [
                "random",
                "tpcrp",
                "kcenter",
                "probcover",
                "tpcrp_ccfl",
                "ccfl_candidate_only",
                "ccfl_unweighted",
            ],
            "3d935adfb8825d34f33d955bd907e45cc46a5e616fab44e81772d32645caecd5",
        ),
        (
            "configs/protocol_v2_confirmation.yaml",
            "simclr",
            ["tpcrp", "tpcrp_ccfl", "ccfl_candidate_only", "ccfl_unweighted"],
            "c4f786bf1e51e9f65e8a461185df7007474e96f885119bc65d038c868b70017f",
        ),
        (
            "configs/protocol_v2_dinov2.yaml",
            "dinov2",
            ["tpcrp", "tpcrp_ccfl"],
            "37d8dbc22da725d5c414cc56fad579287b8a63e73fc7ff69ac509dc51ce3a70e",
        ),
    ],
)
def test_claim_configs_have_frozen_digests(path, backend, methods, expected_digest) -> None:
    config = load_configurations(path)
    validate_protocol_config(config)

    assert config["representation"]["backend"] == backend
    assert config["experiment"]["methods"] == methods
    if "confirmation" in path or "dinov2" in path:
        assert config["experiment"]["replicate_seeds"] == list(range(42, 52))
    assert config_digest(config) == expected_digest


@pytest.mark.parametrize(
    ("variant", "field", "value", "message"),
    [
        ("ccfl_candidate_only", "refine_steps", 1, "refine_steps=0"),
        ("ccfl_unweighted", "use_cluster_weights", True, "disable cluster weights"),
        ("ccfl_weighted", "use_cluster_weights", False, "enable cluster weights"),
        ("tpcrp_ccfl", "refine_steps", 0, "enable facility-location"),
    ],
)
def test_protocol_config_rejects_semantically_mislabeled_ccfl_variants(
    variant: str,
    field: str,
    value,
    message: str,
) -> None:
    config = load_configurations("configs/protocol_v2_pilot.yaml")
    config["ccfl_variants"][variant][field] = value
    with pytest.raises(ValueError, match=message):
        validate_protocol_config(config)
