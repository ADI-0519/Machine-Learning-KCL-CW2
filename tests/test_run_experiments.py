from __future__ import annotations

import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

import scripts.run_experiments as runner
from src.artifacts import (
    artifact_path_for,
    atomic_write_json,
    build_effective_config,
    build_run_artifact,
)
from src.config import load_configurations
from src.protocol import SeedBundle


def _protocol_config(output_root: Path) -> dict:
    return {
        "protocol": {
            "version": "2.0",
            "deterministic": True,
            "test_evaluations_per_round": 1,
        },
        "output": {"root": str(output_root), "resume": True},
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
            "round_query_sizes": [2],
            "knn_k": 2,
            "max_clusters": 10,
            "min_cluster_size": 1,
        },
        "evaluation": {
            "framework": "fully_supervised",
            "epochs": 2,
            "batch_size": 4,
            "lr": 0.01,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "dropout_p": 0.0,
            "test_policy": "once_after_fixed_epochs",
        },
        "experiment": {
            "methods": ["random"],
            "replicate_seeds": [41, 42, 43],
            "primary_comparison": {
                "method_a": "random",
                "method_b": "random",
                "cumulative_budget": 2,
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


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "protocol.yaml"
    path.write_text(
        yaml.safe_dump(_protocol_config(tmp_path / "outputs"), sort_keys=False),
        encoding="utf-8",
    )
    return path


def _synthetic_artifact(config: dict, *, method: str, seed: int, framework: str) -> dict:
    effective = build_effective_config(
        config,
        method=method,
        replicate_seed=seed,
        framework=framework,
    )
    seeds = SeedBundle.for_round(
        replicate=seed,
        dataset="cifar10",
        framework=framework,
        method=method,
        round_id=1,
    )
    round_record = {
        "round": 1,
        "query_size": 2,
        "cumulative_budget": 2,
        "new_indices": [0, 1],
        "selected_indices": [0, 1],
        "seeds": {
            "replicate": seeds.replicate,
            "clustering": seeds.clustering,
            "selector": seeds.selector,
            "training": seeds.training,
            "dataloader": seeds.dataloader,
        },
        "trained_epochs": 2,
        "test_loss": 0.5,
        "test_accuracy": 0.5,
        "selection_diagnostics": {
            "nearest_distance_mean": 1.0,
            "nearest_distance_p95": 1.0,
            "nearest_distance_max": 1.0,
            "selected_pairwise_cosine_mean": 0.0,
            "selected_typicality_mean": 1.0,
        },
        "method_metadata": {},
    }
    return build_run_artifact(
        effective_config=effective,
        environment={
            "git_commit": "b" * 40,
            "git_dirty": False,
            "python_version": "3.11.5",
            "platform": "Windows-test",
            "packages": {"torch": "2.8.0+cpu"},
            "device": "cpu",
            "cuda_runtime": None,
            "deterministic_algorithms": True,
            "cublas_workspace_config": None,
            "checkpoint_path": None,
            "checkpoint_sha256": None,
        },
        rounds=[round_record],
        timings={
            "total_seconds": 0.1,
            "round_seconds": [0.1],
            "selector_seconds": [0.01],
        },
    )


def test_dry_run_lists_jobs_without_executing(monkeypatch, tmp_path) -> None:
    config_path = _write_config(tmp_path)

    def forbidden_run(**_kwargs):
        raise AssertionError("dry-run executed an experiment")

    monkeypatch.setattr(runner, "run_single_experiment", forbidden_run)
    records = runner.run_experiment_grid(config_path=config_path, dry_run=True)

    assert len(records) == 3
    assert {record["status"] for record in records} == {"planned"}


def test_resume_executes_only_missing_run(monkeypatch, tmp_path) -> None:
    config_path = _write_config(tmp_path)
    config = load_configurations(config_path)
    framework = config["evaluation"]["framework"]

    for seed in (41, 42):
        artifact = _synthetic_artifact(
            config,
            method="random",
            seed=seed,
            framework=framework,
        )
        path = artifact_path_for(config["output"]["root"], artifact["run_id"])
        atomic_write_json(path, artifact)

    executed: list[int] = []

    def fake_run(*, config_path, method, seed, framework):
        executed.append(seed)
        loaded = load_configurations(config_path)
        artifact = _synthetic_artifact(
            loaded,
            method=method,
            seed=seed,
            framework=framework,
        )
        path = artifact_path_for(loaded["output"]["root"], artifact["run_id"])
        atomic_write_json(path, artifact)
        return deepcopy(artifact)

    monkeypatch.setattr(runner, "run_single_experiment", fake_run)
    records = runner.run_experiment_grid(config_path=config_path)

    assert executed == [43]
    assert [record["status"] for record in records] == ["skipped", "skipped", "completed"]


def test_resume_rejects_malformed_existing_artifact(monkeypatch, tmp_path) -> None:
    config_path = _write_config(tmp_path)
    config = load_configurations(config_path)
    effective = build_effective_config(
        config,
        method="random",
        replicate_seed=41,
        framework="fully_supervised",
    )
    path = artifact_path_for(config["output"]["root"], runner.build_run_id(effective))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")

    def forbidden_run(**_kwargs):
        raise AssertionError("malformed artifact caused execution")

    monkeypatch.setattr(runner, "run_single_experiment", forbidden_run)
    with pytest.raises(ValueError, match="invalid artifact"):
        runner.run_experiment_grid(config_path=config_path)


def test_two_fresh_processes_produce_same_artifact_except_timings(tmp_path) -> None:
    config_path = _write_config(tmp_path)
    config = load_configurations(config_path)
    effective = build_effective_config(
        config,
        method="random",
        replicate_seed=41,
        framework="fully_supervised",
    )
    path = artifact_path_for(config["output"]["root"], runner.build_run_id(effective))
    command = [
        sys.executable,
        "tests/helpers/run_synthetic_protocol.py",
        str(config_path),
    ]

    subprocess.run(command, check=True, capture_output=True, text=True)
    first = runner.read_artifact(path)
    path.unlink()
    subprocess.run(command, check=True, capture_output=True, text=True)
    second = runner.read_artifact(path)

    first.pop("timings")
    second.pop("timings")
    assert second == first
