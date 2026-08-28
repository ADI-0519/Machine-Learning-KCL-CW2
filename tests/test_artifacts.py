from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

import src.artifacts as artifacts
from src.artifacts import (
    artifact_path_for,
    atomic_write_json,
    build_effective_config,
    build_run_artifact,
    build_run_id,
    canonical_json,
    collect_environment,
    config_digest,
    file_sha256,
    read_artifact,
    validate_artifact,
)
from src.protocol import PROTOCOL_VERSION, SeedBundle


def _effective_config() -> dict:
    return {
        "protocol": {
            "version": "2.0",
            "deterministic": True,
            "test_evaluations_per_round": 1,
        },
        "output": {"root": "unused", "resume": True},
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
            "max_clusters": 2,
            "min_cluster_size": 2,
        },
        "evaluation": {
            "framework": "fully_supervised",
            "epochs": 2,
            "batch_size": 2,
            "lr": 0.1,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "dropout_p": 0.0,
            "test_policy": "once_after_fixed_epochs",
        },
        "experiment": {
            "methods": ["random", "tpcrp", "ccfl_unweighted", "probcover"],
            "replicate_seeds": [42],
            "primary_comparison": {
                "method_a": "tpcrp",
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
        "run": {
            "framework": "fully_supervised",
            "method": "random",
            "replicate_seed": 42,
        },
    }


def _valid_artifact() -> dict:
    effective = _effective_config()
    digest = config_digest(effective)
    seeds = SeedBundle.for_round(
        replicate=42,
        dataset="cifar10",
        framework="fully_supervised",
        method="random",
        round_id=1,
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": build_run_id(effective),
        "config_digest": digest,
        "effective_config": effective,
        "environment": {
            "git_commit": "a" * 40,
            "git_dirty": False,
            "python_version": "3.11.5",
            "platform": "Windows-test",
            "packages": {"numpy": "2.2.6", "torch": "2.8.0+cpu"},
            "device": "cpu",
            "cuda_runtime": None,
            "deterministic_algorithms": True,
            "cublas_workspace_config": None,
            "checkpoint_path": None,
            "checkpoint_sha256": None,
        },
        "rounds": [
            {
                "round": 1,
                "query_size": 2,
                "cumulative_budget": 2,
                "new_indices": [3, 1],
                "selected_indices": [3, 1],
                "seeds": {
                    "replicate": seeds.replicate,
                    "clustering": seeds.clustering,
                    "selector": seeds.selector,
                    "training": seeds.training,
                    "dataloader": seeds.dataloader,
                },
                "trained_epochs": 2,
                "test_loss": 0.7,
                "test_accuracy": 0.5,
                "selection_diagnostics": {
                    "nearest_distance_mean": 1.0,
                    "nearest_distance_p95": 2.0,
                    "nearest_distance_max": 3.0,
                    "selected_pairwise_cosine_mean": 0.25,
                    "selected_typicality_mean": 4.0,
                },
                "method_metadata": {},
            }
        ],
        "timings": {
            "total_seconds": 1.5,
            "round_seconds": [1.0],
            "selector_seconds": [0.25],
        },
    }


def _refresh_identity(payload: dict) -> None:
    payload["config_digest"] = config_digest(payload["effective_config"])
    payload["run_id"] = build_run_id(payload["effective_config"])


def _set_run_method(payload: dict, method: str) -> None:
    payload["effective_config"]["run"]["method"] = method
    run = payload["effective_config"]["run"]
    for round_result in payload["rounds"]:
        seeds = SeedBundle.for_round(
            replicate=run["replicate_seed"],
            dataset=payload["effective_config"]["data"]["name"],
            framework=run["framework"],
            method=method,
            round_id=round_result["round"],
        )
        round_result["seeds"] = {
            "replicate": seeds.replicate,
            "clustering": seeds.clustering,
            "selector": seeds.selector,
            "training": seeds.training,
            "dataloader": seeds.dataloader,
        }


def test_canonical_json_digest_and_run_id_have_fixed_outputs() -> None:
    value = {"b": 2, "a": 1}
    assert canonical_json(value) == '{"a":1,"b":2}'
    assert (
        config_digest(value) == "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777"
    )
    assert build_run_id(value) == "v2.0-43258cff783fe703"


def test_effective_config_rejects_reserved_run_section() -> None:
    with pytest.raises(ValueError, match="reserved run section"):
        build_effective_config(
            {"run": {}},
            method="random",
            replicate_seed=42,
            framework="fully_supervised",
        )


def test_build_and_read_artifact_helpers_round_trip(tmp_path) -> None:
    source = {"protocol": {"version": "2.0"}}
    effective = build_effective_config(
        source,
        method="random",
        replicate_seed=42,
        framework="fully_supervised",
    )
    assert effective["run"] == {
        "framework": "fully_supervised",
        "method": "random",
        "replicate_seed": 42,
    }
    assert "run" not in source

    valid = _valid_artifact()
    built = build_run_artifact(
        effective_config=valid["effective_config"],
        environment=valid["environment"],
        rounds=valid["rounds"],
        timings=valid["timings"],
    )
    path = artifact_path_for(tmp_path, built["run_id"])
    atomic_write_json(path, built)
    assert read_artifact(path) == built


def test_validate_artifact_accepts_complete_payload() -> None:
    validate_artifact(_valid_artifact())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("environment"), "missing fields"),
        (
            lambda payload: payload["rounds"][0].__setitem__("test_accuracy", float("nan")),
            "non-finite test_accuracy",
        ),
        (
            lambda payload: payload["rounds"][0].__setitem__("best_" + "test_accuracy", 0.6),
            "legacy",
        ),
        (
            lambda payload: payload.__setitem__("protocol_version", "1.0"),
            "protocol version mismatch",
        ),
        (
            lambda payload: payload.__setitem__("config_digest", "0" * 64),
            "config digest mismatch",
        ),
        (
            lambda payload: payload.__setitem__("run_id", "v2.0-wrong"),
            "run ID mismatch",
        ),
        (lambda payload: payload.__setitem__("rounds", []), "no rounds"),
    ],
)
def test_validate_artifact_rejects_invalid_payload(mutation, message: str) -> None:
    payload = copy.deepcopy(_valid_artifact())
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        validate_artifact(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.__setitem__("unexpected", True), "unexpected fields"),
        (lambda payload: payload.__setitem__("effective_config", []), "effective_config"),
        (lambda payload: payload.__setitem__("environment", []), "environment"),
        (
            lambda payload: payload["environment"].__setitem__("git_commit", ""),
            "git_commit",
        ),
        (
            lambda payload: payload["environment"].__setitem__("git_dirty", "no"),
            "git_dirty",
        ),
        (
            lambda payload: payload["environment"].__setitem__("packages", {}),
            "packages",
        ),
        (
            lambda payload: payload["environment"].__setitem__("platform", ""),
            "platform",
        ),
        (
            lambda payload: payload["environment"].__setitem__("deterministic_algorithms", "yes"),
            "deterministic_algorithms",
        ),
        (
            lambda payload: payload["environment"].__setitem__("cublas_workspace_config", 4096),
            "cublas_workspace_config",
        ),
        (
            lambda payload: payload["environment"].__setitem__("checkpoint_sha256", "invalid"),
            "checkpoint_sha256",
        ),
        (
            lambda payload: payload["rounds"][0].__setitem__("round", 2),
            "round sequence",
        ),
        (
            lambda payload: payload["rounds"][0].__setitem__("query_size", 0),
            "query_size",
        ),
        (
            lambda payload: payload["rounds"][0].__setitem__("new_indices", [3, 3]),
            "new_indices",
        ),
        (
            lambda payload: payload["rounds"][0].__setitem__("selected_indices", [1, 3]),
            "nestedness",
        ),
        (
            lambda payload: payload["rounds"][0].__setitem__("cumulative_budget", 1),
            "cumulative_budget",
        ),
        (lambda payload: payload.__setitem__("rounds", [None]), "must be an object"),
        (
            lambda payload: payload["rounds"][0].__setitem__("seeds", []),
            "seeds must be an object",
        ),
        (
            lambda payload: payload["rounds"][0]["seeds"].__setitem__("replicate", 7),
            "replicate seed mismatch",
        ),
        (
            lambda payload: payload["rounds"][0].__setitem__("trained_epochs", 0),
            "trained_epochs",
        ),
        (
            lambda payload: payload["rounds"][0].__setitem__("test_accuracy", 1.1),
            "exceeds one",
        ),
        (
            lambda payload: payload["rounds"][0].__setitem__("selection_diagnostics", []),
            "diagnostics must be an object",
        ),
        (
            lambda payload: payload["rounds"][0]["selection_diagnostics"].__setitem__(
                "nearest_distance_mean", 4.0
            ),
            "unordered",
        ),
        (
            lambda payload: payload["rounds"][0]["selection_diagnostics"].__setitem__(
                "selected_pairwise_cosine_mean", 1.1
            ),
            "pairwise cosine",
        ),
        (
            lambda payload: payload["rounds"][0]["selection_diagnostics"].__setitem__(
                "selected_typicality_mean", float("inf")
            ),
            "typicality",
        ),
        (
            lambda payload: payload["rounds"][0].__setitem__("method_metadata", []),
            "method_metadata must be an object",
        ),
        (
            lambda payload: payload["rounds"][0].__setitem__(
                "method_metadata", {"unexpected": True}
            ),
            "unexpected method_metadata",
        ),
        (
            lambda payload: payload["timings"].__setitem__("round_seconds", []),
            "round_seconds",
        ),
        (
            lambda payload: payload["timings"].__setitem__("selector_seconds", []),
            "selector_seconds",
        ),
        (
            lambda payload: payload["timings"].__setitem__("total_seconds", -1.0),
            "negative total_seconds",
        ),
    ],
)
def test_validate_artifact_rejects_schema_invariants(mutation, message: str) -> None:
    payload = copy.deepcopy(_valid_artifact())
    mutation(payload)
    if isinstance(payload.get("effective_config"), dict):
        _refresh_identity(payload)
    with pytest.raises(ValueError, match=message):
        validate_artifact(payload)


def test_atomic_write_leaves_no_temporary_file(tmp_path) -> None:
    target = tmp_path / "run.json"
    payload = _valid_artifact()
    atomic_write_json(target, payload)

    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert list(tmp_path.glob(".*.tmp")) == []


def test_read_artifact_rejects_non_object_root(tmp_path) -> None:
    path = tmp_path / "not-an-artifact.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be an object"):
        read_artifact(path)


def test_validate_artifact_rejects_negative_global_index() -> None:
    payload = _valid_artifact()
    payload["rounds"][0]["new_indices"] = [-1, 1]
    payload["rounds"][0]["selected_indices"] = [-1, 1]
    with pytest.raises(ValueError, match="integer >= 0"):
        validate_artifact(payload)


def test_validate_artifact_rejects_schedule_disagreement() -> None:
    payload = _valid_artifact()
    payload["effective_config"]["selection"]["round_query_sizes"] = [1]
    payload["effective_config"]["experiment"]["primary_comparison"]["cumulative_budget"] = 1
    _refresh_identity(payload)

    with pytest.raises(ValueError, match="query_size disagrees with configured schedule"):
        validate_artifact(payload)


def test_validate_artifact_rejects_training_epoch_disagreement() -> None:
    payload = _valid_artifact()
    payload["effective_config"]["evaluation"]["epochs"] = 3
    _refresh_identity(payload)

    with pytest.raises(ValueError, match="trained_epochs disagrees with evaluation config"):
        validate_artifact(payload)


def test_validate_artifact_rejects_rederived_seed_disagreement() -> None:
    payload = _valid_artifact()
    payload["rounds"][0]["seeds"]["selector"] += 1

    with pytest.raises(ValueError, match="component seeds are inconsistent"):
        validate_artifact(payload)


def test_validate_artifact_rejects_index_outside_cifar_train_split() -> None:
    payload = _valid_artifact()
    payload["rounds"][0]["new_indices"] = [50_000, 1]
    payload["rounds"][0]["selected_indices"] = [50_000, 1]

    with pytest.raises(ValueError, match="outside CIFAR-10 train data"):
        validate_artifact(payload)


def test_validate_artifact_accepts_ccfl_and_probcover_metadata() -> None:
    ccfl_payload = _valid_artifact()
    ccfl_config = {
        "candidates_per_cluster": 5,
        "refine_steps": 1,
        "use_cluster_weights": False,
    }
    _set_run_method(ccfl_payload, "ccfl_unweighted")
    ccfl_payload["effective_config"]["ccfl_variants"]["ccfl_unweighted"] = ccfl_config
    ccfl_payload["rounds"][0]["method_metadata"] = {
        "ccfl_variant": "ccfl_unweighted",
        **ccfl_config,
    }
    _refresh_identity(ccfl_payload)
    validate_artifact(ccfl_payload)

    probcover_payload = _valid_artifact()
    _set_run_method(probcover_payload, "probcover")
    probcover_payload["rounds"][0]["method_metadata"] = {
        "probcover_delta": 0.25,
        "probcover_radius_seed": 17,
        "probcover_cache_digest": "b" * 64,
    }
    _refresh_identity(probcover_payload)
    validate_artifact(probcover_payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("probcover_delta", 0.0, "ProbCover delta"),
        ("probcover_radius_seed", -1, "radius seed"),
        ("probcover_cache_digest", "invalid", "cache digest"),
    ],
)
def test_validate_artifact_rejects_invalid_probcover_metadata(
    field: str,
    value,
    message: str,
) -> None:
    payload = _valid_artifact()
    _set_run_method(payload, "probcover")
    payload["rounds"][0]["method_metadata"] = {
        "probcover_delta": 0.25,
        "probcover_radius_seed": 17,
        "probcover_cache_digest": "b" * 64,
    }
    payload["rounds"][0]["method_metadata"][field] = value
    _refresh_identity(payload)
    with pytest.raises(ValueError, match=message):
        validate_artifact(payload)


def test_atomic_write_preserves_previous_file_when_replace_fails(monkeypatch, tmp_path) -> None:
    target = tmp_path / "run.json"
    previous = {"complete": True}
    target.write_text(json.dumps(previous), encoding="utf-8")

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        atomic_write_json(target, _valid_artifact())

    assert json.loads(target.read_text(encoding="utf-8")) == previous
    assert list(tmp_path.glob(".*.tmp")) == []


def test_file_sha256_has_fixed_output(tmp_path) -> None:
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"abc")
    assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_collect_environment_records_checkpoint_and_cuda(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "representation.pt"
    checkpoint.write_bytes(b"checkpoint")
    git_outputs = {
        ("rev-parse", "HEAD"): "d" * 40,
        ("status", "--porcelain"): " M src/example.py",
    }
    monkeypatch.setattr(artifacts, "_git_output", lambda *args: git_outputs[args])
    monkeypatch.setattr(artifacts.metadata, "version", lambda package: f"{package}-version")
    monkeypatch.setattr(artifacts.torch.cuda, "get_device_name", lambda _device: "Test GPU")

    environment = collect_environment(
        device=torch.device("cuda"),
        checkpoint_path=checkpoint,
    )

    assert environment["git_commit"] == "d" * 40
    assert environment["git_dirty"] is True
    assert environment["device"] == "cuda:Test GPU"
    assert isinstance(environment["platform"], str)
    assert isinstance(environment["deterministic_algorithms"], bool)
    assert environment["checkpoint_path"] == str(checkpoint.resolve())
    assert environment["checkpoint_sha256"] == file_sha256(checkpoint)
    assert environment["packages"]["torch"] == torch.__version__


def test_collect_environment_rejects_missing_checkpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(artifacts, "_git_output", lambda *_args: "clean")
    monkeypatch.setattr(artifacts.metadata, "version", lambda package: f"{package}-version")
    with pytest.raises(FileNotFoundError, match="checkpoint is missing"):
        collect_environment(
            device=torch.device("cpu"),
            checkpoint_path=tmp_path / "missing.pt",
        )
