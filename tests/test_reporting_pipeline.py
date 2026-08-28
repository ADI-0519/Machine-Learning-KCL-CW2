from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.aggregate_results import build_round_metrics, write_protocol_reports
from scripts.check_pilot_gate import evaluate_pilot_gate
from scripts.generate_bullet2_evidence import generate_bullet2_evidence
from scripts.generate_cv_evidence import generate_cv_evidence
from scripts.make_plots import write_protocol_plots
from scripts.run_stats import write_paired_statistics
from src.artifacts import (
    atomic_write_json,
    build_effective_config,
    build_run_artifact,
    read_artifact,
)
from src.protocol import SeedBundle


def _protocol_config(root: Path, *, representation_backend: str = "simclr") -> dict:
    return {
        "protocol": {
            "version": "2.0",
            "deterministic": True,
            "test_evaluations_per_round": 1,
        },
        "output": {"root": str(root), "resume": True},
        "data": {
            "name": "cifar10",
            "root": "unused",
            "num_workers": 0,
            "num_classes": 10,
        },
        "representation": {
            "backend": "simclr",
            "checkpoint_path": "synthetic.pt",
            "projection_dim": 4,
            "batch_size": 8,
            "embedding_seed": 21,
        },
        "selection": {
            "round_query_sizes": [10, 10],
            "knn_k": 2,
            "max_clusters": 20,
            "min_cluster_size": 2,
        },
        "evaluation": {
            "framework": "ssl_embedding",
            "epochs": 60,
            "batch_size": 4,
            "lr": 2.5,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "dropout_p": 0.2,
            "test_policy": "once_after_fixed_epochs",
        },
        "experiment": {
            "methods": ["tpcrp", "tpcrp_ccfl"],
            "replicate_seeds": [1, 2, 3, 4, 5],
            "primary_comparison": {
                "method_a": "tpcrp_ccfl",
                "method_b": "tpcrp",
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


def _set_representation_backend(config: dict, backend: str) -> None:
    if backend == "simclr":
        return
    if backend != "dinov2":
        raise ValueError(f"unsupported synthetic representation: {backend}")
    config["representation"] = {
        "backend": "dinov2",
        "checkpoint_path": "model.safetensors",
        "model_id": "facebook/dinov2-small",
        "model_revision": "e" * 40,
        "weights_sha256": "b" * 64,
        "feature_dim": 384,
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 32,
        "embedding_seed": 21,
    }


def _write_synthetic_grid(
    root: Path,
    *,
    configured_methods: list[str] | None = None,
    written_methods: list[str] | None = None,
    configured_seeds: list[int] | None = None,
    representation_backend: str = "simclr",
) -> list[Path]:
    config = _protocol_config(root, representation_backend=representation_backend)
    _set_representation_backend(config, representation_backend)
    if configured_methods is not None:
        config["experiment"]["methods"] = configured_methods
    if configured_seeds is not None:
        config["experiment"]["replicate_seeds"] = configured_seeds
    methods_to_write = written_methods or config["experiment"]["methods"]
    replicate_count = len(config["experiment"]["replicate_seeds"])
    baseline_accuracy = 0.50 + 0.02 * np.arange(replicate_count)
    improvements = 0.02 + 0.01 * np.arange(replicate_count)
    baseline_distance = np.resize(
        np.array([2.0, 2.2, 1.8, 2.1, 1.9]),
        replicate_count,
    )
    written: list[Path] = []

    for method in methods_to_write:
        for seed_index, seed in enumerate(config["experiment"]["replicate_seeds"]):
            effective = build_effective_config(
                config,
                method=method,
                replicate_seed=seed,
                framework="ssl_embedding",
            )
            rounds = []
            selected: list[int] = []
            selector_seconds = []
            for round_id, budget in enumerate((10, 20), start=1):
                new_indices = list(range((round_id - 1) * 10, round_id * 10))
                selected.extend(new_indices)
                seeds = SeedBundle.for_round(
                    replicate=seed,
                    dataset="cifar10",
                    framework="ssl_embedding",
                    method=method,
                    round_id=round_id,
                )
                base_accuracy = baseline_accuracy[seed_index] + 0.02 * (round_id - 1)
                method_gain = {
                    "tpcrp_ccfl": improvements[seed_index],
                    "ccfl_candidate_only": improvements[seed_index] * 0.4,
                    "ccfl_unweighted": improvements[seed_index] * 0.7,
                }.get(method, 0.0)
                accuracy = base_accuracy + method_gain
                distance = baseline_distance[seed_index]
                if method == "tpcrp_ccfl":
                    distance -= 0.2
                duration = 1.5 if method == "tpcrp_ccfl" else 1.0
                selector_seconds.append(duration)
                rounds.append(
                    {
                        "round": round_id,
                        "query_size": 10,
                        "cumulative_budget": budget,
                        "new_indices": new_indices,
                        "selected_indices": selected.copy(),
                        "seeds": {
                            "replicate": seeds.replicate,
                            "clustering": seeds.clustering,
                            "selector": seeds.selector,
                            "training": seeds.training,
                            "dataloader": seeds.dataloader,
                        },
                        "trained_epochs": 60,
                        "test_loss": 1.0 - accuracy,
                        "test_accuracy": accuracy,
                        "selection_diagnostics": {
                            "nearest_distance_mean": distance,
                            "nearest_distance_p95": distance + 0.5,
                            "nearest_distance_max": distance + 1.0,
                            "selected_pairwise_cosine_mean": 0.1,
                            "selected_typicality_mean": 2.0,
                        },
                        "method_metadata": (
                            {
                                "ccfl_variant": method,
                                **config["ccfl_variants"][method],
                            }
                            if method in config["ccfl_variants"]
                            else {}
                        ),
                    }
                )

            artifact = build_run_artifact(
                effective_config=effective,
                environment={
                    "git_commit": "a" * 40,
                    "git_dirty": False,
                    "python_version": "3.11.5",
                    "platform": "synthetic-platform",
                    "packages": {"torch": "2.8.0+cpu"},
                    "device": "cpu",
                    "cuda_runtime": None,
                    "deterministic_algorithms": True,
                    "cublas_workspace_config": None,
                    "checkpoint_path": "synthetic.pt",
                    "checkpoint_sha256": "b" * 64,
                },
                rounds=rounds,
                timings={
                    "total_seconds": 5.0,
                    "round_seconds": [2.0, 3.0],
                    "selector_seconds": selector_seconds,
                },
            )
            path = root / "runs" / f"{artifact['run_id']}.json"
            atomic_write_json(path, artifact)
            written.append(path)
    return written


def test_aggregation_writes_deterministic_round_and_summary_tables(tmp_path) -> None:
    root = tmp_path / "protocol_v2"
    _write_synthetic_grid(root)

    outputs = write_protocol_reports(root)
    round_path = outputs["round_metrics"]
    summary_path = outputs["summary_metrics"]
    first_round_bytes = round_path.read_bytes()
    first_summary_bytes = summary_path.read_bytes()
    write_protocol_reports(root)

    assert round_path.read_bytes() == first_round_bytes
    assert summary_path.read_bytes() == first_summary_bytes
    rounds = pd.read_csv(round_path)
    summary = pd.read_csv(summary_path)
    assert len(rounds) == 20
    assert rounds[["dataset", "framework"]].drop_duplicates().to_dict("records") == [
        {"dataset": "cifar10", "framework": "ssl_embedding"}
    ]
    assert set(rounds["method"]) == {"tpcrp", "tpcrp_ccfl"}
    assert len(summary) == 4
    assert set(
        [
            "accuracy_mean",
            "accuracy_std",
            "replicate_count",
            "nearest_distance_mean",
            "selector_seconds_mean",
        ]
    ).issubset(summary.columns)


def test_aggregation_rejects_duplicate_logical_rounds(tmp_path) -> None:
    root = tmp_path / "protocol_v2"
    paths = _write_synthetic_grid(root)
    artifact = read_artifact(paths[0])

    with pytest.raises(ValueError, match="Duplicate protocol result"):
        build_round_metrics([artifact, artifact])


def test_aggregation_rejects_run_dimensions_outside_source_config(tmp_path) -> None:
    root = tmp_path / "protocol_v2"
    paths = _write_synthetic_grid(root)
    artifact = read_artifact(paths[0])
    effective_config = artifact["effective_config"]
    effective_config["run"]["method"] = "random"
    invalid = build_run_artifact(
        effective_config=effective_config,
        environment=artifact["environment"],
        rounds=artifact["rounds"],
        timings=artifact["timings"],
    )
    atomic_write_json(root / "runs" / f"{invalid['run_id']}.json", invalid)

    with pytest.raises(ValueError, match="uses unconfigured method 'random'"):
        write_protocol_reports(root)


def test_statistics_use_paired_replicates_and_holm_adjustment(tmp_path) -> None:
    root = tmp_path / "protocol_v2"
    _write_synthetic_grid(root)

    stats = write_paired_statistics(root)

    assert stats["cumulative_budget"].tolist() == [10, 20]
    assert stats["pair_count"].tolist() == [5, 5]
    np.testing.assert_allclose(stats["paired_mean_difference"], [0.04, 0.04])
    np.testing.assert_allclose(stats["paired_ci95_low"], [0.028, 0.028])
    np.testing.assert_allclose(stats["paired_ci95_high"], [0.052, 0.052])
    assert (stats["holm_adjusted_p_value"] >= stats["raw_p_value"]).all()
    persisted = pd.read_csv(root / "reports" / "paired_statistics.csv")
    pd.testing.assert_frame_equal(persisted, stats, check_exact=False, atol=1e-14)


def test_statistics_reject_missing_primary_partner(tmp_path) -> None:
    root = tmp_path / "protocol_v2"
    paths = _write_synthetic_grid(root)
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        run = payload["effective_config"]["run"]
        if run["method"] == "tpcrp" and run["replicate_seed"] == 5:
            path.unlink()
            break

    with pytest.raises(ValueError, match="missing method partner"):
        write_paired_statistics(root)


def test_cv_evidence_is_computed_from_primary_artifact_pairs(tmp_path) -> None:
    root = tmp_path / "protocol_v2"
    _write_synthetic_grid(root)

    evidence = generate_cv_evidence(root)

    assert evidence["protocol_version"] == "2.0"
    assert evidence["total_runs"] == 10
    assert evidence["total_model_fits"] == 20
    assert evidence["datasets"] == ["cifar10"]
    assert evidence["methods"] == ["tpcrp", "tpcrp_ccfl"]
    assert evidence["replicate_count"] == 5
    assert evidence["primary_budget"] == 10
    assert evidence["ccfl_mean_accuracy"] == pytest.approx(0.58)
    assert evidence["tpcrp_mean_accuracy"] == pytest.approx(0.54)
    assert evidence["paired_improvement_pp"] == pytest.approx(4.0)
    assert evidence["paired_ci95_pp"] == pytest.approx([2.8, 5.2])
    assert evidence["coverage_improvement_percent"] == pytest.approx(10.0)
    assert evidence["selector_runtime_ratio"] == pytest.approx(1.5)
    assert len(evidence["run_ids"]) == 10
    assert set(evidence) == {
        "protocol_version",
        "total_runs",
        "total_model_fits",
        "datasets",
        "methods",
        "replicate_count",
        "primary_budget",
        "ccfl_mean_accuracy",
        "tpcrp_mean_accuracy",
        "paired_improvement_pp",
        "paired_ci95_pp",
        "coverage_improvement_percent",
        "selector_runtime_ratio",
        "run_ids",
    }
    persisted = json.loads((root / "reports" / "cv_evidence.json").read_text(encoding="utf-8"))
    assert persisted == evidence


def test_cv_evidence_supports_primary_first_gate_before_larger_grid(tmp_path) -> None:
    root = tmp_path / "protocol_v2"
    _write_synthetic_grid(
        root,
        configured_methods=["random", "tpcrp", "tpcrp_ccfl"],
        written_methods=["tpcrp", "tpcrp_ccfl"],
    )

    evidence = generate_cv_evidence(root)

    assert evidence["total_runs"] == 10
    assert evidence["methods"] == ["tpcrp", "tpcrp_ccfl"]
    assert len(evidence["run_ids"]) == 10


def test_pilot_gate_is_computed_from_five_primary_pairs(tmp_path) -> None:
    root = tmp_path / "protocol_v2"
    _write_synthetic_grid(
        root,
        configured_methods=["random", "tpcrp", "tpcrp_ccfl"],
        written_methods=["tpcrp", "tpcrp_ccfl"],
    )

    decision = evaluate_pilot_gate(root)

    assert decision["pass"] is True
    assert decision["wins"] == 5
    assert decision["paired_improvement_pp"] == pytest.approx(4.0)
    assert all(decision["checks"].values())
    persisted = json.loads((root / "reports" / "pilot_gate.json").read_text("utf-8"))
    assert persisted == decision


def test_cv_evidence_rejects_incomplete_configured_seed_set(tmp_path) -> None:
    root = tmp_path / "protocol_v2"
    paths = _write_synthetic_grid(root)
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["effective_config"]["run"]["replicate_seed"] == 5:
            path.unlink()

    with pytest.raises(ValueError, match="replicate seeds do not match"):
        generate_cv_evidence(root)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("distance", "zero method-B distance"),
        ("runtime", "zero method-B runtime"),
    ],
)
def test_cv_evidence_rejects_zero_denominators(tmp_path, field, message) -> None:
    root = tmp_path / "protocol_v2"
    paths = _write_synthetic_grid(root)
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["effective_config"]["run"]["method"] != "tpcrp":
            continue
        if field == "distance":
            for round_result in payload["rounds"]:
                round_result["selection_diagnostics"]["nearest_distance_mean"] = 0.0
        else:
            payload["timings"]["selector_seconds"] = [0.0, 0.0]
        atomic_write_json(path, payload)

    with pytest.raises(ValueError, match=message):
        generate_cv_evidence(root)


def test_bullet2_evidence_requires_complete_simclr_dinov2_and_ablations(tmp_path) -> None:
    methods = ["tpcrp", "tpcrp_ccfl", "ccfl_candidate_only", "ccfl_unweighted"]
    seeds = list(range(10))
    simclr_root = tmp_path / "simclr"
    dinov2_root = tmp_path / "dinov2"
    _write_synthetic_grid(
        simclr_root,
        configured_methods=methods,
        configured_seeds=seeds,
    )
    _write_synthetic_grid(
        dinov2_root,
        configured_methods=["tpcrp", "tpcrp_ccfl"],
        configured_seeds=seeds,
        representation_backend="dinov2",
    )
    output_path = tmp_path / "bullet2_evidence.json"

    evidence = generate_bullet2_evidence(
        simclr_root=simclr_root,
        dinov2_root=dinov2_root,
        output_path=output_path,
    )

    assert evidence["claim_ready"] is True
    assert evidence["cv_metric_x_pp"] == pytest.approx(6.5)
    assert evidence["cv_metric_x_pp_display"] == "6.5"
    assert evidence["simclr_confirmation"]["primary"]["replicate_count"] == 10
    assert evidence["dinov2_representation_validation"]["primary"]["replicate_count"] == 10
    assert evidence["dinov2_representation_validation"]["ablations"] == {}
    assert set(evidence["simclr_confirmation"]["ablations"]) == {
        "ccfl_candidate_only",
        "ccfl_unweighted",
    }
    assert json.loads(output_path.read_text(encoding="utf-8")) == evidence


def test_bullet2_evidence_rejects_incomplete_ablation_grid(tmp_path) -> None:
    methods = ["tpcrp", "tpcrp_ccfl", "ccfl_candidate_only", "ccfl_unweighted"]
    seeds = list(range(10))
    simclr_root = tmp_path / "simclr"
    dinov2_root = tmp_path / "dinov2"
    paths = _write_synthetic_grid(
        simclr_root,
        configured_methods=methods,
        configured_seeds=seeds,
    )
    _write_synthetic_grid(
        dinov2_root,
        configured_methods=["tpcrp", "tpcrp_ccfl"],
        configured_seeds=seeds,
        representation_backend="dinov2",
    )
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        run = payload["effective_config"]["run"]
        if run["method"] == "ccfl_unweighted" and run["replicate_seed"] == seeds[-1]:
            path.unlink()
            break

    with pytest.raises(ValueError, match="missing method partner|seeds do not match"):
        generate_bullet2_evidence(
            simclr_root=simclr_root,
            dinov2_root=dinov2_root,
            output_path=tmp_path / "unused.json",
        )


def test_plots_are_separate_by_framework_and_label_standard_deviation(tmp_path) -> None:
    root = tmp_path / "protocol_v2"
    _write_synthetic_grid(root)
    paths = write_protocol_plots(root)

    assert [path.name for path in paths] == ["accuracy_cifar10_ssl_embedding.png"]
    assert paths[0].is_file() and paths[0].stat().st_size > 0
    source = Path("scripts/make_plots.py").read_text(encoding="utf-8")
    assert "±1 SD" in source
