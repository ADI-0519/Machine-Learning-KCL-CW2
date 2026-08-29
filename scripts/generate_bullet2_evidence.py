"""Generate the auditable evidence contract for the TPCRP-CCFL CV bullet."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.aggregate_results import (
    build_round_metrics,
    load_validated_artifacts,
    require_uniform_environment,
)
from src.artifacts import atomic_write_json, config_digest
from src.statistics import pair_method_rows, paired_bootstrap_ci

FULL_METHOD = "tpcrp_ccfl"
BASELINE_METHOD = "tpcrp"
CONFIRMATION_REPLICATES = 10
CONFIRMATION_SEEDS = list(range(42, 52))
PRIMARY_BUDGET = 10
SIMCLR_CONFIG_SHA256 = "5c0ec92e5bb01ef6c000ff397e11e5510c7d2bc22d8f4b348ddde9514e79d2e8"
DINOV2_CONFIG_SHA256 = "37d8dbc22da725d5c414cc56fad579287b8a63e73fc7ff69ac509dc51ce3a70e"


def _comparison(
    rows: pd.DataFrame,
    *,
    method_a: str,
    method_b: str,
    budget: int,
    expected_seeds: list[int],
) -> dict[str, Any]:
    paired = pair_method_rows(rows, method_a=method_a, method_b=method_b)
    paired = paired.loc[paired["cumulative_budget"] == budget].copy()
    if paired.empty:
        raise ValueError(f"no {method_a}/{method_b} pairs exist at budget {budget}")
    actual_seeds = sorted(int(seed) for seed in paired["replicate_seed"])
    if actual_seeds != expected_seeds:
        raise ValueError(
            f"{method_a}/{method_b} seeds do not match the locked confirmation set: "
            f"expected {expected_seeds}, found {actual_seeds}"
        )
    differences = paired["test_accuracy_a"].to_numpy(dtype=float) - paired[
        "test_accuracy_b"
    ].to_numpy(dtype=float)
    ci_low, ci_high = paired_bootstrap_ci(differences)
    run_ids = sorted(set(paired["run_id_a"].astype(str)).union(paired["run_id_b"].astype(str)))
    return {
        "method_a": method_a,
        "method_b": method_b,
        "cumulative_budget": budget,
        "replicate_count": len(actual_seeds),
        "replicate_seeds": actual_seeds,
        "mean_accuracy_a": float(paired["test_accuracy_a"].mean()),
        "mean_accuracy_b": float(paired["test_accuracy_b"].mean()),
        "paired_improvement_pp": float(np.mean(differences) * 100.0),
        "paired_ci95_pp": [ci_low * 100.0, ci_high * 100.0],
        "wins": int(np.count_nonzero(differences > 0.0)),
        "ties": int(np.count_nonzero(differences == 0.0)),
        "run_ids": run_ids,
    }


def _root_evidence(
    root: Path,
    *,
    expected_backend: str,
    require_ablations: bool,
) -> dict[str, Any]:
    artifacts, config = load_validated_artifacts(root)
    source_digest = config_digest(config)
    expected_source_digest = (
        SIMCLR_CONFIG_SHA256 if expected_backend == "simclr" else DINOV2_CONFIG_SHA256
    )
    if source_digest != expected_source_digest:
        raise ValueError(
            f"{expected_backend} artifacts do not use the frozen confirmation configuration: "
            f"expected {expected_source_digest}, found {source_digest}"
        )
    environment = require_uniform_environment(artifacts, context=f"{expected_backend} confirmation")
    checkpoint_digest = str(environment["checkpoint_sha256"])
    representation = config["representation"]
    if representation["backend"] != expected_backend:
        raise ValueError(
            f"expected {expected_backend!r} representation under {root}, "
            f"found {representation['backend']!r}"
        )
    if checkpoint_digest != representation["weights_sha256"]:
        raise ValueError(
            f"{expected_backend} artifact checkpoint digest disagrees with the locked config"
        )
    if config["data"]["name"] != "cifar10":
        raise ValueError("bullet-2 evidence is locked to CIFAR-10")
    comparison_config = config["experiment"]["primary_comparison"]
    if (
        comparison_config["method_a"] != FULL_METHOD
        or comparison_config["method_b"] != BASELINE_METHOD
        or comparison_config["metric"] != "test_accuracy"
    ):
        raise ValueError("source config does not contain the locked TPCRP-CCFL comparison")
    expected_seeds = sorted(int(seed) for seed in config["experiment"]["replicate_seeds"])
    if expected_seeds != CONFIRMATION_SEEDS:
        raise ValueError(
            f"bullet-2 confirmation requires frozen seeds {CONFIRMATION_SEEDS}; "
            f"found {expected_seeds}"
        )

    rows = build_round_metrics(artifacts)
    budget = int(comparison_config["cumulative_budget"])
    if budget != PRIMARY_BUDGET:
        raise ValueError(f"bullet-2 evidence is locked to cumulative budget {PRIMARY_BUDGET}")
    primary = _comparison(
        rows,
        method_a=FULL_METHOD,
        method_b=BASELINE_METHOD,
        budget=budget,
        expected_seeds=expected_seeds,
    )
    ablations = {}
    if require_ablations:
        ablations = {
            "facility_refinement": _comparison(
                rows,
                method_a="ccfl_unweighted",
                method_b="ccfl_candidate_only",
                budget=budget,
                expected_seeds=expected_seeds,
            ),
            "cluster_weighting": _comparison(
                rows,
                method_a=FULL_METHOD,
                method_b="ccfl_unweighted",
                budget=budget,
                expected_seeds=expected_seeds,
            ),
        }

    primary_pairs = pair_method_rows(rows, method_a=FULL_METHOD, method_b=BASELINE_METHOD)
    primary_pairs = primary_pairs.loc[primary_pairs["cumulative_budget"] == budget]
    baseline_distance = float(primary_pairs["nearest_distance_mean_b"].mean())
    baseline_runtime = float(primary_pairs["selector_seconds_b"].mean())
    if baseline_distance <= 0.0 or baseline_runtime <= 0.0:
        raise ValueError("coverage and runtime baselines must be positive")
    full_distance = float(primary_pairs["nearest_distance_mean_a"].mean())
    full_runtime = float(primary_pairs["selector_seconds_a"].mean())

    representation_evidence: dict[str, Any] = {
        "backend": expected_backend,
        "checkpoint_path": representation["checkpoint_path"],
        "checkpoint_sha256": checkpoint_digest,
    }
    if expected_backend == "dinov2":
        representation_evidence.update(
            {
                "model_id": representation["model_id"],
                "model_revision": representation["model_revision"],
                "weights_sha256": representation["weights_sha256"],
            }
        )
    return {
        "artifact_root": str(root),
        "source_config_sha256": source_digest,
        "git_commit": environment["git_commit"],
        "representation": representation_evidence,
        "primary": primary,
        "ablations": ablations,
        "coverage_improvement_percent": (
            (baseline_distance - full_distance) / baseline_distance * 100.0
        ),
        "selector_runtime_ratio": full_runtime / baseline_runtime,
    }


def generate_bullet2_evidence(
    *,
    simclr_root: Path,
    dinov2_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate both confirmation grids and write a no-manual-arithmetic summary."""
    simclr = _root_evidence(
        simclr_root,
        expected_backend="simclr",
        require_ablations=True,
    )
    dinov2 = _root_evidence(
        dinov2_root,
        expected_backend="dinov2",
        require_ablations=False,
    )
    if simclr["git_commit"] != dinov2["git_commit"]:
        raise ValueError("SimCLR and DINOv2 confirmations must use the same clean Git commit")
    checks = {
        "simclr_ten_paired_seeds": (
            simclr["primary"]["replicate_count"] == CONFIRMATION_REPLICATES
        ),
        "simclr_gain_at_least_2pp": simclr["primary"]["paired_improvement_pp"] >= 2.0,
        "simclr_ci95_excludes_zero": simclr["primary"]["paired_ci95_pp"][0] > 0.0,
        "simclr_coverage_improves": simclr["coverage_improvement_percent"] > 0.0,
        "simclr_runtime_at_most_2x": simclr["selector_runtime_ratio"] <= 2.0,
        "dinov2_ten_paired_seeds": (
            dinov2["primary"]["replicate_count"] == CONFIRMATION_REPLICATES
        ),
        "dinov2_mean_gain_positive": dinov2["primary"]["paired_improvement_pp"] > 0.0,
        "simclr_ablation_grid_complete": all(
            result["replicate_count"] == CONFIRMATION_REPLICATES
            for result in simclr["ablations"].values()
        ),
    }
    evidence = {
        "protocol_version": "2.0",
        "metric_definition": (
            "100 * mean_seed(test_accuracy_tpcrp_ccfl - test_accuracy_tpcrp) "
            "at CIFAR-10 cumulative budget 10 under the SimCLR confirmation config"
        ),
        "cv_metric_x_pp": simclr["primary"]["paired_improvement_pp"],
        "cv_metric_x_pp_display": f"{simclr['primary']['paired_improvement_pp']:.1f}",
        "simclr_confirmation": simclr,
        "dinov2_representation_validation": dinov2,
        "claim_checks": checks,
        "claim_ready": all(checks.values()),
    }
    atomic_write_json(output_path, evidence)
    return evidence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simclr-root",
        type=Path,
        default=Path("results/protocol_v2_confirmation_simclr"),
    )
    parser.add_argument(
        "--dinov2-root",
        type=Path,
        default=Path("results/protocol_v2_confirmation_dinov2"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/bullet2_evidence.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    evidence = generate_bullet2_evidence(
        simclr_root=args.simclr_root,
        dinov2_root=args.dinov2_root,
        output_path=args.output,
    )
    status = "READY" if evidence["claim_ready"] else "NOT READY"
    print(f"Bullet 2 evidence: {status}; X={evidence['cv_metric_x_pp_display']}pp")
    print(f"Wrote {args.output}")
    if not evidence["claim_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
