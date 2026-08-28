"""Build deterministic round-level and summary reports from canonical artifacts."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from src.artifacts import canonical_json, config_digest, read_artifact
from src.config import validate_protocol_config

ROUND_REPORT_COLUMNS = [
    "protocol_version",
    "source_config_digest",
    "run_id",
    "config_digest",
    "dataset",
    "framework",
    "method",
    "replicate_seed",
    "round",
    "query_size",
    "cumulative_budget",
    "trained_epochs",
    "test_loss",
    "test_accuracy",
    "nearest_distance_mean",
    "nearest_distance_p95",
    "nearest_distance_max",
    "selected_pairwise_cosine_mean",
    "selected_typicality_mean",
    "selector_seconds",
    "new_indices_json",
    "selected_indices_json",
    "method_metadata_json",
]
ROUND_SORT_COLUMNS = [
    "dataset",
    "framework",
    "method",
    "replicate_seed",
    "cumulative_budget",
]
SUMMARY_COLUMNS = [
    "dataset",
    "framework",
    "method",
    "cumulative_budget",
    "accuracy_mean",
    "accuracy_std",
    "replicate_count",
    "nearest_distance_mean",
    "nearest_distance_p95_mean",
    "nearest_distance_max_mean",
    "selected_pairwise_cosine_mean",
    "selected_typicality_mean",
    "selector_seconds_mean",
]


def load_validated_artifacts(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load one internally consistent artifact set without tolerating bad runs."""
    artifact_paths = sorted((root / "runs").glob("*.json"))
    if not artifact_paths:
        raise FileNotFoundError(f"No canonical run artifacts found under {root / 'runs'}")

    artifacts: list[dict[str, Any]] = []
    run_ids: set[str] = set()
    source_digests: set[str] = set()
    source_config: dict[str, Any] | None = None

    for path in artifact_paths:
        artifact = read_artifact(path)
        run_id = str(artifact["run_id"])
        if path.stem != run_id:
            raise ValueError(
                f"Artifact filename {path.name!r} does not match embedded run_id {run_id!r}"
            )
        if run_id in run_ids:
            raise ValueError(f"Duplicate run_id across artifacts: {run_id!r}")
        run_ids.add(run_id)

        effective_config = deepcopy(artifact["effective_config"])
        run_config = effective_config.pop("run")
        validate_protocol_config(effective_config)
        if run_config["method"] not in effective_config["experiment"]["methods"]:
            raise ValueError(
                f"Artifact {run_id!r} uses unconfigured method {run_config['method']!r}"
            )
        if run_config["replicate_seed"] not in effective_config["experiment"]["replicate_seeds"]:
            raise ValueError(
                f"Artifact {run_id!r} uses unconfigured seed {run_config['replicate_seed']!r}"
            )
        if run_config["framework"] != effective_config["evaluation"]["framework"]:
            raise ValueError(
                f"Artifact {run_id!r} framework {run_config['framework']!r} disagrees "
                "with the configured evaluation framework"
            )
        expected_source_digest = config_digest(effective_config)
        source_digests.add(expected_source_digest)
        if source_config is None:
            source_config = effective_config
        artifacts.append(artifact)

    if len(source_digests) != 1:
        raise ValueError(
            "Artifacts from different source configurations cannot be aggregated together"
        )
    if source_config is None:  # pragma: no cover - guarded by artifact_paths
        raise RuntimeError("Artifact loading produced no source configuration")
    return artifacts, source_config


def require_uniform_environment(
    artifacts: Sequence[dict[str, Any]],
    *,
    context: str,
) -> dict[str, Any]:
    """Return one clean environment shared exactly by every supplied artifact."""
    if not artifacts:
        raise ValueError(f"{context} contains no artifacts")
    environments = {
        canonical_json(artifact["environment"]): artifact["environment"] for artifact in artifacts
    }
    if len(environments) != 1:
        raise ValueError(f"{context} must use one identical execution environment")
    environment = next(iter(environments.values()))
    if environment["git_dirty"]:
        raise ValueError(f"{context} artifacts must come from a clean Git worktree")
    return environment


def build_round_metrics(artifacts: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Flatten validated artifacts into one deterministic row per model fit."""
    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, int, int]] = set()

    for artifact in artifacts:
        rounds = artifact["rounds"]
        selector_seconds = artifact["timings"]["selector_seconds"]
        if len(rounds) != len(selector_seconds):
            raise ValueError(
                f"Artifact {artifact['run_id']!r} has inconsistent round timing cardinality"
            )

        for round_result, selection_seconds in zip(rounds, selector_seconds, strict=True):
            logical_key = (
                str(artifact["effective_config"]["data"]["name"]),
                str(artifact["effective_config"]["run"]["framework"]),
                str(artifact["effective_config"]["run"]["method"]),
                int(artifact["effective_config"]["run"]["replicate_seed"]),
                int(round_result["cumulative_budget"]),
            )
            if logical_key in seen_keys:
                raise ValueError(
                    "Duplicate protocol result for "
                    f"dataset/framework/method/seed/budget={logical_key!r}"
                )
            seen_keys.add(logical_key)

            diagnostics = round_result["selection_diagnostics"]
            effective_config = deepcopy(artifact["effective_config"])
            run_config = effective_config.pop("run")
            rows.append(
                {
                    "protocol_version": artifact["protocol_version"],
                    "source_config_digest": config_digest(effective_config),
                    "run_id": artifact["run_id"],
                    "config_digest": artifact["config_digest"],
                    "dataset": effective_config["data"]["name"],
                    "framework": run_config["framework"],
                    "method": run_config["method"],
                    "replicate_seed": run_config["replicate_seed"],
                    "round": round_result["round"],
                    "query_size": round_result["query_size"],
                    "cumulative_budget": round_result["cumulative_budget"],
                    "trained_epochs": round_result["trained_epochs"],
                    "test_loss": round_result["test_loss"],
                    "test_accuracy": round_result["test_accuracy"],
                    "nearest_distance_mean": diagnostics["nearest_distance_mean"],
                    "nearest_distance_p95": diagnostics["nearest_distance_p95"],
                    "nearest_distance_max": diagnostics["nearest_distance_max"],
                    "selected_pairwise_cosine_mean": diagnostics["selected_pairwise_cosine_mean"],
                    "selected_typicality_mean": diagnostics["selected_typicality_mean"],
                    "selector_seconds": selection_seconds,
                    "new_indices_json": canonical_json(round_result["new_indices"]),
                    "selected_indices_json": canonical_json(round_result["selected_indices"]),
                    "method_metadata_json": canonical_json(round_result["method_metadata"]),
                }
            )

    if not rows:
        raise ValueError("Canonical artifacts contain no completed rounds")
    return (
        pd.DataFrame(rows, columns=ROUND_REPORT_COLUMNS)
        .sort_values(ROUND_SORT_COLUMNS, kind="stable")
        .reset_index(drop=True)
    )


def build_summary_metrics(round_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarise replicates while keeping datasets and frameworks separate."""
    required = set(ROUND_REPORT_COLUMNS)
    missing = sorted(required - set(round_metrics.columns))
    if missing:
        raise ValueError(f"Round metrics are missing required columns: {missing}")

    group_columns = ["dataset", "framework", "method", "cumulative_budget"]
    summary = (
        round_metrics.groupby(group_columns, as_index=False, sort=True)
        .agg(
            accuracy_mean=("test_accuracy", "mean"),
            accuracy_std=("test_accuracy", "std"),
            replicate_count=("test_accuracy", "count"),
            nearest_distance_mean=("nearest_distance_mean", "mean"),
            nearest_distance_p95_mean=("nearest_distance_p95", "mean"),
            nearest_distance_max_mean=("nearest_distance_max", "mean"),
            selected_pairwise_cosine_mean=("selected_pairwise_cosine_mean", "mean"),
            selected_typicality_mean=("selected_typicality_mean", "mean"),
            selector_seconds_mean=("selector_seconds", "mean"),
        )
        .sort_values(group_columns, kind="stable")
        .reset_index(drop=True)
    )
    return summary.loc[:, SUMMARY_COLUMNS]


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a stable CSV without exposing a partially written report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(
            temporary_path,
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_protocol_reports(root: Path) -> dict[str, Path]:
    """Validate canonical artifacts and write round and summary report tables."""
    artifacts, _ = load_validated_artifacts(root)
    round_metrics = build_round_metrics(artifacts)
    summary_metrics = build_summary_metrics(round_metrics)
    report_dir = root / "reports"
    paths = {
        "round_metrics": report_dir / "round_metrics.csv",
        "summary_metrics": report_dir / "summary_metrics.csv",
    }
    atomic_write_csv(round_metrics, paths["round_metrics"])
    atomic_write_csv(summary_metrics, paths["summary_metrics"])
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/protocol_v2"),
        help="Protocol artifact root containing runs/ (default: results/protocol_v2)",
    )
    return parser.parse_args()


def main() -> None:
    paths = write_protocol_reports(parse_args().root)
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    main()
