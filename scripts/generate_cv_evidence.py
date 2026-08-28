"""Generate traceable CV evidence from the locked primary comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from scripts.aggregate_results import build_round_metrics, load_validated_artifacts
from src.artifacts import atomic_write_json
from src.statistics import pair_method_rows, paired_bootstrap_ci

MINIMUM_VALIDATED_PAIRS = 5


def _single_value(values: set[str], label: str) -> str:
    if len(values) != 1:
        raise ValueError(
            f"Primary evidence must contain exactly one {label}; found {sorted(values)}"
        )
    return next(iter(values))


def generate_cv_evidence(root: Path) -> dict[str, Any]:
    """Compute the Section 10 evidence fields from validated artifact pairs."""
    artifacts, source_config = load_validated_artifacts(root)
    round_metrics = build_round_metrics(artifacts)
    comparison = source_config["experiment"]["primary_comparison"]
    method_a = str(comparison["method_a"])
    method_b = str(comparison["method_b"])
    primary_budget = int(comparison["cumulative_budget"])
    if comparison["metric"] != "test_accuracy":
        raise ValueError("CV evidence requires primary_comparison.metric='test_accuracy'")
    if (method_a, method_b) != ("tpcrp_ccfl", "tpcrp"):
        raise ValueError(
            "CV evidence field names require method_a='tpcrp_ccfl' and method_b='tpcrp'"
        )

    paired_all_budgets = pair_method_rows(
        round_metrics,
        method_a=method_a,
        method_b=method_b,
    )
    primary = paired_all_budgets.loc[
        paired_all_budgets["cumulative_budget"] == primary_budget
    ].copy()
    if primary.empty:
        raise ValueError(f"No paired rows exist at primary budget {primary_budget}")
    _single_value(set(primary["dataset"].astype(str)), "dataset")
    _single_value(set(primary["framework"].astype(str)), "framework")

    actual_seeds = sorted(int(seed) for seed in primary["replicate_seed"])
    expected_seeds = sorted(int(seed) for seed in source_config["experiment"]["replicate_seeds"])
    if actual_seeds != expected_seeds:
        raise ValueError(
            "Primary evidence replicate seeds do not match the configured confirmation set: "
            f"expected {expected_seeds}, found {actual_seeds}"
        )
    if len(actual_seeds) < MINIMUM_VALIDATED_PAIRS:
        raise ValueError(
            f"CV evidence requires at least {MINIMUM_VALIDATED_PAIRS} validated replicate pairs"
        )

    differences = primary["test_accuracy_a"].to_numpy(dtype=float) - primary[
        "test_accuracy_b"
    ].to_numpy(dtype=float)
    interval_low, interval_high = paired_bootstrap_ci(differences)
    mean_distance_a = float(primary["nearest_distance_mean_a"].mean())
    mean_distance_b = float(primary["nearest_distance_mean_b"].mean())
    if mean_distance_b == 0.0:
        raise ValueError("Cannot compute coverage improvement with zero method-B distance")
    mean_runtime_a = float(primary["selector_seconds_a"].mean())
    mean_runtime_b = float(primary["selector_seconds_b"].mean())
    if mean_runtime_b == 0.0:
        raise ValueError("Cannot compute selector runtime ratio with zero method-B runtime")

    run_ids = sorted(set(primary["run_id_a"].astype(str)).union(primary["run_id_b"].astype(str)))
    evidence: dict[str, Any] = {
        "protocol_version": str(artifacts[0]["protocol_version"]),
        "total_runs": len({str(artifact["run_id"]) for artifact in artifacts}),
        "total_model_fits": len(round_metrics),
        "datasets": sorted(round_metrics["dataset"].astype(str).unique().tolist()),
        "methods": sorted(round_metrics["method"].astype(str).unique().tolist()),
        "replicate_count": len(actual_seeds),
        "primary_budget": primary_budget,
        "ccfl_mean_accuracy": float(primary["test_accuracy_a"].mean()),
        "tpcrp_mean_accuracy": float(primary["test_accuracy_b"].mean()),
        "paired_improvement_pp": float(np.mean(differences) * 100.0),
        "paired_ci95_pp": [interval_low * 100.0, interval_high * 100.0],
        "coverage_improvement_percent": (
            (mean_distance_b - mean_distance_a) / mean_distance_b * 100.0
        ),
        "selector_runtime_ratio": mean_runtime_a / mean_runtime_b,
        "run_ids": run_ids,
    }
    atomic_write_json(root / "reports" / "cv_evidence.json", evidence)
    return evidence


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
    root = parse_args().root
    generate_cv_evidence(root)
    print(f"Wrote CV evidence: {root / 'reports' / 'cv_evidence.json'}")


if __name__ == "__main__":
    main()
