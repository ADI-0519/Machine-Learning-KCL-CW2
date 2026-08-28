"""Evaluate the frozen five-seed Gate-B decision without manual arithmetic."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from scripts.aggregate_results import build_round_metrics, load_validated_artifacts
from scripts.generate_cv_evidence import generate_cv_evidence
from src.artifacts import atomic_write_json
from src.statistics import pair_method_rows


def evaluate_pilot_gate(root: Path) -> dict[str, Any]:
    """Write and return the predeclared Gate-B checks for one pilot root."""
    evidence = generate_cv_evidence(root)
    artifacts, config = load_validated_artifacts(root)
    configured_seeds = sorted(int(seed) for seed in config["experiment"]["replicate_seeds"])
    if len(configured_seeds) != 5:
        raise ValueError("Gate B requires exactly five configured replicate seeds")
    comparison = config["experiment"]["primary_comparison"]
    rows = build_round_metrics(artifacts)
    paired = pair_method_rows(
        rows,
        method_a=str(comparison["method_a"]),
        method_b=str(comparison["method_b"]),
    )
    paired = paired.loc[
        paired["cumulative_budget"] == int(comparison["cumulative_budget"])
    ]
    differences = paired["test_accuracy_a"].to_numpy(dtype=float) - paired[
        "test_accuracy_b"
    ].to_numpy(dtype=float)
    wins = int(np.count_nonzero(differences > 0.0))
    checks = {
        "five_valid_pairs": evidence["replicate_count"] == 5,
        "positive_mean_improvement": evidence["paired_improvement_pp"] > 0.0,
        "wins_at_least_four_of_five": wins >= 4,
        "mean_improvement_at_least_3pp": evidence["paired_improvement_pp"] >= 3.0,
        "coverage_improves": evidence["coverage_improvement_percent"] > 0.0,
        "selector_runtime_at_most_2x": evidence["selector_runtime_ratio"] <= 2.0,
    }
    decision = {
        "protocol_version": evidence["protocol_version"],
        "primary_budget": evidence["primary_budget"],
        "paired_improvement_pp": evidence["paired_improvement_pp"],
        "wins": wins,
        "replicate_count": evidence["replicate_count"],
        "coverage_improvement_percent": evidence["coverage_improvement_percent"],
        "selector_runtime_ratio": evidence["selector_runtime_ratio"],
        "checks": checks,
        "pass": all(checks.values()),
        "run_ids": evidence["run_ids"],
    }
    atomic_write_json(root / "reports" / "pilot_gate.json", decision)
    return decision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/protocol_v2"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    decision = evaluate_pilot_gate(args.root)
    status = "PASS" if decision["pass"] else "FAIL"
    print(f"Gate B: {status}; evidence={args.root / 'reports' / 'pilot_gate.json'}")
    if not decision["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
