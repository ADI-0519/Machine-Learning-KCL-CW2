"""Compute deterministic paired inference from canonical protocol artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from scripts.aggregate_results import (
    atomic_write_csv,
    build_round_metrics,
    load_validated_artifacts,
)
from src.statistics import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    holm_adjust,
    pair_method_rows,
    paired_bootstrap_ci,
)

STATISTICS_COLUMNS = [
    "dataset",
    "framework",
    "method_a",
    "method_b",
    "cumulative_budget",
    "pair_count",
    "method_a_mean_accuracy",
    "method_b_mean_accuracy",
    "paired_mean_difference",
    "paired_ci95_low",
    "paired_ci95_high",
    "method_a_wins",
    "t_statistic",
    "raw_p_value",
    "holm_adjusted_p_value",
    "bootstrap_confidence",
    "bootstrap_resamples",
    "bootstrap_seed",
]


def _one_sample_test(differences: np.ndarray) -> tuple[float, float]:
    """Run a two-sided paired t-test, handling constant differences explicitly."""
    if differences.size < 2:
        raise ValueError("Paired inference requires at least two replicate pairs")
    if np.all(differences == differences[0]):
        mean_difference = float(np.mean(differences))
        if mean_difference == 0.0:
            return 0.0, 1.0
        return float(np.copysign(np.inf, mean_difference)), 0.0
    result = ttest_1samp(differences, popmean=0.0, alternative="two-sided")
    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    if not np.isfinite(statistic) or not np.isfinite(p_value):
        raise ValueError("Paired t-test returned a non-finite result")
    return statistic, p_value


def _comparison_config(source_config: dict[str, Any]) -> tuple[str, str, str]:
    comparison = source_config["experiment"]["primary_comparison"]
    return (
        str(comparison["method_a"]),
        str(comparison["method_b"]),
        str(comparison["metric"]),
    )


def write_paired_statistics(root: Path) -> pd.DataFrame:
    """Write paired results per dataset, framework, and cumulative budget."""
    artifacts, source_config = load_validated_artifacts(root)
    round_metrics = build_round_metrics(artifacts)
    method_a, method_b, metric = _comparison_config(source_config)
    if metric != "test_accuracy":
        raise ValueError(
            f"Phase 7 supports primary_comparison.metric='test_accuracy'; received {metric!r}"
        )

    paired = pair_method_rows(round_metrics, method_a=method_a, method_b=method_b)
    group_columns = ["dataset", "framework", "cumulative_budget"]
    rows: list[dict[str, Any]] = []
    raw_p_values: list[float] = []

    for group_key, group in paired.groupby(group_columns, sort=True):
        dataset, framework, cumulative_budget = group_key
        differences = group["test_accuracy_a"].to_numpy(dtype=float) - group[
            "test_accuracy_b"
        ].to_numpy(dtype=float)
        interval_low, interval_high = paired_bootstrap_ci(differences)
        t_statistic, p_value = _one_sample_test(differences)
        raw_p_values.append(p_value)
        rows.append(
            {
                "dataset": dataset,
                "framework": framework,
                "method_a": method_a,
                "method_b": method_b,
                "cumulative_budget": int(cumulative_budget),
                "pair_count": int(differences.size),
                "method_a_mean_accuracy": float(group["test_accuracy_a"].mean()),
                "method_b_mean_accuracy": float(group["test_accuracy_b"].mean()),
                "paired_mean_difference": float(np.mean(differences)),
                "paired_ci95_low": interval_low,
                "paired_ci95_high": interval_high,
                "method_a_wins": int(np.count_nonzero(differences > 0.0)),
                "t_statistic": t_statistic,
                "raw_p_value": p_value,
                "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
        )

    if not rows:
        raise ValueError("No primary-comparison rows were available for paired inference")
    adjusted_p_values = holm_adjust(raw_p_values)
    for row, adjusted_p_value in zip(rows, adjusted_p_values, strict=True):
        row["holm_adjusted_p_value"] = adjusted_p_value

    statistics = (
        pd.DataFrame(rows, columns=STATISTICS_COLUMNS)
        .sort_values(group_columns, kind="stable")
        .reset_index(drop=True)
    )
    atomic_write_csv(statistics, root / "reports" / "paired_statistics.csv")
    return statistics


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
    statistics = write_paired_statistics(root)
    print(f"Wrote paired statistics: {root / 'reports' / 'paired_statistics.csv'}")
    print(statistics.to_string(index=False))


if __name__ == "__main__":
    main()
