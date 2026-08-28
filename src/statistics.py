"""Pure pairing and multiplicity helpers for protocol-v2 reporting."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import pandas as pd

PAIRING_KEYS = (
    "dataset",
    "framework",
    "cumulative_budget",
    "replicate_seed",
)
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_201


def pair_method_rows(
    rows: pd.DataFrame,
    *,
    method_a: str,
    method_b: str,
) -> pd.DataFrame:
    """Return deterministic one-to-one method rows joined on protocol pairing keys."""
    if not isinstance(rows, pd.DataFrame):
        raise ValueError("rows must be a pandas DataFrame")
    if not isinstance(method_a, str) or not method_a:
        raise ValueError("method_a must be a non-empty string")
    if not isinstance(method_b, str) or not method_b:
        raise ValueError("method_b must be a non-empty string")
    if method_a == method_b:
        raise ValueError("method_a and method_b must differ")

    required = {*PAIRING_KEYS, "method", "run_id"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"pairing table is missing columns: {sorted(missing)}")

    relevant = rows.loc[rows["method"].isin((method_a, method_b))].copy()
    if relevant.empty:
        raise ValueError("pairing table contains neither configured comparison method")
    duplicate_mask = relevant.duplicated([*PAIRING_KEYS, "method"], keep=False)
    if duplicate_mask.any():
        duplicate_keys = relevant.loc[
            duplicate_mask,
            [*PAIRING_KEYS, "method"],
        ].sort_values([*PAIRING_KEYS, "method"])
        raise ValueError(
            f"duplicate pairing key for method rows: {duplicate_keys.to_dict(orient='records')}"
        )

    left = relevant.loc[relevant["method"] == method_a].drop(columns="method")
    right = relevant.loc[relevant["method"] == method_b].drop(columns="method")
    paired = left.merge(
        right,
        on=list(PAIRING_KEYS),
        how="outer",
        suffixes=("_a", "_b"),
        indicator=True,
        validate="one_to_one",
        sort=False,
    )
    missing_partner = paired["_merge"] != "both"
    if missing_partner.any():
        missing_keys = paired.loc[missing_partner, [*PAIRING_KEYS, "_merge"]]
        raise ValueError(
            f"missing method partner for pairing keys: {missing_keys.to_dict(orient='records')}"
        )

    paired = paired.drop(columns="_merge").sort_values(list(PAIRING_KEYS))
    return paired.reset_index(drop=True)


def paired_bootstrap_ci(
    differences: np.ndarray,
    *,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Return a deterministic percentile interval for a vector of paired differences."""
    raw_differences = np.asarray(differences)
    if (
        raw_differences.ndim != 1
        or len(raw_differences) < 2
        or not np.issubdtype(raw_differences.dtype, np.number)
        or np.issubdtype(raw_differences.dtype, np.bool_)
    ):
        raise ValueError("differences must be a finite vector with at least two values")
    diff = raw_differences.astype(np.float64, copy=False)
    if not np.isfinite(diff).all():
        raise ValueError("differences must be a finite vector with at least two values")
    if (
        not isinstance(confidence, Real)
        or isinstance(confidence, (bool, np.bool_))
        or not np.isfinite(confidence)
        or not 0.0 < float(confidence) < 1.0
    ):
        raise ValueError("confidence must be a finite number in (0, 1)")
    if (
        not isinstance(resamples, Integral)
        or isinstance(resamples, (bool, np.bool_))
        or resamples <= 0
    ):
        raise ValueError("resamples must be a positive integer")
    if not isinstance(seed, Integral) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(diff), size=(int(resamples), len(diff)))
    means = diff[indices].mean(axis=1)
    alpha = 1.0 - float(confidence)
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return Holm-adjusted p-values in the same order as the input vector."""
    raw_values = np.asarray(p_values)
    if (
        raw_values.ndim != 1
        or not len(raw_values)
        or not np.issubdtype(raw_values.dtype, np.number)
        or np.issubdtype(raw_values.dtype, np.bool_)
    ):
        raise ValueError("p-values must be a non-empty one-dimensional vector in [0, 1]")
    values = raw_values.astype(np.float64, copy=False)
    if not np.isfinite(values).all() or not ((values >= 0.0) & (values <= 1.0)).all():
        raise ValueError("p-values must be a non-empty one-dimensional vector in [0, 1]")

    order = np.argsort(values, kind="stable")
    ranked = values[order]
    multipliers = np.arange(len(values), 0, -1, dtype=np.float64)
    adjusted_ranked = np.maximum.accumulate(np.minimum(1.0, ranked * multipliers))
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted
