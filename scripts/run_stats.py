from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from scipy.stats import ttest_rel, wilcoxon

COMPARISONS = [
    ("tpcrp", "random"),
    ("tpcrp", "tpcrand"),
    ("tpcrp_ccfl", "tpcrp"),
]


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    dedup_keys = ["method", "budget", "seed"]
    if "framework" in df.columns:
        dedup_keys = ["framework"] + dedup_keys
    return df.drop_duplicates(subset=dedup_keys, keep="last")


def cohens_dz(diff: np.ndarray) -> float:
    """Compute Cohen's dz for paired samples."""
    if diff.size < 2:
        return float("nan")
    sd = float(np.std(diff, ddof=1))
    if sd <= 1e-12:
        # All differences (almost) identical: dz undefined/infinite in theory.
        return 0.0 if abs(float(np.mean(diff))) <= 1e-12 else float("nan")
    return float(np.mean(diff) / sd)


def paired_row(sub: pd.DataFrame, method_a: str, method_b: str, budget: int) -> dict[str, Any]:
    """Compute paired statistics for one budget and method pair."""
    a = sub[sub["method"] == method_a][["seed", "best_test_accuracy"]].rename(columns={"best_test_accuracy": "acc_a"})
    b = sub[sub["method"] == method_b][["seed", "best_test_accuracy"]].rename(columns={"best_test_accuracy": "acc_b"})
    merged = pd.merge(a, b, on="seed", how="inner")

    n = int(len(merged))
    mean_a = float(merged["acc_a"].mean()) if n else float("nan")
    mean_b = float(merged["acc_b"].mean()) if n else float("nan")
    mean_diff = float((merged["acc_a"] - merged["acc_b"]).mean()) if n else float("nan")

    if n < 2:
        t_stat, t_p = float("nan"), float("nan")
    else:
        t_stat, t_p = ttest_rel(merged["acc_a"], merged["acc_b"])
        t_stat, t_p = float(t_stat), float(t_p)

    if n == 0:
        w_stat, w_p = float("nan"), float("nan")
    else:
        diff = (merged["acc_a"] - merged["acc_b"]).to_numpy(dtype=float)
        if np.allclose(diff, 0.0):
            # Wilcoxon is undefined when all diffs are zero.
            w_stat, w_p = 0.0, 1.0
        else:
            try:
                w = wilcoxon(diff, zero_method="wilcox", correction=False, alternative="two-sided")
                w_stat, w_p = float(w.statistic), float(w.pvalue)
            except ValueError:
                # Graceful fallback in pathological edge cases.
                w_stat, w_p = float("nan"), float("nan")

    dz = cohens_dz((merged["acc_a"] - merged["acc_b"]).to_numpy(dtype=float)) if n else float("nan")

    return {
        "budget": int(budget),
        "method_a": method_a,
        "method_b": method_b,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": mean_diff,
        "t_stat": t_stat,
        "t_p_value": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p_value": w_p,
        "cohens_dz": dz,
        "n_pairs": n,
    }


def run_framework(df: pd.DataFrame, framework: str | None) -> pd.DataFrame:
    """Run all configured pairwise comparisons for one framework slice."""
    if framework is not None:
        sub_df = df[df["framework"] == framework]
    else:
        sub_df = df

    rows: list[dict[str, Any]] = []
    for budget in sorted(sub_df["budget"].unique().tolist()):
        bdf = sub_df[sub_df["budget"] == budget]
        methods_present = set(bdf["method"].unique().tolist())
        for method_a, method_b in COMPARISONS:
            if method_a not in methods_present or method_b not in methods_present:
                continue
            row = paired_row(bdf, method_a, method_b, int(budget))
            if framework is not None:
                row["framework"] = framework
            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    """Run paired t-test + Wilcoxon + Cohen's dz and save stats_summary.csv."""
    metrics_path = Path("results/metrics/metrics.csv")
    output_dir = Path("results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Could not find metrics file at {metrics_path}")

    try:
        df = pd.read_csv(metrics_path)
    except EmptyDataError as exc:
        raise ValueError(
            f"Metrics file exists but is empty: {metrics_path}. Run experiments first to generate rows."
        ) from exc

    required = {"method", "budget", "seed", "best_test_accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metrics CSV is missing required columns: {missing}")

    df = dedupe(df)

    all_results: list[pd.DataFrame] = []
    if "framework" in df.columns:
        for framework in sorted(df["framework"].unique().tolist()):
            out = run_framework(df, framework=framework)
            if not out.empty:
                all_results.append(out)
    else:
        out = run_framework(df, framework=None)
        if not out.empty:
            all_results.append(out)

    if not all_results:
        raise ValueError("No comparable method pairs found. Check metrics coverage.")

    stats_df = pd.concat(all_results, ignore_index=True)
    sort_cols = [c for c in ["framework", "budget", "method_a", "method_b"] if c in stats_df.columns]
    stats_df = stats_df.sort_values(sort_cols)

    output_path = output_dir / "stats_summary.csv"
    stats_df.to_csv(output_path, index=False)

    print(f"Saved statistical summary to: {output_path}")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
