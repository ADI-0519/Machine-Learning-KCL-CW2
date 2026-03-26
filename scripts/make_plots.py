from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import pandas as pd
from pandas.errors import EmptyDataError

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ABLATION_METHODS = ["random", "tpcrand", "tpcrp", "tpcinv", "tpcnoclust"]
MODIFICATION_METHODS = ["tpcrp", "tpcrp_ccfl"]
SECONDARY_METHODS = ["kcenter", "uncertainty", "margin", "entropy", "dbal", "bald", "badge"]

def build_global_df_from_raw_metrics(metrics_path: Path) -> pd.DataFrame:
    """Recompute pooled method/budget means and SEs from raw metrics, not from already-aggregated framework rows."""
    if not metrics_path.exists():
        raise FileNotFoundError(f"Could not find raw metrics file at {metrics_path}")

    raw_df = pd.read_csv(metrics_path)

    required_cols = {"method", "budget", "seed", "best_test_accuracy"}
    missing = required_cols - set(raw_df.columns)
    if missing:
        raise ValueError(f"Raw metrics CSV is missing required columns: {missing}")

    dedup_keys = ["method", "budget", "seed"]
    if "framework" in raw_df.columns:
        dedup_keys = ["framework"] + dedup_keys
    raw_df = raw_df.drop_duplicates(subset=dedup_keys, keep="last")

    global_df = (
        raw_df.groupby(["method", "budget"], as_index=False)
        .agg(
            best_mean=("best_test_accuracy", "mean"),
            best_std=("best_test_accuracy", "std"),
            best_n=("best_test_accuracy", "count"),
        )
        .sort_values(["budget", "method"])
        .reset_index(drop=True)
    )

    global_df["best_std"] = global_df["best_std"].fillna(0.0)
    global_df["best_se"] = global_df["best_std"] / global_df["best_n"].pow(0.5)
    return global_df

def plot_group(df: pd.DataFrame,methods: Sequence[str],output_path: Path,title: str) -> None:
    """Plot mean best accuracy with standard-error bands for a selected method group."""
    plot_df = df[df["method"].isin(methods)].copy()
    if plot_df.empty:
        print(f"Skipping {output_path.name}: no rows for methods {list(methods)}")
        return

    plt.figure(figsize=(7.2, 4.4))
    for method in methods:
        method_df = plot_df[plot_df["method"] == method].sort_values("budget")
        if method_df.empty:
            continue
        x = method_df["budget"].to_numpy()
        y = (method_df["best_mean"] * 100).to_numpy()
        se = (method_df["best_se"] * 100).to_numpy()
        plt.plot(x, y, marker="o", linewidth=2.0, label=method)
        plt.fill_between(x, y - se, y + se, alpha=0.18)

    plt.xlabel("Budget")
    plt.ylabel("Best Test Accuracy (%)")
    plt.title(title)
    plt.grid(alpha=0.30)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {output_path}")


def main() -> None:
    """Create ablation, modification, and secondary-baseline plots from aggregated metrics."""
    aggregated_path = Path("results/metrics/aggregated_metrics.csv")
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not aggregated_path.exists():
        raise FileNotFoundError(
            f"Could not find aggregated metrics at {aggregated_path}. Run scripts/aggregate_results.py first."
        )

    try:
        df = pd.read_csv(aggregated_path)
    except EmptyDataError as exc:
        raise ValueError(
            f"Aggregated metrics file exists but is empty: {aggregated_path}. "
            "Run scripts/aggregate_results.py after experiments."
        ) from exc

    required_cols = {"method", "budget", "best_mean", "best_se"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Aggregated CSV is missing required columns: {missing}")


    metrics_path = Path("results/metrics/metrics.csv")
    try:
        global_df = build_global_df_from_raw_metrics(metrics_path)
        plot_group(
            df=global_df,
            methods=ABLATION_METHODS,
            output_path=output_dir / "ablation_plot_all_frameworks.png",
            title="Ablation Plot (All Frameworks)",
        )
        plot_group(
            df=global_df,
            methods=MODIFICATION_METHODS,
            output_path=output_dir / "modification_plot_all_frameworks.png",
            title="Modification Plot (All Frameworks)",
        )
        plot_group(
            df=global_df,
            methods=SECONDARY_METHODS,
            output_path=output_dir / "secondary_baselines_plot_all_frameworks.png",
            title="Secondary Baselines (All Frameworks)",
        )
    except FileNotFoundError:
        print("Skipping all-framework plots: raw metrics.csv not found.")

    # Per-framework plots when the column exists.
    if "framework" in df.columns:
        for framework in sorted(df["framework"].unique().tolist()):
            fw_df = df[df["framework"] == framework].copy()
            plot_group(
                df=fw_df,
                methods=ABLATION_METHODS,
                output_path=output_dir / f"ablation_plot_{framework}.png",
                title=f"Ablation Plot ({framework})",
            )
            plot_group(
                df=fw_df,
                methods=MODIFICATION_METHODS,
                output_path=output_dir / f"modification_plot_{framework}.png",
                title=f"Modification Plot ({framework})",
            )
            plot_group(
                df=fw_df,
                methods=SECONDARY_METHODS,
                output_path=output_dir / f"secondary_baselines_plot_{framework}.png",
                title=f"Secondary Baselines ({framework})",
            )


if __name__ == "__main__":
    main()
