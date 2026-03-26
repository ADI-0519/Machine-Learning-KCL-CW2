from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError

MAIN_METHODS = ["random", "tpcrand", "tpcrp", "tpcinv", "tpcnoclust", "tpcrp_ccfl"]
MOD_METHODS = ["tpcrp", "tpcrp_ccfl"]

def format_mean_std(mean: float, std: float) -> str:
    """Format mean and std accuracy as a percent plus-minus string."""
    return f"{mean * 100:.2f} $\\pm$ {std * 100:.2f}"


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    dedup_keys = ["method", "budget", "seed"]
    if "framework" in df.columns:
        dedup_keys = ["framework"] + dedup_keys
    return df.drop_duplicates(subset=dedup_keys, keep="last")


def _aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    group_keys = ["method", "budget"]
    if "framework" in df.columns:
        group_keys = ["framework"] + group_keys

    grouped = (
        df.groupby(group_keys, as_index=False)
        .agg(
            best_mean=("best_test_accuracy", "mean"),
            best_std=("best_test_accuracy", "std"),
            best_n=("best_test_accuracy", "count"),
            final_mean=("final_test_accuracy", "mean"),
            final_std=("final_test_accuracy", "std"),
            final_n=("final_test_accuracy", "count"),
        )
        .sort_values([k for k in ["framework", "budget", "method"] if k in group_keys])
    )

    grouped["best_std"] = grouped["best_std"].fillna(0.0)
    grouped["final_std"] = grouped["final_std"].fillna(0.0)

    grouped["best_se"] = grouped["best_std"] / grouped["best_n"].pow(0.5)
    grouped["final_se"] = grouped["final_std"] / grouped["final_n"].pow(0.5)

    grouped["best_ci95"] = 1.96 * grouped["best_se"]
    grouped["final_ci95"] = 1.96 * grouped["final_se"]

    grouped["best_mean_pm_std"] = grouped.apply(
        lambda row: format_mean_std(float(row["best_mean"]), float(row["best_std"])),
        axis=1,
    )
    grouped["final_mean_pm_std"] = grouped.apply(
        lambda row: format_mean_std(float(row["final_mean"]), float(row["final_std"])),
        axis=1,
    )
    return grouped


def _write_pivots(df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    index_cols: list[str] = ["method"]
    if "framework" in df.columns:
        index_cols = ["framework", "method"]

    best_pivot = (
        df.pivot(index=index_cols, columns="budget", values="best_mean_pm_std")
        .sort_index()
        .sort_index(axis=1)
    )
    final_pivot = (
        df.pivot(index=index_cols, columns="budget", values="final_mean_pm_std")
        .sort_index()
        .sort_index(axis=1)
    )

    best_path = output_dir / f"{stem}_pivot_best.csv"
    final_path = output_dir / f"{stem}_pivot_final.csv"
    best_pivot.to_csv(best_path)
    final_pivot.to_csv(final_path)
    print(f"Saved pivot (best) to: {best_path}")
    print(f"Saved pivot (final) to: {final_path}")


def print_latex_rows(df: pd.DataFrame, title: str) -> None:
    print(f"\nLaTeX-ready rows ({title}, best accuracy mean ± std):")
    if df.empty:
        print("No rows.")
        return

    budgets = sorted(df["budget"].unique().tolist())
    if "framework" in df.columns:
        for framework in sorted(df["framework"].unique().tolist()):
            sub = df[df["framework"] == framework]
            print(f"% framework: {framework}")
            for method in sorted(sub["method"].unique().tolist()):
                method_row = sub[sub["method"] == method].set_index("budget")
                vals = [method_row.loc[b, "best_mean_pm_std"] if b in method_row.index else "--" for b in budgets]
                print(" & ".join([method] + vals) + r" \\")
    else:
        for method in sorted(df["method"].unique().tolist()):
            method_row = df[df["method"] == method].set_index("budget")
            vals = [method_row.loc[b, "best_mean_pm_std"] if b in method_row.index else "--" for b in budgets]
            print(" & ".join([method] + vals) + r" \\")


def main() -> None:
    """Aggregate metrics with SE/CI, save subsets/pivots, and print LaTeX-ready rows."""
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

    required = {"method", "budget", "seed", "best_test_accuracy", "final_test_accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metrics CSV is missing required columns: {missing}")

    df = dedupe(df)
    agg = _aggregate_metrics(df)

    all_path = output_dir / "aggregated_metrics.csv"
    agg.to_csv(all_path, index=False)
    print(f"Saved aggregated metrics to: {all_path}")

    agg_main = agg[agg["method"].isin(MAIN_METHODS)].copy()
    main_path = output_dir / "aggregated_metrics_main.csv"
    agg_main.to_csv(main_path, index=False)
    print(f"Saved main-methods aggregate to: {main_path}")

    agg_mod = agg[agg["method"].isin(MOD_METHODS)].copy()
    mod_path = output_dir / "aggregated_metrics_modification.csv"
    agg_mod.to_csv(mod_path, index=False)
    print(f"Saved modification aggregate to: {mod_path}")

    _write_pivots(agg, output_dir, "aggregated_metrics")
    _write_pivots(agg_main, output_dir, "aggregated_metrics_main")
    _write_pivots(agg_mod, output_dir, "aggregated_metrics_modification")

    print_latex_rows(agg_main, "main methods")
    print_latex_rows(agg_mod, "modification methods")


if __name__ == "__main__":
    main()
