from pathlib import Path
import pandas as pd

def format_mean_std(mean,std):
    return f"{mean*100:.2f} $\\pm$ {std*100:.2f}"

def main():
    metrics_path = Path("results/metrics/metrics.csv")
    output_dir = Path("results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Could not find metrics file at {metrics_path}")

    df = pd.read_csv(metrics_path)

    required_cols = {"method", "budget", "seed", "best_test_accuracy"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Metrics CSV is missing required columns: {missing}")

    # Drop duplicate experiment rows if any slipped in
    df = df.drop_duplicates(subset=["method", "budget", "seed"], keep="last")

    grouped = (
        df.groupby(["method", "budget"], as_index=False)
        .agg(
            mean_accuracy=("best_test_accuracy", "mean"),
            std_accuracy=("best_test_accuracy", "std"),
            n_runs=("best_test_accuracy", "count"),
        )
        .sort_values(["budget", "method"])
    )

    grouped["std_accuracy"] = grouped["std_accuracy"].fillna(0.0)
    grouped["mean_pm_std"] = grouped.apply(
        lambda row: format_mean_std(row["mean_accuracy"], row["std_accuracy"]),
        axis=1,
    )

    aggregated_path = output_dir / "aggregated_metrics.csv"
    grouped.to_csv(aggregated_path, index=False)

    pivot = grouped.pivot(index="method", columns="budget", values="mean_pm_std")
    pivot = pivot.sort_index()

    pivot_path = output_dir / "aggregated_metrics_pivot.csv"
    pivot.to_csv(pivot_path)

    print(f"Saved aggregated metrics to: {aggregated_path}")
    print(f"Saved pivot table to: {pivot_path}")

    print("\nLaTeX table rows:")
    for method in pivot.index:
        values = [pivot.loc[method, col] if col in pivot.columns else "--" for col in pivot.columns]
        row = " & ".join([method] + values) + r" \\"
        print(row)

if __name__ == "__main__":
    main()