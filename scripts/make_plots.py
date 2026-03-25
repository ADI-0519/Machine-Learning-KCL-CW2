from pathlib import Path
import matplotlib
import pandas as pd
from pandas.errors import EmptyDataError

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    aggregated_path = Path("results/metrics/aggregated_metrics.csv")
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not aggregated_path.exists():
        raise FileNotFoundError(
            f"Could not find aggregated metrics at {aggregated_path}. "
            "Run scripts/aggregate_results.py first."
        )

    try:
        df = pd.read_csv(aggregated_path)
    except EmptyDataError as exc:
        raise ValueError(
            f"Aggregated metrics file exists but is empty: {aggregated_path}. "
            "Run scripts/aggregate_results.py after experiments."
        ) from exc

    required_cols = {"method", "budget", "mean_accuracy", "std_accuracy"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Aggregated CSV is missing required columns: {missing}")

    frameworks = sorted(df["framework"].unique()) if "framework" in df.columns else [None]

    for framework in frameworks:
        plot_df = df if framework is None else df[df["framework"] == framework]

        plt.figure(figsize=(6.5, 4.0))
        for method in sorted(plot_df["method"].unique()):
            method_df = plot_df[plot_df["method"] == method].sort_values("budget")
            plt.plot(
                method_df["budget"],
                method_df["mean_accuracy"] * 100,
                marker="o",
                label=method,
            )
            plt.fill_between(
                method_df["budget"],
                (method_df["mean_accuracy"] - method_df["std_accuracy"]) * 100,
                (method_df["mean_accuracy"] + method_df["std_accuracy"]) * 100,
                alpha=0.15,
            )

        plt.xlabel("Budget")
        plt.ylabel("Test Accuracy (%)")
        title = "Accuracy vs Budget" if framework is None else f"Accuracy vs Budget ({framework})"
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = "accuracy_vs_budget.png" if framework is None else f"accuracy_vs_budget_{framework}.png"
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to: {output_path}")

if __name__ == "__main__":
    main()
