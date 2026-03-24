from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def main():
    aggregated_path = Path("results/metrics/aggregated_metrics.csv")
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not aggregated_path.exists():
        raise FileNotFoundError(
            f"Could not find aggregated metrics at {aggregated_path}. "
            "Run scripts/aggregate_results.py first."
        )

    df = pd.read_csv(aggregated_path)

    required_cols = {"method", "budget", "mean_accuracy", "std_accuracy"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Aggregated CSV is missing required columns: {missing}")

    plt.figure(figsize=(6.5, 4.0))

    for method in sorted(df["method"].unique()):
        method_df = df[df["method"] == method].sort_values("budget")

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
    plt.title("Accuracy vs Budget")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "accuracy_vs_budget.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to: {output_path}")

if __name__ == "__main__":
    main()