from pathlib import Path
import pandas as pd
from scipy.stats import ttest_rel

def paired_test(df: pd.DataFrame, method_a: str, method_b: str) -> pd.DataFrame:
    rows = []

    for budget in sorted(df["budget"].unique()):
        sub = df[df["budget"] == budget]

        a = sub[sub["method"] == method_a][["seed", "best_test_accuracy"]].rename(
            columns={"best_test_accuracy": "acc_a"}
        )
        b = sub[sub["method"] == method_b][["seed", "best_test_accuracy"]].rename(
            columns={"best_test_accuracy": "acc_b"}
        )

        merged = pd.merge(a, b, on="seed", how="inner")

        if len(merged) < 2:
            p_value = float("nan")
            t_stat = float("nan")
        else:
            t_stat, p_value = ttest_rel(merged["acc_a"], merged["acc_b"])

        rows.append(
            {
                "budget": budget,
                "method_a": method_a,
                "method_b": method_b,
                "mean_a": merged["acc_a"].mean() if len(merged) else float("nan"),
                "mean_b": merged["acc_b"].mean() if len(merged) else float("nan"),
                "t_stat": t_stat,
                "p_value": p_value,
                "n_pairs": len(merged),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    metrics_path = Path("results/metrics/metrics.csv")
    output_dir = Path("results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Could not find metrics file at {metrics_path}")

    df = pd.read_csv(metrics_path)
    df = df.drop_duplicates(subset=["method", "budget", "seed"], keep="last")

    comparisons = [
        ("tpcrp", "random"),
        ("tpcrp", "tpcrand"),
    ]

    all_results = []
    for method_a, method_b in comparisons:
        result = paired_test(df, method_a, method_b)
        all_results.append(result)

    stats_df = pd.concat(all_results, ignore_index=True)
    output_path = output_dir / "stats_summary.csv"
    stats_df.to_csv(output_path, index=False)

    print(f"Saved statistical summary to: {output_path}")
    print(stats_df)


if __name__ == "__main__":
    main()