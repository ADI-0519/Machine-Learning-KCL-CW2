"""Plot framework-separated protocol results with explicit variability bands."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.aggregate_results import (
    build_round_metrics,
    build_summary_metrics,
    load_validated_artifacts,
)


def _filename_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not component:
        raise ValueError(f"Cannot create a safe plot filename from {value!r}")
    return component


def write_protocol_plots(root: Path) -> list[Path]:
    """Write one accuracy plot per dataset/framework; frameworks are never pooled."""
    artifacts, _ = load_validated_artifacts(root)
    summary = build_summary_metrics(build_round_metrics(artifacts))
    insufficient = summary[summary["replicate_count"] < 2]
    if not insufficient.empty:
        keys = insufficient[
            ["dataset", "framework", "method", "cumulative_budget", "replicate_count"]
        ]
        raise ValueError(
            "Standard-deviation bands require at least two replicates per plotted point: "
            f"{keys.to_dict(orient='records')}"
        )
    output_dir = root / "reports" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for (dataset, framework), group in summary.groupby(["dataset", "framework"], sort=True):
        figure, axis = plt.subplots(figsize=(7.2, 4.4))
        for method in sorted(group["method"].unique()):
            method_rows = group[group["method"] == method].sort_values("cumulative_budget")
            budgets = method_rows["cumulative_budget"].to_numpy(dtype=int)
            accuracy = method_rows["accuracy_mean"].to_numpy(dtype=float) * 100.0
            deviation = method_rows["accuracy_std"].to_numpy(dtype=float) * 100.0
            axis.plot(budgets, accuracy, marker="o", linewidth=2.0, label=method)
            axis.fill_between(
                budgets,
                accuracy - deviation,
                accuracy + deviation,
                alpha=0.18,
            )

        axis.set_xlabel("Cumulative label budget")
        axis.set_ylabel("Test accuracy (%) with ±1 SD bands")
        axis.set_title(f"{dataset} — {framework}")
        axis.grid(alpha=0.30)
        axis.legend()
        figure.tight_layout()
        output_path = output_dir / (
            f"accuracy_{_filename_component(str(dataset))}_"
            f"{_filename_component(str(framework))}.png"
        )
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)

    if not output_paths:
        raise ValueError("No dataset/framework groups were available for plotting")
    return output_paths


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
    for path in write_protocol_plots(parse_args().root):
        print(f"Wrote plot: {path}")


if __name__ == "__main__":
    main()
