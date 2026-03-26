from src.config import load_configurations
from src.experiment import run_single_experiment


def main() -> None:
    """Run configured experiment grid across frameworks, methods, budgets and seeds."""
    cfg = load_configurations("configs/default.yaml")

    methods = cfg["experiment"]["methods"]
    budgets = cfg["selection"]["budgets"]
    seeds = cfg["experiment"]["seeds"]
    frameworks = cfg["experiment"].get("frameworks", [cfg["experiment"].get("framework", "fully_supervised")])

    for framework in frameworks:
        for seed in seeds:
            for budget in budgets:
                for method in methods:
                    print("\n" + "=" * 80)
                    print(
                        f"Running framework={framework}, method={method}, "
                        f"budget={budget}, seed={seed}"
                    )
                    print("=" * 80)
                    run_single_experiment(
                        config_path="configs/default.yaml",
                        method=method,
                        budget=budget,
                        seed=seed,
                        framework=framework,
                    )


if __name__ == "__main__":
    main()
