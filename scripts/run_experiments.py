from src.config import load_configurations
from src.experiment import run_single_experiment

def main() -> None:
    cfg = load_configurations("configs/default.yaml")

    methods = cfg["experiment"]["methods"]
    budgets = cfg["selection"]["budgets"]
    seeds = cfg["experiment"]["seeds"]

    for seed in seeds:
        for budget in budgets:
            for method in methods:
                print("\n" + "=" * 80)
                print(f"Running method={method}, budget={budget}, seed={seed}")
                print("=" * 80)
                run_single_experiment(
                    config_path="configs/default.yaml",
                    method=method,
                    budget=budget,
                    seed=seed
                )


if __name__ == "__main__":
    main()