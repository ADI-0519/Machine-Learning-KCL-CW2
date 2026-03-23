from src.experiment import run_single_experiment

def main() -> None:
    methods = ["random", "tpcrand", "tpcrp"]

    for method in methods:
        print(f"\nRunning method: {method}\n")
        run_single_experiment(
            config_path="configs/default.yaml",
            method=method,
            budget=10,
            seed=21
        )


if __name__ == "__main__":
    main()