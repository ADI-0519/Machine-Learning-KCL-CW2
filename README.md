# Machine-Learning-KCL-CW2

Reproduction and extension of **TypiClust / TPCRP** for low-budget active learning on **CIFAR-10**.

This repository contains:
- TPCRP-style selectors and baselines
- iterative active learning experiments across three frameworks
- result aggregation, statistical testing, and plotting scripts
- a custom modification: `tpcrp_ccfl`

## 1) Environment Setup

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

## 2) Data and Checkpoints

- CIFAR-10 is downloaded automatically via torchvision.
- SimCLR checkpoint expected at:
  - `results/checkpoints/simclr_resnet18.pt`
  - configurable via `simclr.save_path` in `configs/default.yaml`

Train SimCLR (if needed):

```bash
python -m scripts.train_simclr
```

## 3) Run Experiments

Default experiment config: `configs/default.yaml`.

Run full experiment grid configured in YAML:

```bash
python -m scripts.run_experiments
```

This appends rows to:
- `results/metrics/metrics.csv`

Each row key is:
- `(framework,method,budget,seed,rounds,best_epoch,best_test_accuracy,final_test_accuracy,final_test_loss,num_selected)`

## 4) Post-Processing Pipeline

Aggregate runs:

```bash
python -m scripts.aggregate_results
```

Statistical comparisons:

```bash
python -m scripts.run_stats
```

Generate plots:

```bash
python -m scripts.make_plots
```

Key outputs:
- `results/metrics/aggregated_metrics.csv`
- `results/metrics/aggregated_metrics_pivot.csv`
- `results/metrics/stat_tests.csv`
- `results/plots/*.png`

## 5) Implemented Methods

Core methods:
- `random`
- `tpcrand`
- `tpcrp`
- `tpcinv`
- `tpcnoclust`
- `kcenter`
- `uncertainty`
- `margin`
- `entropy`
- `dbal`
- `bald`
- `badge`

Modification:
- `tpcrp_ccfl`

## 6) Frameworks

- `fully_supervised`
- `ssl_embedding`
- `semi_supervised`

## 7) Important Notes / Limitations

- `tpcrp_ccfl` is implemented as an additional method and does not replace `tpcrp`.
- `semi_supervised` is a lightweight LabelSpreading-based pipeline (not a full FlexMatch reimplementation).
- `dbal`/`bald` are implemented as practical approximations consistent with this codebase design.

## 8) Reproducibility

- Seeds are controlled via:
  - top-level `seed`
  - per-experiment seeds in `experiment.seeds`
- Deterministic behavior depends on hardware/backend; minor run-to-run variation can still occur.
