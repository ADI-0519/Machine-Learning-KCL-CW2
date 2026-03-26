# Machine-Learning-KCL-CW2

Reproduction and extension of **TypiClust / TPCRP** for low-budget active learning on **CIFAR-10**, including a custom modification: **`tpcrp_ccfl`**.

This project was built for coursework focused on:
- implementing TPCRP from the paper,
- reproducing core experimental comparisons,
- proposing and evaluating a modification.

## Repository Structure

```text
Machine-Learning-KCL-CW2/
├── configs/
│   └── default.yaml                  # Main experiment configuration
├── scripts/
│   ├── train_simclr.py               # Train SimCLR encoder checkpoint
│   ├── run_experiments.py            # Run AL experiments over framework/method/budget/seed
│   ├── aggregate_results.py          # Aggregate metrics + export LaTeX-ready tables
│   ├── run_stats.py                  # Paired statistical tests
│   └── make_plots.py                 # Generate report plots
├── src/
│   ├── data.py                       # CIFAR-10 datasets, transforms, subset loaders
│   ├── models.py                     # CIFAR ResNet-18 + SimCLR model components
│   ├── simclr.py                     # Contrastive SimCLR epoch training
│   ├── train_classifier.py           # Supervised classifier training/evaluation
│   ├── typicality.py                 # Typicality and cluster-aware scoring
│   ├── selectors.py                  # TPCRP/baselines/modification selectors
│   ├── clustering.py                 # K-Means / MiniBatchKMeans clustering utilities
│   ├── embeddings.py                 # Embedding extraction helpers
│   ├── evaluate.py                   # Label-distribution summary helpers
│   ├── seed.py                       # Reproducibility utilities
│   ├── config.py                     # YAML configuration loading
│   └── experiment.py                 # Main iterative AL experiment pipeline
├── results/
│   ├── metrics/                      # Raw + aggregated + statistical results
│   └── plots/                        # Generated report figures
├── requirements.txt
├── .gitignore
├── pyproject.toml
├── LICENSE
└── README.md

```

## Method Summary

### TPCRP (TypiClust-style low-budget AL)
TPCRP selects labeled samples in three stages:

1. **Representation Learning (SimCLR)**  
   Learn semantic embeddings from unlabeled CIFAR-10 images using SimCLR.

2. **Clustering for Diversity**  
   Cluster embedding space to spread queries across the data distribution.

3. **Typicality-based Selection**  
   Select high-density (typical) points using inverse average KNN distance (`k=20`).

### Proposed Modification: `tpcrp_ccfl`
`tpcrp_ccfl` keeps the same cluster-based pipeline but refines per-cluster candidates with a lightweight **coverage objective** (facility-location style), improving low-budget selection quality.

## Frameworks

1. **`fully_supervised`**  
   Train CIFAR ResNet-18 from scratch on selected labels only.

2. **`ssl_embedding`**  
   Freeze SimCLR embeddings and train a classifier head on selected labels.

3. **`semi_supervised`**  
   Lightweight semi-supervised approximation over embeddings (LabelSpreading-based).

## Implemented Methods

### Main report methods
- `random`
- `tpcrand`
- `tpcrp`
- `tpcinv`
- `tpcnoclust`
- `tpcrp_ccfl` (modification)

### Secondary baselines
- `kcenter`
- `uncertainty`
- `margin`
- `entropy`
- `dbal`
- `bald`
- `badge`

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## Data and Checkpoints

- CIFAR-10 is downloaded automatically by torchvision.
- SimCLR checkpoint path is configured by:
  - `simclr.save_path` in `configs/default.yaml`
  - default: `results/checkpoints/simclr_resnet18.pt`

Train SimCLR:

```bash
python -m scripts.train_simclr
```

## Run Experiments

Use `configs/default.yaml`
Run full experiment grid configured in YAML (which runs all 3 frameworks and the different baselines including the modified tpcrp):

```bash
python -m scripts.run_experiments
```

This appends rows to:
- `results/metrics/metrics.csv`

Each row key is:
- `(framework,method,budget,seed,rounds,best_epoch,best_test_accuracy,final_test_accuracy,final_test_loss,num_selected)`

Note: Metrics for 3 seeds are already included within results/metrics/metrics.csv.

## Post-Processing

Aggregate results:

```bash
python -m scripts.aggregate_results
```

Run statistical tests:

```bash
python -m scripts.run_stats
```

Generate plots:

```bash
python -m scripts.make_plots
```
Note: Aggregated csv files, stats_summary.csv and plot files from the 3 seeds are also included in the repository. 
## Output Files

### Metrics
- `results/metrics/metrics.csv`
- `results/metrics/aggregated_metrics.csv`
- `results/metrics/aggregated_metrics_main.csv`
- `results/metrics/aggregated_metrics_modification.csv`
- `results/metrics/stats_summary.csv`

### Plots
- `results/plots/ablation_plot_*.png`
- `results/plots/modification_plot_*.png`
- `results/plots/secondary_baselines_plot_*.png`

## Reproducibility Notes

- Seeds are controlled in `configs/default.yaml`.
- Minor numerical variation can still occur across hardware/backends.

## Limitations

- `semi_supervised` is not a full FlexMatch reimplementation.
- `dbal` and `bald` are practical approximations in this coursework pipeline.

