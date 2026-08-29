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
│   ├── default.yaml                  # SimCLR/legacy coursework configuration
│   └── protocol_v2_pilot.yaml        # Locked clean-evaluation pilot
├── scripts/
│   ├── train_simclr.py               # Train SimCLR encoder checkpoint
│   ├── run_experiments.py            # Dry-run/resumable protocol-v2 runner
│   ├── aggregate_results.py          # Canonical artifact-to-CSV reporting
│   ├── run_stats.py                  # Paired bootstrap/t-test inference
│   ├── make_plots.py                 # Framework-separated accuracy plots
│   └── generate_cv_evidence.py       # Traceable machine-readable evidence
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
│   ├── statistics.py                 # Pairing, bootstrap, Holm correction
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

3. **`label_spreading_proxy`**
   A lightweight LabelSpreading baseline over embeddings. The explicit `proxy`
   suffix avoids presenting it as a faithful reproduction of a modern
   semi-supervised learning framework.

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
- `probcover`
- `uncertainty`
- `margin`
- `entropy`
- `dbal_proxy`
- `bald_proxy`
- `badge_proxy`

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -e ".[dev]"
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

Training is permitted only from a clean Git worktree. The command uses the
fixed 100-epoch schedule, atomically writes the final-epoch checkpoint, refuses
to overwrite an existing checkpoint by default, and records a sidecar manifest
containing the code revision, training-config digest, and checkpoint SHA-256.
The resulting checkpoint is frozen at SHA-256
`05a6b66f8c5605f73f6592db57dd5b28c075ce3bbde8e48fcd0a977c0bb2a814` in
both protocol-v2 SimCLR configs. The experiment loader verifies this digest and
the manifest's code, training-config, seed, epoch, and checkpoint provenance
before computing embeddings.

Prepare the second frozen representation. This downloads an exact Git revision
of `facebook/dinov2-small`, verifies the model file against the SHA-256 locked in
`configs/protocol_v2_dinov2.yaml`, and writes a local manifest. Experiments then
load these files locally without network access:

```bash
python -m scripts.prepare_dinov2
```

## Run Experiments

Validate and inspect the locked protocol-v2 pilot without downloading data or
running a model:

```bash
python -m scripts.run_experiments --config configs/protocol_v2_pilot.yaml --dry-run
```

Run one configured job after the representation checkpoint exists:

```bash
python -m scripts.run_experiments --config configs/protocol_v2_pilot.yaml --method tpcrp --seed 42
```

Each run writes exactly one validated artifact under
`results/protocol_v2/runs/<run-id>.json`. The run ID is derived from the complete
effective configuration. Existing artifacts are validated and skipped only when
resume is enabled and their digest matches.

The locked pilot compares `random`, `tpcrp`, iterative `kcenter`, `probcover`,
and three non-duplicate CCFL configurations across five replicate seeds: the
full `tpcrp_ccfl`, candidate-only, and unweighted-refinement variants. The
schema retains `ccfl_weighted` as a compatibility alias for the full method,
but it is deliberately excluded from grids because its parameters are
identical. ProbCover's radius cache is keyed by the complete radius-estimator
inputs, including a digest of the embeddings.

## Bullet-2 confirmation workflow

Do not begin claim-producing runs from an uncommitted worktree. Commit the
reviewed protocol first; every final artifact must record the same clean Git
commit. Then run the predeclared five-seed pilot for the two primary methods:

```bash
python -m scripts.run_experiments --config configs/protocol_v2_pilot.yaml --method tpcrp
python -m scripts.run_experiments --config configs/protocol_v2_pilot.yaml --method tpcrp_ccfl
python -m scripts.check_pilot_gate --root results/protocol_v2
```

Proceed only if Gate B in the experiment specification passes. The SimCLR
confirmation contains 40 jobs (10 seeds times the primary pair and two
ablations); the DINOv2 validation contains 20 jobs (10 seeds times the primary
pair):

```bash
python -m scripts.run_experiments --config configs/protocol_v2_confirmation.yaml
python -m scripts.run_experiments --config configs/protocol_v2_dinov2.yaml
python -m scripts.generate_bullet2_evidence
```

The final command writes `results/bullet2_evidence.json`. The
`cv_metric_x_pp_display` field is the only permitted source for `+X.xpp`: the
mean paired seed-level
accuracy difference `TPCRP-CCFL - TypiClust` at CIFAR-10 budget 10 under the
SimCLR confirmation. DINOv2 is a separate robustness validation and is never
pooled into `X`. Use the bullet only when `claim_ready` is `true`.

The locked source-config SHA-256 digests are:

- pilot: `c9a65de80f68ce02175a94ef65f8f1c0a8f56027f0df0eeae5243ecba5ba8305`
- SimCLR confirmation: `5c0ec92e5bb01ef6c000ff397e11e5510c7d2bc22d8f4b348ddde9514e79d2e8`
- DINOv2 confirmation: `37d8dbc22da725d5c414cc56fad579287b8a63e73fc7ff69ac509dc51ce3a70e`

Any configuration edit changes its digest and constitutes a different
experiment; update the specification before running it.

The evidence command enforces these exact config digests, the predeclared seed
sets (`42`–`46` for Gate B and `42`–`51` for confirmation), cumulative budget
10, a single clean execution environment per grid, and one checkpoint per
representation. The ablation report isolates facility refinement as
`ccfl_unweighted - ccfl_candidate_only` and cluster weighting as
`tpcrp_ccfl - ccfl_unweighted`.

Every round artifact records cumulative representation coverage (mean, p95,
and maximum distance to the selected set), mean selected-pair cosine,
full-representation selected typicality, resolved method parameters, and
selector-only runtime. These selection diagnostics consume train embeddings
only. Label balance is available separately as an explicitly post-hoc reporting
function and cannot enter selection or training.

Training uses a fixed epoch count. The test set is evaluated once after that
schedule; it is not used to select an epoch or checkpoint. Existing legacy
results are not evidence for this cleaned protocol and must not be mixed with
new rows.

## Protocol-v2 Reporting

After the configured runs have completed, generate every report directly from
the validated immutable JSON artifacts:

```bash
python -m scripts.aggregate_results --root results/protocol_v2
python -m scripts.run_stats --root results/protocol_v2
python -m scripts.make_plots --root results/protocol_v2
python -m scripts.generate_cv_evidence --root results/protocol_v2
```

These commands write deterministic round and summary CSVs, paired statistics,
framework-separated plots with `±1 SD` bands, and `cv_evidence.json` under
`results/protocol_v2/reports/`. Reporting validates every input artifact and
fails on duplicate comparison keys or missing method partners; it never silently
deduplicates rows. The primary paired bootstrap uses the predeclared comparison,
seed 20260201, 10,000 resamples, and replicate-level test accuracies. Paired
t-test p-values are secondary and Holm-adjusted across budgets.

`cv_evidence.json` is generated only when the primary budget contains exactly
the configured replicate seeds and at least five valid method pairs. Any future
CV accuracy, coverage, or runtime wording must be populated from that file and
its recorded run IDs—not copied from a plot, notebook, or historical CSV.

To verify the reporting implementation without running CIFAR-10 training:

```bash
python -m pytest tests/test_statistics.py tests/test_reporting_pipeline.py
```

## Protocol-v2 Output

- `results/protocol_v2/runs/<run-id>.json`: immutable validated run artifacts
- `results/protocol_v2/cache/embeddings/`: checkpoint-digest-keyed embedding cache
- `results/protocol_v2/checkpoints/`: round model checkpoints when applicable
- `results/protocol_v2/reports/round_metrics.csv`: one row per model fit
- `results/protocol_v2/reports/summary_metrics.csv`: framework-separated means/SDs
- `results/protocol_v2/reports/paired_statistics.csv`: paired inference by budget
- `results/protocol_v2/reports/cv_evidence.json`: traceable CV evidence contract
- `results/protocol_v2/reports/plots/`: framework-separated accuracy figures

> **Legacy-results warning:** historical results under `results/metrics/` and
> `results/plots/` used test accuracy to select the best training checkpoint.
> That test-selected protocol is biased and those files must never be presented,
> pooled, or regenerated as protocol-v2 evidence.

## Reproducibility Notes

- Component seeds and DataLoader generators are derived deterministically from
  the replicate, framework, method family, and active-learning round. The
  TypiClust/CCFL ablation family shares selector randomness to isolate component
  effects; unrelated selectors retain independent streams.
- Strict deterministic PyTorch operations are enabled. Unsupported operations
  fail rather than silently becoming nondeterministic.
- The experiment launcher sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before loading
  PyTorch so strict deterministic CUDA matrix operations are configured.
- Bitwise repeatability is guaranteed only on the same recorded platform,
  device, package versions, code revision, and representation checkpoint.
- Protocol-v2 implementation is phased. Do not claim new benchmark results
  until every protocol gate in `specs/001-clean-active-learning-evaluation/`
  passes and fresh runs have been generated.

## Limitations

- `label_spreading_proxy` is not a full FlexMatch reimplementation.
- `dbal_proxy`, `bald_proxy`, and `badge_proxy` are explicitly noncanonical
  approximations and are excluded from protocol-v2 benchmark claims.
- Phase 7 validates reporting logic on synthetic artifacts; it does not produce
  a CIFAR-10 result or authorize a performance claim.
- Statistical validation requires at least five complete paired replicates at
  the predeclared primary budget. Missing runs are errors, not exclusions.
- Accuracy plots describe within-method replicate variability (`±1 SD`); they
  are not paired confidence intervals and never pool evaluation frameworks.
- Selector runtime is environment-dependent and is meaningful only with the
  recorded hardware, software versions, checkpoint, and code revision.
