# Feature Specification: Deterministic Active-Learning Evaluation

**Feature ID:** `001-clean-active-learning-evaluation`  
**Status:** Implemented through pre-run hardening; empirical gates pending
**Created:** 2026-08-27  
**Scope owner:** Repository maintainer

The key words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** in
this document are to be interpreted as described in RFC 2119.

## 1. Objective

Replace the current test-selected, append-only experiment pipeline with a
deterministic and auditable protocol that can support defensible CV claims
about TypiClust and TPCRP-CCFL.

The completed system must answer four questions without manual reconstruction:

1. Which exact examples were selected at every active-learning round?
2. Which exact configuration, code revision, representation, and seeds produced
   each score?
3. Was the CIFAR test set used only for reporting, never for model or
   hyperparameter selection?
4. Does CCFL improve accuracy, representation coverage, and/or runtime against
   faithful baselines across paired random seeds?

## 2. Non-goals

The claim-producing benchmark MUST NOT include:

- FlexMatch or another expensive semi-supervised trainer.
- DCoM or Uncertainty Herding in the locked primary comparison.
- Distributed multi-node execution.
- A claim of state-of-the-art performance.
- Migration or reinterpretation of the existing `best_test_accuracy` results as
  clean protocol results.

FlexMatch and recent selection baselines may be separate follow-up features
after the deterministic CIFAR-10 frozen-embedding benchmark passes its decision
gate. DINOv2 is admitted only as a separately reported representation-robustness
validation and MUST NOT be pooled into the primary SimCLR estimate.

## 3. Definitions

- **Selection seed:** Controls stochastic query selection only.
- **Clustering seed:** Controls K-Means/MiniBatchKMeans and is shared across
  TypiClust-family methods in a paired comparison.
- **Training seed:** Controls model initialization, augmentation order, and
  optimizer randomness and is shared across methods in a paired comparison.
- **Run:** One dataset, framework, method, replicate seed, and acquisition
  schedule.
- **Round:** One query operation followed by one model fit and one test-set
  report.
- **Acquisition schedule:** Explicit query sizes per round, for example
  `[10, 10, 10, 10, 10]`.
- **Cumulative budget:** Total number of labeled examples after a round.
- **Protocol v2:** The clean protocol defined by this specification.
- **Legacy result:** Any result containing `best_test_accuracy`, produced by the
  pre-v2 code, or lacking `protocol_version`.

## 4. User stories and independent tests

### US1 — Test-isolated model evaluation (P1)

As an experiment author, I need training to be independent of test performance
so that reported accuracy estimates are not optimistically selected.

**Independent test:** Instrument the evaluator and demonstrate that a run with
`R` active-learning rounds invokes test evaluation exactly `R` times, after
training in each round, while training history contains no test metrics.

**Acceptance scenarios:**

1. Given a three-epoch supervised run, when training completes, then the test
   evaluator is called exactly once.
2. Given a three-epoch frozen-embedding run, when training completes, then
   `accuracy_score` and `log_loss` are each called exactly once on test data.
3. Given a completed result, when it is serialized, then it contains
   `test_accuracy` and does not contain `best_test_accuracy`.

### US2 — Repeatable paired experiments (P1)

As a researcher, I need repeated executions on the same environment to produce
the same selections and metrics so that differences between methods are
auditable.

**Independent test:** Run the synthetic integration protocol twice in separate
processes and compare canonical result JSON after excluding timestamps and
durations.

**Acceptance scenarios:**

1. Given identical configuration, code, inputs, and environment, two executions
   produce identical selected indices and scalar metrics within `1e-8` on CPU.
2. Given TypiClust and CCFL with the same replicate, both receive the same
   clustering and training seeds for a round.
3. Given different method families, family-specific selector randomness does
   not advance or alter the shared clustering/training RNG streams. TypiClust
   and its CCFL component ablations deliberately share selector randomness so
   their contrasts isolate algorithmic components rather than tie-breaking.

### US3 — Immutable, resumable result artifacts (P1)

As a maintainer, I need every run to be independently persisted and validated so
that an interrupted grid can resume without duplicate or mixed-protocol rows.

**Independent test:** Interrupt a three-run synthetic grid after two artifacts,
resume it, and verify that only the missing run executes and all three artifacts
validate against the schema.

**Acceptance scenarios:**

1. A run writes one atomic JSON artifact under a deterministic run ID.
2. A complete matching artifact is skipped on resume.
3. A malformed, legacy, or configuration-mismatched artifact causes a visible
   failure; it is never silently deduplicated.
4. Aggregation reads protocol-v2 artifacts only.

### US4 — Faithful low-budget baselines (P2)

As a reviewer, I need baseline names to correspond to their canonical algorithms
so that comparisons are meaningful.

**Independent test:** Validate Random, TypiClust, iterative CoreSet, and
ProbCover on deterministic synthetic geometries with known invariants.

**Acceptance scenarios:**

1. Iterative CoreSet computes distance from every pool point to the union of
   existing labeled points and new batch selections.
2. No selector returns an already-labeled index, duplicate index, or the wrong
   query count.
3. Framework-specific proxies are not exposed under canonical method names.
4. ProbCover's implementation and radius policy are documented and checked
   against the official implementation on a fixed fixture before benchmark use.

### US5 — Explainable CCFL evaluation (P2)

As an ML interviewer or reviewer, I need evidence for why CCFL works, not only a
single accuracy difference.

**Independent test:** On a synthetic clustered dataset, calculate accuracy-
independent selection diagnostics for TypiClust and every CCFL ablation.

**Acceptance scenarios:**

1. The benchmark includes candidate-only, unweighted facility-location,
   weighted facility-location, and full configured CCFL variants.
2. Every selection artifact records mean, p95, and maximum nearest-selected
   distance over the remaining unlabeled representation after that round; mean
   pairwise selected cosine similarity; mean selected typicality; and
   selector-only runtime.
3. Diagnostics use no labels except the explicitly named post-hoc class-balance
   analysis.
4. The post-hoc class-balance metric can never affect selection, training,
   checkpointing, or the decision gate.

### US6 — Statistically defensible comparison (P2)

As a CV author, I need one predeclared comparison whose result cannot be selected
after observing many budgets and methods.

**Independent test:** Feed a fixed paired fixture into the statistics command and
match expected paired differences, bootstrap interval, and Holm-adjusted p-values.

**Acceptance scenarios:**

1. Pairing keys are dataset, framework, cumulative budget, and replicate seed.
2. The primary comparison is declared in configuration before execution.
3. The primary pilot uses five seeds; the confirmation uses ten seeds.
4. Confidence intervals are computed on paired differences using a fixed
   bootstrap seed and at least 10,000 resamples.
5. Multiple reported hypothesis tests receive Holm correction.
6. No result with fewer than five valid pairs is described as statistically
   validated.

### US7 — Cross-dataset confirmation (P3)

As a recruiter or reviewer, I need to know whether the result generalizes beyond
CIFAR-10.

**Independent test:** Run the complete data/embedding/selection smoke pipeline
on small deterministic subsets of both CIFAR-10 and CIFAR-100.

**Acceptance scenarios:**

1. Dataset-specific class counts and normalization are read from a registry,
   not hard-coded as `10`.
2. Embedding cache keys include dataset, split, representation configuration,
   checkpoint digest, and protocol version.
3. CIFAR-100 confirmation starts only after the CIFAR-10 confirmation gate passes.

## 5. Functional requirements

### Evaluation isolation

- **FR-001:** Neural evaluators MUST train for a fixed, configuration-declared
  number of epochs.
- **FR-002:** Test data MUST NOT influence epoch count, checkpoint choice,
  hyperparameters, acquisition, or experiment continuation.
- **FR-003:** Test evaluation MUST occur exactly once after training in each
  active-learning round.
- **FR-004:** Training history MUST contain training metrics only.
- **FR-005:** Protocol-v2 result objects MUST use `test_accuracy` and
  `test_loss`; legacy `best_*` and `final_*` metric aliases MUST NOT be emitted.

### Active-learning trajectory

- **FR-006:** Acquisition MUST be configured as explicit per-round query sizes.
- **FR-007:** Selected sets MUST be nested across rounds.
- **FR-008:** Every round artifact MUST record `query_size`,
  `cumulative_budget`, and ordered selected indices.
- **FR-009:** Model weights MUST be reinitialized each round for the supervised
  and frozen-linear protocols.

### Determinism

- **FR-010:** Python, NumPy, PyTorch CPU, PyTorch CUDA, DataLoader, clustering,
  selection, and training RNGs MUST be explicitly controlled.
- **FR-011:** Component seeds MUST be derived with a stable cryptographic hash,
  never Python's process-randomized `hash()`.
- **FR-012:** Shared paired components MUST receive method-independent seeds.
- **FR-013:** Stochastic operations MUST receive a separate selector seed.
  TypiClust, full CCFL, and CCFL component ablations MUST share one selector
  seed within a paired replicate; unrelated method families MUST use distinct
  selector seeds.
- **FR-014:** Strict mode MUST enable PyTorch deterministic algorithms and
  disable cuDNN benchmarking.
- **FR-015:** Documentation MUST state that bitwise reproducibility is guaranteed
  only for the same recorded platform, device, dependency versions, and code.

### Artifacts and schema

- **FR-016:** Protocol-v2 outputs MUST live under `results/protocol_v2/`.
- **FR-017:** Each run MUST have a deterministic ID derived from canonicalized
  effective configuration and protocol version.
- **FR-018:** Each run MUST write atomically to one JSON file.
- **FR-019:** Every artifact MUST include effective config, protocol version,
  git commit, dirty-tree flag, dependency versions, device description,
  checkpoint SHA-256, seed bundle, round records, and timings.
- **FR-020:** Aggregation MUST reject artifacts with missing fields, non-finite
  metrics, mixed protocol versions, duplicate run IDs, or mismatched config hashes.
- **FR-021:** Legacy CSV files MUST remain read-only historical evidence and MUST
  NOT be mixed into protocol-v2 outputs.

### Selectors and diagnostics

- **FR-022:** All selectors MUST return exactly `query_size` unique global train
  indices from the unlabeled pool or raise a descriptive exception.
- **FR-023:** CoreSet MUST condition on existing labeled indices.
- **FR-024:** Proxy methods MUST use names containing `_proxy` and MUST be
  excluded from the primary benchmark.
- **FR-025:** Selection diagnostics MUST be computed from train embeddings only.
- **FR-026:** Selector runtime MUST use a monotonic high-resolution clock and
  exclude embedding loading and model training.

### Statistics and reporting

- **FR-027:** The primary endpoint MUST be declared before confirmation runs.
- **FR-028:** Summary tables MUST report mean accuracy, standard deviation,
  sample count, paired mean difference, and 95% paired confidence interval.
- **FR-029:** Statistical inference MUST operate on replicate-level values, not
  aggregated means.
- **FR-030:** Frameworks MUST NOT be pooled into a single accuracy mean.
- **FR-031:** Generated plots MUST label whether bands represent standard
  deviation, standard error, or confidence intervals.

Protocol-v2 reporting uses the following locked conventions:

- `reports/round_metrics.csv` contains one row per validated artifact round,
  sorted by dataset, framework, method, replicate seed, and cumulative budget.
- `reports/summary_metrics.csv` groups only by dataset, framework, method, and
  cumulative budget and records mean accuracy, sample standard deviation, and
  replicate count. Frameworks and datasets are never pooled.
- Pairing keys are dataset, framework, cumulative budget, and replicate seed.
  Duplicate method/key rows or a missing partner are fatal errors.
- Every paired difference is `method_a - method_b`. The paired bootstrap uses
  seed `20260201`, 10,000 resamples, and a 95% percentile interval.
- Budget-level paired t-test p-values are secondary analyses and receive one
  Holm correction across all budgets in the configured method comparison.
- Accuracy plots use replicate standard-deviation bands and state `±1 SD` in
  their y-axis/legend text. A plot contains one dataset/framework only.
- Coverage improvement is
  `(mean_distance_b - mean_distance_a) / mean_distance_b * 100`, so a positive
  value means method A reduced nearest-selected distance. A zero method-B
  denominator is rejected rather than assigned an arbitrary percentage.
- Selector runtime ratio is `mean_runtime_a / mean_runtime_b`; a zero method-B
  denominator is rejected.
- `cv_evidence.json` contains the run IDs used for traceability in addition to
  the required numeric fields in Section 10. Evidence generation never reads
  legacy CSV files.

## 6. Configuration contract

Protocol-v2 uses the following top-level structure:

```yaml
protocol:
  version: "2.0"
  deterministic: true
  test_evaluations_per_round: 1

output:
  root: "./results/protocol_v2"
  resume: true

data:
  name: "cifar10"
  root: "./data"
  num_workers: 2
  num_classes: 10

representation:
  backend: "simclr"
  checkpoint_path: "./results/checkpoints/simclr_resnet18.pt"
  weights_sha256: "<64-character SHA-256 frozen before experiments>"
  projection_dim: 128
  batch_size: 256
  embedding_seed: 21

selection:
  round_query_sizes: [10, 10, 10, 10, 10]
  knn_k: 20
  max_clusters: 500
  min_cluster_size: 5

evaluation:
  framework: "ssl_embedding"
  epochs: 60
  batch_size: 64
  lr: 2.5
  momentum: 0.9
  weight_decay: 0.0005
  dropout_p: 0.2
  test_policy: "once_after_fixed_epochs"

experiment:
  methods: ["random", "tpcrp", "kcenter", "probcover", "tpcrp_ccfl"]
  replicate_seeds: [42, 43, 44, 45, 46]
  primary_comparison:
    method_a: "tpcrp_ccfl"
    method_b: "tpcrp"
    cumulative_budget: 10
    metric: "test_accuracy"

ccfl_variants:
  tpcrp_ccfl:
    candidates_per_cluster: 5
    refine_steps: 1
    use_cluster_weights: true
  ccfl_candidate_only:
    candidates_per_cluster: 5
    refine_steps: 0
    use_cluster_weights: false
  ccfl_unweighted:
    candidates_per_cluster: 5
    refine_steps: 1
    use_cluster_weights: false
  ccfl_weighted:
    candidates_per_cluster: 5
    refine_steps: 1
    use_cluster_weights: true

probcover:
  alpha: 0.95
  delta_search:
    minimum: 0.05
    maximum: 1.50
    step: 0.01
```

Representation, optimizer, regularization, and CCFL settings are part of the
configuration digest because changing any of them can change the scientific
result. They MUST NOT be supplied by source-code fallbacks.

Unknown keys MUST cause configuration validation to fail. Defaults that alter
scientific behavior MUST NOT be hidden in source code.

The ProbCover radius MUST be the largest configured candidate whose pseudo-label
ball purity is at least `alpha`. Pseudo-labels MUST come from K-Means over train
embeddings with `n_clusters` equal to the dataset class count and the recorded
clustering seed. Radius candidates MUST be generated with decimal arithmetic,
including both endpoints when aligned, so binary floating-point stepping cannot
change the candidate list.

Selection diagnostics MUST receive the remaining unlabeled global indices
explicitly; they MUST NOT infer this set from labels. Selector-only durations
MUST be stored in `timings.selector_seconds`, aligned one-to-one with rounds,
and MUST NOT be mixed into deterministic `selection_diagnostics` values.
Selected typicality MUST be computed exactly for the cumulative selected points
against the full train representation; implementations MUST NOT perform an
unnecessary all-points k-NN query when only selected-point scores are reported.

Post-hoc class balance MUST be reporting-only and return class counts,
proportions, observed-class coverage, and normalized entropy. It MUST accept
labels explicitly and MUST NOT be imported by selection code.

## 7. Code-quality requirements

- **QR-001:** New and changed public functions MUST have type annotations and
  docstrings that describe indices as global or pool-local.
- **QR-002:** `ruff check`, `ruff format --check`, the full pytest suite, and the
  protocol static guards MUST pass before a phase is complete.
- **QR-003:** Tests MUST cover at least 90% of statements in new protocol,
  artifact, diagnostic, and statistical modules.
- **QR-004:** Tests MUST NOT depend on network access, CIFAR downloads, CUDA, or
  existing result/checkpoint files.
- **QR-005:** Scientific defaults MUST be resolved by configuration validation;
  production functions MUST NOT contain behavior-changing fallback constants.
- **QR-006:** Exceptions MUST identify the run, method, round, and invalid value
  when that context is available.
- **QR-007:** Implementations MUST use existing project dependencies where they
  are sufficient and MUST justify every added runtime dependency.

## 8. Success criteria

- **SC-001:** All protocol-v2 tests pass on CPU from a clean checkout.
- **SC-002:** Static search finds no per-epoch test evaluation and no emitted
  `best_test_accuracy` in protocol-v2 code.
- **SC-003:** The synthetic end-to-end run is repeatable across two fresh
  processes in the same environment.
- **SC-004:** An interrupted grid resumes without duplicate execution.
- **SC-005:** Five-seed pilot results contain five valid paired observations for
  every declared primary endpoint.
- **SC-006:** Confirmation work begins only if the pilot gate in Section 9 passes.
- **SC-007:** A generated evidence summary can populate every number in the
  target CV bullets without manual arithmetic.

## 9. Deterministic experiment gates

### Gate A — Protocol integrity

Pass only when SC-001 through SC-004 are satisfied. No GPU benchmark result
produced before this gate may be used in the new report or CV.

### Gate B — Five-seed CIFAR-10 pilot

Run frozen-embedding TypiClust and CCFL through the predeclared cumulative
budgets 10 and 20. The decision is made only at the primary budget of 10;
budget 20 is descriptive and cannot change the gate outcome.
Proceed when all conditions hold at the predeclared primary budget:

- Five valid paired seeds.
- Positive mean paired CCFL improvement.
- CCFL wins on at least four of five seeds.
- Mean improvement is at least 0.03 absolute accuracy (3 percentage points).
- Mean nearest-selected distance improves, or the facility objective improves,
  without more than 2x selector runtime.

Failure does not authorize changing the endpoint. Investigate and record the
failure; any revised hypothesis requires a new protocol version.

### Gate C — Ten-seed CIFAR-10 confirmation

Run the locked implementation and configuration with ten seeds. Proceed to
CIFAR-100 when:

- The paired 95% bootstrap interval does not cross zero.
- The improvement remains at least 0.02 absolute accuracy.
- No run is missing or excluded without a documented infrastructure failure.

### Gate D — CIFAR-100 confirmation

Use the same method implementation and analysis code. Dataset-specific budgets
may scale with the number of classes, but this policy MUST be declared before
execution. Cross-dataset claims require positive mean paired improvement on both
datasets; they do not require statistical significance on each dataset.

## 10. CV evidence contract

The evidence generator MUST output a machine-readable `cv_evidence.json` with:

```json
{
  "protocol_version": "2.0",
  "total_runs": 0,
  "total_model_fits": 0,
  "datasets": [],
  "methods": [],
  "replicate_count": 0,
  "primary_budget": 0,
  "ccfl_mean_accuracy": 0.0,
  "tpcrp_mean_accuracy": 0.0,
  "paired_improvement_pp": 0.0,
  "paired_ci95_pp": [0.0, 0.0],
  "coverage_improvement_percent": 0.0,
  "selector_runtime_ratio": 0.0
}
```

CV wording MUST use values from this artifact. It MUST NOT use values copied
from notebooks, plots, legacy CSVs, or manually selected runs.

## 11. Assumptions

- The existing SimCLR checkpoint may be reused for the initial CIFAR-10 pilot
  if its digest and training configuration are recorded.
- Frozen-embedding evaluation is the MVP because it is inexpensive enough for
  five- and ten-seed paired studies.
- Full paper-protocol SimCLR retraining and fully supervised confirmation are
  later evidence upgrades, not blockers for protocol-v2 implementation.
- Existing user-deleted result files will not be restored or modified by this
  feature.
