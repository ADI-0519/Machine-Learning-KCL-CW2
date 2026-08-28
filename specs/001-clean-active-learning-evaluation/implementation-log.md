# Implementation Log

## 2026-08-27 — Phase 0 baseline

- Git worktree already contained user-owned deletions under `results/metrics/`
  and `results/plots/`; implementation must not restore or alter them.
- Baseline commit: `b3713066c1aec0d0dfcce5cef1463b3087a07ba4`
- Python: 3.11.5
- PyTorch: 2.8.0+cpu
- CUDA runtime: unavailable
- scikit-learn: 1.6.1
- NumPy: 2.2.6
- `pytest`, `pytest-cov`, and `ruff`: available in the active environment
- SimCLR checkpoint: absent
- CIFAR-10 train embedding cache: absent
- CIFAR-10 test embedding cache: absent

Consequently, Phases 0–2 are verified with CPU-only synthetic fixtures. No
CIFAR result generated in this environment is claimed or inferred.

The committed baseline source parsed successfully. A regression audit against
that immutable commit confirmed both leakage signatures before replacement:

- supervised evaluation called `evaluate_classifier` from the epoch loop;
- frozen-embedding evaluation selected state under `if acc > best_acc`.

## 2026-08-27 — Phases 1 and 2 complete

- Added SHA-256-derived component seeds with fixed regression vectors.
- Seeded every current DataLoader call site explicitly, including workers.
- Changed supervised and frozen-embedding training to fixed epochs followed by
  exactly one test evaluation.
- Normalized evaluator returns to `TrainingOutcome` and parameterized the
  frozen/proxy class count.
- Renamed the public LabelSpreading framework to `label_spreading_proxy`.
- Migrated result consumers to `test_accuracy` and banned the two legacy metric
  names under `src/`, `scripts/`, and `configs/`.
- Verification: `11 passed`; touched-file Ruff check passed; AST parsing passed;
  the legacy-name search produced no matches.

No dataset was downloaded and no experiment result was written. Phase 3
(explicit nested acquisition trajectories) is intentionally not started.

## 2026-08-27 — Phase 3 complete

- The project-local `.venv` now uses PyTorch `2.8.0+cu128`; CUDA `12.8` sees an
  NVIDIA GeForce RTX 3060 Laptop GPU.
- Replaced implicit total-budget splitting with validated
  `selection.round_query_sizes`.
- Each `(framework, method, seed)` now executes one nested trajectory instead
  of independent runs for each total budget.
- Selector results are validated and normalized to global training indices at
  the experiment boundary. Cluster-based selection uses the recorded shared
  clustering seed directly.
- Ordered acquisition history is stored separately from set membership. Every
  round records its query, cumulative selection, component seeds, evaluation,
  and an explicit diagnostics placeholder for Phase 6.
- Selection uses a fresh method-specific RNG; training is reseeded with the
  shared training seed immediately before each evaluator call; the round
  DataLoader seed is passed independently.
- Added the locked two-method `configs/protocol_v2_pilot.yaml` configuration.
- Verification: `32 passed`; `src.protocol` statement coverage is `95%`; full
  Ruff lint and format checks passed; AST parsing and legacy metric guards
  passed.

The synthetic checkpoint produced cumulative budgets 10 and 20, preserved the
round-1 sequence as the round-2 prefix, selected 20 unique indices, and invoked
the evaluator exactly twice. The SimCLR checkpoint and embedding caches remain
absent, so no CIFAR experiment was launched and no accuracy claim was produced.

## 2026-08-27 — Phase 4 / Gate A complete

- Added canonical JSON configuration hashing and deterministic protocol-scoped
  run IDs.
- Added a strict full-schema validator covering root metadata, environment,
  ordered nested rounds, component seeds, finite metrics, timing alignment,
  checkpoint provenance, and legacy-field rejection.
- Added atomic JSON replacement with cleanup guarantees; a simulated replacement
  failure preserved the previously complete artifact.
- `run_single_experiment` now builds one in-memory artifact and writes it only
  after every round succeeds. Protocol-v2 no longer appends legacy metrics CSVs
  or emits mutable selection side files.
- Protocol output is isolated under `results/protocol_v2`; embedding cache names
  include the representation checkpoint digest.
- The effective config now explicitly includes representation, optimizer,
  regularization, and CCFL parameters discovered to be missing from the initial
  draft. The spec and pilot were updated together before any run.
- Added strict unknown-key configuration validation and a CLI with `--config`,
  `--method`, `--seed`, and `--dry-run`.
- Resume validates an existing artifact and skips it only when its requested
  digest matches. Malformed artifacts fail visibly. A three-run synthetic test
  with two existing runs executed seed 43 only.
- Provenance records the effective config, Git commit and dirty flag, Python and
  dependency versions, OS platform, device, CUDA runtime, strict deterministic
  state, cuBLAS workspace setting, and checkpoint path/SHA-256.
- Verification: `70 passed`; combined `src.protocol`/`src.artifacts` statement
  coverage is `91%`; full Ruff lint/format, AST parse, static leakage guard, and
  `git diff --check` passed.
- Pilot dry-run planned ten deterministic TypiClust/CCFL jobs and created no
  output directory. Two fresh synthetic executions in separate Python processes
  matched after removing only measured timings.

The SimCLR checkpoint remains absent, so Gate A contains no CIFAR accuracy and
does not authorize a CV performance claim. Phase 5 baseline-fidelity work is the
next implementation stage before any pilot execution.

## 2026-08-27 — Phase 5 / baseline-fidelity gate complete

- Replaced the pool-only k-center approximation with iterative, global-index
  CoreSet selection conditioned on every previously labeled embedding. Anchor
  distances are accumulated without allocating a pool-by-labeled-by-feature
  tensor, preserving reference outputs while bounding memory use.
- Implemented an inclusive decimal radius grid, K-Means pseudo-label purity,
  largest-valid-radius selection, and deterministic pool-only greedy ProbCover.
  The dataset class count is now an explicit hashed configuration field.
- ProbCover estimates its radius once per representation/configuration and
  stores it in an atomic cache keyed by all estimator inputs, including the
  embedding-array digest. The chosen radius, seed, and cache digest are recorded
  in run diagnostics.
- Pinned the official compatibility fixture to TypiClust repository commit
  `4097a71c348f60492ab22be0c4c9da224e637af6`. The test compares both selected
  global indices and cumulative coverage with a small CPU port of the official
  greedy graph procedure. Exact-radius boundary semantics are tested separately.
- Renamed the noncanonical public method IDs to `dbal_proxy`, `bald_proxy`, and
  `badge_proxy`; none are admitted by the protocol-v2 config validator or pilot.
- Expanded the locked pilot to Random, TypiClust, iterative CoreSet, ProbCover,
  and CCFL over five replicate seeds. Dry-run planned 25 deterministic artifacts
  and did not create `results/protocol_v2`.
- Verification: `88 passed`; full Ruff lint/format and `git diff --check` passed;
  the official fixture, selector validation, radius-cache reuse, and strict
  proxy-name guards are covered by tests.

No CIFAR run was launched and no accuracy or performance claim was produced.
The SimCLR checkpoint and embedding caches remain absent. Phase 6 diagnostics
and CCFL ablations are the next gate before the pilot is executed.

## 2026-08-27 — Phase 6 / CCFL diagnostics gate complete

- Added exact cumulative selection diagnostics over the remaining unlabeled
  train representation: nearest-selected mean/p95/maximum distance, selected
  pairwise cosine, and selected-point full-representation typicality.
- Timed selector calls with a monotonic high-resolution clock and stored the
  aligned durations separately from deterministic diagnostics.
- Added explicit candidate-only, unweighted-refinement, weighted-refinement,
  and full CCFL configurations. Scientific CCFL parameters have no source-code
  defaults, and semantic variant constraints are validated before execution.
- Added resolved CCFL and ProbCover method metadata to every immutable round
  artifact and strengthened numeric, provenance, and nested-trajectory schema
  validation.
- Added reporting-only post-hoc class-balance counts, proportions, observed
  coverage, and normalized entropy, plus static guards preventing label-aware
  reporting from entering selection or training modules.
- Removed the unreachable legacy proxy execution branch from the protocol-v2
  runner. Standalone legacy ablation selectors remain available outside the
  locked benchmark.
- Audited every Python and configuration file; fixed strict-CUDA launcher
  setup, hidden SSL fallbacks, malformed embedding-cache acceptance, empty
  loader handling, assert-only invariants, and input-validation gaps.
- Verification: `142 passed`; statement coverage is `92%` for artifacts,
  `100%` for diagnostics, `95%` for protocol, and `100%` for reporting. Ruff's
  configured checks and the broader E/F/I/B/UP/SIM/PERF/PIE/RUF rules, AST
  parsing, imports, dependency checks, YAML parsing, static leakage guards, and
  `git diff --check` passed. Dry-run planned 40 jobs and created no output.

CUDA is available (`torch 2.8.0+cu128`), but the SimCLR checkpoint remains
absent. No CIFAR benchmark was launched and no accuracy or CV performance claim
was produced. Phase 7 artifact aggregation and statistical evidence is next.

## 2026-08-27 — Phase 7 / reporting and evidence gate complete

- Replaced legacy CSV consumers with a fail-closed canonical-artifact loader.
  It validates artifact schemas, source configurations, filenames/run IDs, run
  dimensions, source-config consistency, and duplicate logical result keys.
- Added deterministic round-level and framework-separated summary CSVs. No row
  is silently deduplicated and no dataset or evaluation framework is pooled.
- Added exact one-to-one replicate pairing, a fixed-seed 10,000-resample paired
  percentile bootstrap, paired t-tests by cumulative budget, and one Holm
  correction across the configured comparison.
- Replaced pooled legacy figures with one plot per dataset/framework using
  explicitly labeled replicate `±1 SD` bands. Plotting rejects groups with
  fewer than two replicates rather than depicting undefined variation as zero.
- Added `cv_evidence.json` generation for the predeclared primary budget. It
  requires at least five complete pairs with exactly the configured seed set,
  records the exact primary-comparison run IDs, and computes accuracy, paired
  interval, coverage improvement, and selector-runtime ratio from artifacts.
- Documented the exact reporting commands, evidence limitations, and the
  prominent prohibition on using historical test-selected checkpoints.
- Verification: `171 passed`; `src.statistics` statement coverage is `94%`;
  configured and broad Ruff checks, formatting, compilation, imports,
  dependency checks, YAML parsing, CLI help smoke tests, static legacy-metric
  guards, and `git diff --check` passed. The deterministic synthetic reporting
  fixture regenerated byte-identical CSVs and checked exact expected evidence.
  Dry-run planned 40 jobs and created no output directory.

The CUDA environment remains available, but the SimCLR checkpoint is absent.
No CIFAR training was launched, no protocol-v2 report was created in `results/`,
and no empirical or CV performance claim is authorized. Phase 8 begins by
creating or supplying the locked representation checkpoint, then executing the
pilot artifacts before any evidence generator is run on real results.

## 2026-08-28 — Phase 8 pre-run hardening

- Made artifact validation cross-check the complete validated source config,
  configured round schedule, fixed epoch count, recomputed component seeds,
  CIFAR train-index bounds, representation path, and DINOv2 digest.
- Made Gate B require the exact frozen pilot config, seeds 42–46, budget 10,
  and one identical clean execution environment and checkpoint across all
  primary-pair artifacts.
- Locked final bullet evidence to the exact SimCLR/DINOv2 source-config
  digests, seeds 42–51, CIFAR-10 budget 10, and uniform per-grid provenance.
- Corrected component analysis to compare unweighted refinement against the
  candidate-only control, and full CCFL against unweighted refinement.
- Shared selector randomness across TypiClust and all CCFL variants so cluster
  tie-breaking and small-cluster fallbacks cannot confound component contrasts;
  unrelated method families retain independent selector streams.
- Changed SimCLR checkpointing from best observed training loss to the fixed
  final epoch. Writes are atomic, refuse accidental overwrite, require a clean
  Git worktree, and emit a manifest with Git/config/checkpoint digests.
- Reduced the Gate-B pilot acquisition schedule to its declared budgets 10 and
  20; confirmation remains the five-budget trajectory.
- Added synthetic checkpoint-training and fail-closed provenance regression
  tests. No CIFAR data was downloaded and no representation or benchmark result
  was produced.
- Verification: `207 passed`; overall statement coverage increased to `81%`,
  with `scripts.train_simclr` and `src.simclr` both at `90%+`. Configured and
  broad Ruff checks, Ruff formatting, compilation, dependency checks, three
  config dry-runs, CUDA SimCLR forward/backward, and `git diff --check` passed.

The remaining pre-pilot action is T054C: create the real SimCLR checkpoint,
insert and enforce its SHA-256, then update the two affected frozen config
digests before committing the final experiment definition.
