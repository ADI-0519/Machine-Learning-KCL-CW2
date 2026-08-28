# Tasks: Deterministic Active-Learning Evaluation

Every task is intentionally small and independently reviewable. Tests precede
production code. A phase is complete only after its checkpoint command passes.

## Dependency order

```text
Phase 1 (protocol types)
  -> Phase 2 (test isolation)
  -> Phase 3 (nested trajectory)
  -> Phase 4 (artifacts/resume)
  -> Gate A
  -> Phase 5 (baseline fidelity)
  -> Phase 6 (CCFL diagnostics/ablations)
  -> Phase 7 (statistics/evidence)
  -> Gate B pilot
  -> Gate C confirmation
  -> Phase 8 (CIFAR-100)
  -> Gate D
```

## Phase 0 — Safety baseline

- [x] **T001** Record `git status --short`, the current commit, Python version,
  PyTorch version, CUDA version, and whether the existing SimCLR checkpoint and
  cached embeddings exist; do not restore deleted legacy artifacts.
- [x] **T002** Add `pytest`, `pytest-cov`, and `ruff` as development dependencies
  in `pyproject.toml`; configure them there and document
  `pip install -e ".[dev]"` in `README.md`.
- [x] **T003** Add `tests/conftest.py` with synthetic embeddings, labels, and
  tiny in-memory datasets; fixtures MUST NOT download data or require CUDA.
- [x] **T004** Run the unmodified source through an AST parse and record the
  expected legacy leakage tests as failing before implementation.

**Checkpoint:** `python -m pytest -q` executes the new test harness; explicitly
marked regression tests fail for the expected legacy behavior only.

## Phase 1 — Protocol types and seeds

- [x] **T005** Add fixed-output tests for `derive_seed` and `SeedBundle.for_round`
  in `tests/test_determinism.py`.
- [x] **T006** Add tests proving clustering/training seeds are equal across
  `tpcrp` and `tpcrp_ccfl`, while selector seeds differ.
- [x] **T007** Implement `PROTOCOL_VERSION`, `TEST_POLICY`, `derive_seed`,
  `SeedBundle`, `EvaluationMetrics`, and `TrainingOutcome` in `src/protocol.py`.
- [x] **T008** Add repeatable worker/generator tests in
  `tests/test_determinism.py`.
- [x] **T009** Implement strict `set_seed`, `seed_worker`, and `make_generator`
  in `src/seed.py`.
- [x] **T010** Update `make_subset_loader` in `src/data.py` to require a seed and
  update all call sites with a temporary explicit seed.

**Checkpoint:**

```powershell
python -m pytest -q tests/test_determinism.py
python -B -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('src').glob('*.py')]"
```

## Phase 2 — Test-isolated evaluators

- [x] **T011** Add a supervised evaluator test in
  `tests/test_evaluation_protocol.py` that monkeypatches `CIFARClassifier` with a
  tiny model and counts calls to `evaluate_classifier`.
- [x] **T012** Add a frozen-embedding test that counts test calls and asserts
  test keys are absent from training history.
- [x] **T013** Refactor `train_classifier` in `src/train_classifier.py` to fixed
  epochs, one final test, and `TrainingOutcome` as defined in `plan.md` Section 4.
- [x] **T014** Refactor `_train_eval_ssl_embedding` in `src/experiment.py` to the
  same test-once contract and parameterize `num_classes`.
- [x] **T015** Normalize `_train_eval_semi_supervised` to `TrainingOutcome` and
  rename its public framework identifier to `label_spreading_proxy`.
- [x] **T016** Update checkpoint metadata to contain `trained_epochs`,
  `test_loss`, `test_accuracy`, and protocol version only.
- [x] **T017** Add a static regression test that forbids legacy metric names in
  `src/`, `scripts/`, and `configs/`.

**Checkpoint:**

```powershell
python -m pytest -q tests/test_evaluation_protocol.py
rg -n "best_test_accuracy|final_test_accuracy" src scripts configs
```

The `rg` command must produce no matches.

## Phase 3 — Nested acquisition trajectory

- [x] **T018** Add tests for valid/invalid `round_query_sizes` in
  `tests/test_protocol_v2_integration.py`.
- [x] **T019** Add tests for `validate_query`: wrong size, duplicates, labeled
  indices, out-of-range indices, and valid global indices.
- [x] **T020** Remove `_round_query_sizes` and change `run_single_experiment` in
  `src/experiment.py` to consume explicit per-round query sizes.
- [x] **T021** Normalize all selector adapters in `src/experiment.py` to return
  global training indices.
- [x] **T022** Preserve ordered acquisition history separately from sorted set
  membership and assert nestedness after each round.
- [x] **T023** Reset the training seed immediately before every model
  construction and pass the round DataLoader seed explicitly.
- [x] **T024** Add `configs/protocol_v2_pilot.yaml` with the exact configuration
  contract from `spec.md`, initially limited to `tpcrp` and `tpcrp_ccfl`.

**Checkpoint:** A two-round synthetic experiment produces cumulative budgets 10
and 20, never reselects an index, and test evaluation occurs twice total.

## Phase 4 — Immutable artifacts and resume

- [x] **T025** Add unit tests for canonical JSON, config digests, and stable run
  IDs in `tests/test_artifacts.py`.
- [x] **T026** Add artifact rejection tests for missing fields, NaN/Infinity,
  legacy keys, wrong protocol, wrong digest, wrong run ID, and empty rounds.
- [x] **T027** Add an atomic-write test that verifies no temporary file remains
  after success and the previous complete artifact survives a simulated failure.
- [x] **T028** Implement `src/artifacts.py` from `plan.md` Section 6 and complete
  the validator for the full schema.
- [x] **T029** Update `src/experiment.py` to return an in-memory run artifact and
  write it once after all rounds succeed.
- [x] **T030** Refactor `scripts/run_experiments.py` to support `--config`,
  `--method`, `--seed`, `--dry-run`, and validated resume behavior.
- [x] **T031** Add a resume integration test: execute two of three runs, restart,
  and assert only the missing run executes.
- [x] **T032** Store environment and provenance: effective config, git commit,
  dirty flag, Python/package versions, device, and checkpoint SHA-256.

**Gate A checkpoint:**

```powershell
python -m pytest -q
python -m scripts.run_experiments --config configs/protocol_v2_pilot.yaml --dry-run
```

Two fresh synthetic executions must produce identical canonical artifacts after
removing timestamps and measured durations.

## Phase 5 — Baseline fidelity

- [x] **T033** Add hand-calculated initial and iterative CoreSet fixtures in
  `tests/test_selectors.py`.
- [x] **T034** Replace `kcenter_selector` in `src/selectors.py` with the
  global-index iterative implementation from `plan.md` Section 7.
- [x] **T035** Rename noncanonical BADGE/DBAL/BALD variants with `_proxy` and
  remove them from `configs/protocol_v2_pilot.yaml`.
- [x] **T036** Add exact tests for decimal radius generation, pseudo-label ball
  purity, deterministic radius choice, and greedy tie breaking in
  `tests/test_selectors.py`.
- [x] **T037** Implement the label-free pseudo-label purity radius policy and
  `probcover_selector` from `plan.md` Section 8 in `src/selectors.py`.
- [x] **T038** Compare selected indices and coverage on the fixed fixture with
  the official implementation; save the fixture, not external generated caches,
  under `tests/fixtures/`.
- [x] **T039** Expand the pilot methods to Random, TypiClust, iterative CoreSet,
  ProbCover, and CCFL only after T038 passes.

**Checkpoint:** All selectors return exact-size, unique, pool-only global indices
and deterministic outputs on the fixture.

## Phase 6 — CCFL ablations and diagnostics

- [x] **T040** Add exact-distance and exact-cosine unit tests in
  `tests/test_diagnostics.py`.
- [x] **T041** Implement `selection_diagnostics` and `timed_selection` in
  `src/diagnostics.py`.
- [x] **T042** Add selector tests proving `refine_steps=0` equals candidate-only
  output and disabled weights equal unit weights.
- [x] **T043** Add explicit CCFL variant configuration and remove hidden CCFL
  defaults from `src/experiment.py`.
- [x] **T044** Record cumulative-set diagnostics and selector-only runtime in
  every round artifact.
- [x] **T045** Add a post-hoc class-balance analysis function that accepts labels
  explicitly and is callable only from reporting code, never selection code.
- [x] **T046** Add an import-boundary test preventing `src/selectors.py` from
  importing dataset targets or post-hoc label analysis.

**Checkpoint:** The synthetic benchmark reports all diagnostics for TypiClust and
each CCFL variant without reading labels during selection.

## Phase 7 — Aggregation, statistics, and CV evidence

- [x] **T047** Add tests for duplicate/missing pairing keys in
  `tests/test_statistics.py`.
- [x] **T048** Add fixed expected tests for `paired_bootstrap_ci` and
  `holm_adjust`.
- [x] **T049** Implement `src/statistics.py` from `plan.md` Section 10.
- [x] **T050** Rewrite `scripts/aggregate_results.py` to validate per-run JSON
  and emit protocol-v2 round-level CSV; remove silent deduplication.
- [x] **T051** Rewrite `scripts/run_stats.py` to use paired replicate-level
  `test_accuracy`, fixed bootstrap seed, and Holm correction.
- [x] **T052** Rewrite `scripts/make_plots.py` to keep frameworks separate and
  use explicitly labeled 95% paired intervals or standard deviations.
- [x] **T053** Implement `scripts/generate_cv_evidence.py` and validate its output
  against the schema in `spec.md` Section 10.
- [x] **T054** Update `README.md` with protocol-v2 commands, limitations, and a
  prominent statement that legacy results used test-selected checkpoints.

**Checkpoint:** Delete only generated protocol-v2 summaries, regenerate them from
run JSON, and compare canonical outputs for equality.

## Phase 8 — Pilot and confirmation

- [x] **T054A** Harden pre-run evidence: require exact config digests, endpoint,
  seed sets, clean uniform environments, checkpoint provenance, configured
  artifact schedules/epochs/seeds, and mechanistically correct ablation
  contrasts.
- [x] **T054B** Make SimCLR checkpoint creation final-epoch, atomic,
  overwrite-safe, and accompanied by a code/config/checksum manifest.
- [ ] **T054C** Train SimCLR once from a clean commit, insert its SHA-256 into
  both SimCLR protocol configs, enforce it in the loader, and re-freeze the
  pilot and confirmation config digests.
- [ ] **T055** Freeze the Gate-B config digest and record it in
  `configs/protocol_v2_pilot.yaml` documentation.
- [ ] **T056** Run the five-seed CIFAR-10 frozen-embedding pilot for TypiClust and
  CCFL first; do not run the larger grid yet.
- [ ] **T057** Generate Gate-B evidence and record a pass/fail decision without
  changing thresholds or the primary endpoint.
- [ ] **T058** If Gate B passes, run all five faithful methods and CCFL ablations
  under the locked pilot protocol.
- [x] **T059** Create `configs/protocol_v2_confirmation.yaml` by extending the
  replicate seed list to ten without changing method behavior or endpoints.
- [ ] **T060** Run Gate-C confirmation and generate `cv_evidence.json`.

**Checkpoint:** The ten-seed evidence artifact contains the exact run IDs used in
the primary comparison and passes the Gate-C rules automatically.

## Phase 9 — CIFAR-100 only after Gate C

- [ ] **T061** Add dataset-registry tests for CIFAR-10/100 without downloading
  data.
- [ ] **T062** Implement `DatasetSpec` and parameterize class count,
  normalization, model output, transforms, and label summaries.
- [ ] **T063** Add cache-key tests proving different datasets, checkpoints, or
  representation configs cannot share an embedding cache path.
- [ ] **T064** Add CIFAR-100 smoke configuration and run a small subset pipeline.
- [ ] **T065** Declare CIFAR-100 budgets and representation source before the
  full confirmation run.
- [ ] **T066** Run Gate D and regenerate the evidence artifact.

## Final review checklist

- [ ] **T067** Every MUST requirement in `spec.md` maps to at least one test or
  explicit review item.
- [ ] **T068** Every configured scientific parameter is present in the effective
  config and run digest.
- [ ] **T069** No test set enters selection, epoch choice, hyperparameter choice,
  continuation logic, or checkpoint choice.
- [ ] **T070** No primary result pools frameworks or uses legacy metrics.
- [ ] **T071** README commands work from a clean environment.
- [ ] **T072** CV bullets are populated only from generated `cv_evidence.json`.

## Implementation rule

Implement one phase at a time. Do not begin GPU experiments while any earlier
phase checkpoint is failing. Do not weaken a failing test to make production
code pass; change a requirement through a reviewed protocol-version update.
