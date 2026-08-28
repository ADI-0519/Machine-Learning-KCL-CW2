# Technical Plan: Deterministic Active-Learning Evaluation

This plan is implementation-prescriptive. Code blocks define the intended
interfaces and critical control flow; an implementation may improve naming or
factorization only if all requirements and tests remain satisfied.

## 1. Architecture decisions

### AD-001 — Fixed epochs instead of validation splitting

The selected labeled sets are extremely small. Holding out part of ten labels
for early stopping would change the learning problem and annotation accounting.
Protocol v2 therefore locks hyperparameters and epoch counts before a run,
trains for exactly that duration, and evaluates the test set once afterward.

### AD-002 — One nested trajectory per method and seed

The current code independently runs each total budget and splits it across two
rounds. Protocol v2 instead consumes an explicit acquisition schedule. A schedule
of `[10, 10, 10, 10, 10]` produces cumulative budgets 10, 20, 30, 40, and 50 in
one nested trajectory.

### AD-003 — Isolated component seeds

Mutable RNG call order must not couple methods. Each round derives independent
clustering, selection, training, and DataLoader seeds. Clustering and training
seeds omit the method name so paired methods see the same cluster initialization
and model initialization. Selector seeds include the method name.

### AD-004 — Immutable per-run JSON

CSV append is neither atomic nor safe for resumption. Each run writes one JSON
artifact to a deterministic path. Aggregated CSV files are derived outputs and
can always be regenerated.

### AD-005 — Fail closed on legacy data

Protocol-v2 analysis accepts only schema-v2 JSON. It must never infer clean
fields from `best_test_accuracy` or silently combine protocol versions.

## 2. Target file map

```text
configs/
├── protocol_v2_pilot.yaml
└── protocol_v2_confirmation.yaml
src/
├── protocol.py               # protocol constants, typed results, seed derivation
├── artifacts.py              # canonical JSON, run IDs, atomic writes, validation
├── diagnostics.py            # label-free selection diagnostics
├── statistics.py             # pairing, bootstrap CI, Holm correction
├── data.py                    # deterministic DataLoaders + dataset registry
├── train_classifier.py       # fixed-epoch train, one final test
├── experiment.py             # explicit nested acquisition schedule
├── selectors.py              # iterative CoreSet + CCFL variants
└── seed.py                   # strict deterministic configuration
scripts/
├── run_experiments.py        # validated CLI + resume
├── aggregate_results.py      # JSON artifacts -> tables
├── run_stats.py              # paired protocol-v2 statistics
└── generate_cv_evidence.py   # machine-readable evidence summary
tests/
├── test_evaluation_protocol.py
├── test_determinism.py
├── test_artifacts.py
├── test_selectors.py
├── test_diagnostics.py
├── test_statistics.py
└── test_protocol_v2_integration.py
```

## 3. Chunk 1 — Protocol types and stable seeds

### 3.1 Add `src/protocol.py`

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

PROTOCOL_VERSION = "2.0"
TEST_POLICY = "once_after_fixed_epochs"


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable 31-bit seed without Python's randomized hash()."""
    material = "\x1f".join([str(base_seed), *(str(part) for part in parts)])
    digest = sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


@dataclass(frozen=True)
class SeedBundle:
    replicate: int
    clustering: int
    selector: int
    training: int
    dataloader: int

    @classmethod
    def for_round(
        cls,
        *,
        replicate: int,
        dataset: str,
        framework: str,
        method: str,
        round_id: int,
    ) -> "SeedBundle":
        shared = (dataset, framework, round_id)
        return cls(
            replicate=replicate,
            clustering=derive_seed(replicate, *shared, "clustering"),
            selector=derive_seed(replicate, *shared, method, "selector"),
            training=derive_seed(replicate, *shared, "training"),
            dataloader=derive_seed(replicate, *shared, "dataloader"),
        )


@dataclass(frozen=True)
class EvaluationMetrics:
    trained_epochs: int
    test_loss: float
    test_accuracy: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass
class TrainingOutcome:
    model: Any
    metrics: EvaluationMetrics
    history: list[dict[str, int | float]]
```

### 3.2 Update `src/seed.py`

```python
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
```

`CUBLAS_WORKSPACE_CONFIG=:4096:8` must be set by the process launcher before
PyTorch initializes CUDA. `PYTHONHASHSEED` cannot be changed retrospectively, so
protocol code must never depend on hash iteration order; the launcher records its
value for provenance. `set_seed()` must run before model, dataset, or DataLoader
construction. The environment manifest records strict deterministic mode.

The seed regression fixture for replicate 42, CIFAR-10, frozen embeddings, and
round 1 is:

| Method | Clustering | Selector | Training | DataLoader |
|---|---:|---:|---:|---:|
| `tpcrp` | 824551655 | 826506532 | 1924623876 | 200152058 |
| `tpcrp_ccfl` | 824551655 | 1939970908 | 1924623876 | 200152058 |

### 3.3 Update DataLoaders in `src/data.py`

Extend `make_subset_loader` with a required keyword-only `seed` argument:

```python
def make_subset_loader(
    dataset,
    indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    *,
    seed: int,
) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )
```

Every call site must pass the appropriate `SeedBundle.dataloader` value. Helper
loaders used only for deterministic inference must also receive an explicit seed.

## 4. Chunk 2 — Remove test-driven checkpoint selection

### 4.1 Replace the control flow in `src/train_classifier.py`

`evaluate_classifier` remains a pure evaluation function. `train_classifier`
must follow this structure:

```python
def train_classifier(
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    device: torch.device,
    checkpoint_path: str | Path | None = None,
    verbose: bool = True,
) -> TrainingOutcome:
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    model = CIFARClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=epochs)
    history: list[dict[str, int | float]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimiser=optimiser,
            criterion=criterion,
            device=device,
        )
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "lr": optimiser.param_groups[0]["lr"],
            }
        )

    test_metrics = evaluate_classifier(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )
    metrics = EvaluationMetrics(
        trained_epochs=epochs,
        test_loss=test_metrics["loss"],
        test_accuracy=test_metrics["accuracy"],
    )

    if checkpoint_path is not None:
        save_checkpoint(
            model=model,
            optimiser=optimiser,
            scheduler=scheduler,
            epoch=epochs,
            metrics=metrics.to_dict(),
            checkpoint_path=checkpoint_path,
        )

    return TrainingOutcome(model=model, metrics=metrics, history=history)
```

There must be no `best_state_dict`, `best_epoch`, or per-epoch test call.

### 4.2 Replace `_train_eval_ssl_embedding` control flow

Train on selected train embeddings for all fixed epochs. Test tensors must not be
moved to the device or passed through the model until the training loop ends.

```python
for epoch in range(1, epochs + 1):
    linear.train()
    optimizer.zero_grad(set_to_none=True)
    logits = linear(x_train_device)
    loss = criterion(logits, y_train_device)
    loss.backward()
    optimizer.step()
    scheduler.step()
    history.append(
        {
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
    )

linear.eval()
with torch.no_grad():
    test_logits = linear(x_test.to(device, non_blocking=True))
    test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()

metrics = EvaluationMetrics(
    trained_epochs=epochs,
    test_loss=float(log_loss(test_labels, test_probs, labels=np.arange(num_classes))),
    test_accuracy=float(accuracy_score(test_labels, test_probs.argmax(axis=1))),
)
return TrainingOutcome(model=linear, metrics=metrics, history=history)
```

The function must accept `num_classes`; it must not hard-code ten output units.

### 4.3 Normalize the LabelSpreading return type

LabelSpreading already tests once. Return `TrainingOutcome` with
`trained_epochs=1` and the same `EvaluationMetrics` schema. Rename the framework
to `label_spreading_proxy` everywhere; it is excluded from primary experiments.

## 5. Chunk 3 — Explicit nested acquisition schedule

### 5.1 Replace total-budget splitting

Remove `_round_query_sizes(total_budget, rounds)`. `run_single_experiment` accepts
one schedule from validated configuration:

```python
round_query_sizes = [int(value) for value in selection_cfg["round_query_sizes"]]
if not round_query_sizes or any(value <= 0 for value in round_query_sizes):
    raise ValueError("selection.round_query_sizes must contain positive integers")

for round_id, query_size in enumerate(round_query_sizes, start=1):
    pool_indices = np.setdiff1d(all_indices, labeled_indices, assume_unique=False)
    if query_size > len(pool_indices):
        raise ValueError(
            f"round {round_id} query_size={query_size} exceeds pool={len(pool_indices)}"
        )
    seeds = SeedBundle.for_round(
        replicate=replicate_seed,
        dataset=data_name,
        framework=framework,
        method=method,
        round_id=round_id,
    )
    # Selection receives seeds.clustering and a fresh
    # np.random.default_rng(seeds.selector).
    # Training starts with set_seed(seeds.training) and constructs its loader
    # with seeds.dataloader.
```

After selection, validate global indices before training:

```python
def validate_query(
    query: np.ndarray,
    *,
    pool_indices: np.ndarray,
    query_size: int,
) -> None:
    query = np.asarray(query, dtype=int)
    if query.ndim != 1:
        raise ValueError("query must be one-dimensional")
    if len(query) != query_size:
        raise ValueError(f"expected {query_size} query indices, got {len(query)}")
    if len(np.unique(query)) != query_size:
        raise ValueError("query contains duplicate indices")
    if not np.isin(query, pool_indices).all():
        raise ValueError("query contains indices outside the unlabeled pool")
```

All selector boundaries must use global training indices. Remove the current
mixture of pool-local and global indices; local conversions may exist only inside
selector adapters and must be tested.

### 5.2 Round record

Every round emits exactly:

```python
round_record = {
    "round": round_id,
    "query_size": query_size,
    "cumulative_budget": len(labeled_indices),
    "new_indices": newly_selected.tolist(),
    "selected_indices": labeled_indices.tolist(),
    "seeds": asdict(seeds),
    "trained_epochs": outcome.metrics.trained_epochs,
    "test_loss": outcome.metrics.test_loss,
    "test_accuracy": outcome.metrics.test_accuracy,
    "selection_diagnostics": diagnostics,
}
```

The selected index lists are ordered. Preserve acquisition order separately from
the sorted set used for membership checks.

## 6. Chunk 4 — Immutable artifacts and resumption

### 6.1 Add `src/artifacts.py`

```python
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from .protocol import PROTOCOL_VERSION


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_digest(effective_config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(effective_config).encode("utf-8")).hexdigest()


def build_run_id(effective_config: dict[str, Any]) -> str:
    return f"v{PROTOCOL_VERSION}-{config_digest(effective_config)[:16]}"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_artifact(payload: dict[str, Any]) -> None:
    required = {
        "protocol_version",
        "run_id",
        "config_digest",
        "effective_config",
        "environment",
        "rounds",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"artifact missing fields: {sorted(missing)}")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("artifact protocol version mismatch")
    expected_digest = config_digest(payload["effective_config"])
    if payload["config_digest"] != expected_digest:
        raise ValueError("artifact config digest mismatch")
    if payload["run_id"] != f"v{PROTOCOL_VERSION}-{expected_digest[:16]}":
        raise ValueError("artifact run ID mismatch")
    if not payload["rounds"]:
        raise ValueError("artifact has no rounds")
    for record in payload["rounds"]:
        for key in ("test_accuracy", "test_loss"):
            if not math.isfinite(float(record[key])):
                raise ValueError(f"non-finite {key}")
        if "best_test_accuracy" in record:
            raise ValueError("legacy best-test metric is forbidden")
```

The production validator should additionally verify the complete schema defined
in `spec.md`; tests cover every rejection branch.

### 6.2 Resume behavior

Before a run:

1. Build its effective config, including one method and one replicate seed.
2. Derive the run ID.
3. If the artifact does not exist, execute.
4. If it exists, parse and validate it.
5. Skip only when the stored digest equals the requested digest.
6. Never use `drop_duplicates(keep="last")` in protocol-v2 processing.

## 7. Chunk 5 — Faithful iterative CoreSet

Replace the current pool-only k-center selector with this global-index contract:

```python
def kcenter_selector(
    embeddings: np.ndarray,
    pool_indices: np.ndarray,
    labeled_indices: np.ndarray,
    query_size: int,
) -> np.ndarray:
    pool = np.asarray(pool_indices, dtype=int)
    labeled = np.asarray(labeled_indices, dtype=int)
    if query_size <= 0 or query_size > len(pool):
        raise ValueError("invalid query_size")

    pool_emb = embeddings[pool]
    if len(labeled):
        anchors = embeddings[labeled]
        min_d2 = np.min(
            np.sum((pool_emb[:, None, :] - anchors[None, :, :]) ** 2, axis=2),
            axis=1,
        )
    else:
        center = embeddings.mean(axis=0, keepdims=True)
        min_d2 = np.sum((pool_emb - center) ** 2, axis=1)

    chosen_local: list[int] = []
    available = np.ones(len(pool), dtype=bool)
    for _ in range(query_size):
        scores = np.where(available, min_d2, -np.inf)
        next_local = int(np.argmax(scores))  # deterministic lowest-index tie break
        chosen_local.append(next_local)
        available[next_local] = False
        new_d2 = np.sum((pool_emb - pool_emb[next_local]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, new_d2)

    return pool[np.asarray(chosen_local, dtype=int)]
```

For large arrays, a later optimization may chunk anchor distances without
changing outputs. The unoptimized reference is the correctness oracle.

## 8. Chunk 6 — Deterministic ProbCover

Protocol v2 follows the paper's label-free radius heuristic: K-Means pseudo-
labels with the known dataset class count, then the largest radius at which at
least 95% of balls contain a single pseudo-label. The search grid is explicit in
the effective config.

### 8.1 Decimal radius grid

```python
from decimal import Decimal


def decimal_grid(minimum: float, maximum: float, step: float) -> np.ndarray:
    low = Decimal(str(minimum))
    high = Decimal(str(maximum))
    increment = Decimal(str(step))
    if increment <= 0 or high < low:
        raise ValueError("invalid delta search range")
    values: list[float] = []
    current = low
    while current <= high:
        values.append(float(current))
        current += increment
    return np.asarray(values, dtype=np.float64)
```

### 8.2 Radius estimation

For each point, compute the distance to its nearest neighbor with a different
pseudo-label, limited to the largest candidate radius. A ball is pure at radius
`delta` exactly when no differently labeled neighbor has distance `<= delta`.

```python
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def estimate_probcover_delta(
    embeddings: np.ndarray,
    *,
    num_classes: int,
    candidates: np.ndarray,
    alpha: float,
    clustering_seed: int,
) -> float:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if candidates.ndim != 1 or not len(candidates):
        raise ValueError("candidates must be a non-empty vector")
    if np.any(np.diff(candidates) <= 0) or candidates[0] <= 0:
        raise ValueError("candidates must be strictly increasing and positive")

    pseudo = KMeans(
        n_clusters=num_classes,
        random_state=clustering_seed,
        n_init=10,
    ).fit_predict(embeddings)
    neighbors = NearestNeighbors(
        radius=float(candidates[-1]),
        metric="euclidean",
    ).fit(embeddings)
    distances, indices = neighbors.radius_neighbors(
        embeddings,
        return_distance=True,
        sort_results=True,
    )

    first_impure = np.full(len(embeddings), np.inf, dtype=np.float64)
    for center, (center_distances, center_indices) in enumerate(zip(distances, indices)):
        different = pseudo[center_indices] != pseudo[center]
        if np.any(different):
            first_impure[center] = float(center_distances[np.flatnonzero(different)[0]])

    purity = np.asarray(
        [np.mean(first_impure > delta) for delta in candidates],
        dtype=np.float64,
    )
    valid = np.flatnonzero(purity >= alpha)
    if not len(valid):
        raise ValueError("no ProbCover radius satisfies the configured purity threshold")
    return float(candidates[valid[-1]])
```

### 8.3 Greedy selection

```python
def probcover_selector(
    embeddings: np.ndarray,
    pool_indices: np.ndarray,
    labeled_indices: np.ndarray,
    query_size: int,
    *,
    delta: float,
) -> np.ndarray:
    pool = np.sort(np.asarray(pool_indices, dtype=int))
    labeled = np.asarray(labeled_indices, dtype=int)
    if query_size <= 0 or query_size > len(pool):
        raise ValueError("invalid query_size")

    graph = NearestNeighbors(radius=delta, metric="euclidean").fit(embeddings)
    neighborhoods = graph.radius_neighbors(
        embeddings,
        return_distance=False,
    )
    uncovered = np.ones(len(embeddings), dtype=bool)
    for index in labeled:
        uncovered[neighborhoods[index]] = False

    available = np.ones(len(pool), dtype=bool)
    selected: list[int] = []
    for _ in range(query_size):
        gains = np.asarray(
            [uncovered[neighborhoods[index]].sum() if available[pos] else -1
             for pos, index in enumerate(pool)],
            dtype=np.int64,
        )
        position = int(np.argmax(gains))
        chosen = int(pool[position])
        selected.append(chosen)
        available[position] = False
        uncovered[neighborhoods[chosen]] = False

    return np.asarray(selected, dtype=int)
```

The reference code prioritizes correctness and deterministic lowest-global-index
tie breaking. A sparse/vectorized optimization may replace it only after exact
fixture equivalence is established. Radius estimation is performed once per
representation/configuration, cached by its full digest, and never tuned on test
accuracy or ground-truth labels.

## 9. Chunk 7 — CCFL variants and diagnostics

### 9.1 Explicit CCFL variants

Add configuration entries rather than new hand-written selectors:

```yaml
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
```

`tpcrp_ccfl_selector` receives `use_cluster_weights: bool`; passing false must
create unit weights. Method metadata records the resolved parameters.

### 9.2 Add `src/diagnostics.py`

```python
from __future__ import annotations

from time import perf_counter

import numpy as np


def selection_diagnostics(
    embeddings: np.ndarray,
    selected_indices: np.ndarray,
    *,
    evaluation_indices: np.ndarray,
    selected_typicality_scores: np.ndarray | None = None,
) -> dict[str, float]:
    selected = np.asarray(selected_indices, dtype=int)
    evaluation = np.asarray(evaluation_indices, dtype=int)
    if not len(selected):
        raise ValueError("selected_indices cannot be empty")

    selected_emb = embeddings[selected]
    evaluation_emb = embeddings[evaluation]
    d2 = (
        np.sum(evaluation_emb**2, axis=1, keepdims=True)
        - 2.0 * evaluation_emb @ selected_emb.T
        + np.sum(selected_emb**2, axis=1)[None, :]
    )
    nearest = np.sqrt(np.maximum(d2.min(axis=1), 0.0))

    normalized = selected_emb / (
        np.linalg.norm(selected_emb, axis=1, keepdims=True) + 1e-12
    )
    cosine = normalized @ normalized.T
    if len(selected) > 1:
        mask = ~np.eye(len(selected), dtype=bool)
        mean_pairwise_cosine = float(cosine[mask].mean())
    else:
        mean_pairwise_cosine = 0.0

    output = {
        "nearest_distance_mean": float(nearest.mean()),
        "nearest_distance_p95": float(np.quantile(nearest, 0.95)),
        "nearest_distance_max": float(nearest.max()),
        "selected_pairwise_cosine_mean": mean_pairwise_cosine,
    }
    if selected_typicality_scores is not None:
        output["selected_typicality_mean"] = float(
            np.asarray(selected_typicality_scores).mean()
        )
    return output


def timed_selection(callable_, /, *args, **kwargs):
    started = perf_counter()
    selected = callable_(*args, **kwargs)
    elapsed = perf_counter() - started
    return selected, elapsed
```

Compute diagnostics over train embeddings only. For iterative rounds, report
diagnostics for the complete cumulative selected set, evaluate coverage over
the remaining unlabeled pool after the round, and record runtime for the new
query operation under `timings.selector_seconds`.

Compute selected-point typicality by querying only the cumulative selected
embeddings against the full train-neighbor index. Do not run an all-points k-NN
query merely to discard every unselected score.

The reporting-only post-hoc class-balance function returns JSON-compatible
class counts, proportions, observed-class coverage, and normalized entropy. No
selector may import that function or any dataset target provider.

## 10. Chunk 8 — Statistics

### 10.1 Pair first, then calculate

`src/statistics.py` must expose pure functions. The paired bootstrap is the
primary interval:

```python
def paired_bootstrap_ci(
    differences: np.ndarray,
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 20260201,
) -> tuple[float, float]:
    diff = np.asarray(differences, dtype=float)
    if diff.ndim != 1 or len(diff) < 2 or not np.isfinite(diff).all():
        raise ValueError("differences must be a finite vector with at least two values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diff), size=(resamples, len(diff)))
    means = diff[indices].mean(axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not ((0.0 <= values) & (values <= 1.0)).all():
        raise ValueError("p-values must be a one-dimensional vector in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.maximum.accumulate(
        np.minimum(1.0, ranked * np.arange(len(values), 0, -1))
    )
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted
```

The pairing table must reject duplicate pairing keys and missing method partners.
Statistics must use `test_accuracy`, never a best-epoch score.

The locked Phase 7 implementation uses `method_a - method_b`, bootstrap seed
`20260201`, 10,000 resamples, and a 95% percentile interval. It applies Holm
correction to the paired t-test p-values for every common cumulative budget in
the configured comparison. Round and summary tables are deterministic CSVs
under `reports/`; summaries never pool datasets or frameworks. Plots use
replicate standard-deviation bands. Coverage and runtime formulas follow the
explicit conventions in `spec.md` Statistics and reporting.

### 10.2 Evidence generation

`scripts/generate_cv_evidence.py` accepts the locked primary comparison and
writes `results/protocol_v2/reports/cv_evidence.json`. It must compute numbers
from validated run artifacts and include artifact run IDs for traceability.

## 11. Chunk 9 — Configuration validation and CLI

Use a small typed validator rather than adding a large framework. Dataclasses or
`TypedDict` plus explicit validation are acceptable. The loader must:

- reject unknown scientific keys;
- reject missing protocol version;
- reject `test_policy` other than `once_after_fixed_epochs`;
- reject empty/invalid acquisition schedules;
- reject duplicate methods or seeds;
- resolve all defaults into the effective config before hashing;
- accept `--config`, `--method`, `--seed`, and `--dry-run` CLI options;
- print the deterministic run IDs during dry-run without loading data.

Representative CLI:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()
```

## 12. Chunk 10 — Dataset registry and CIFAR-100

Do not parameterize the dataset until Gates A through C pass. Then replace
hard-coded CIFAR-10 constructors and class counts with:

```python
@dataclass(frozen=True)
class DatasetSpec:
    dataset_type: type
    num_classes: int
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


DATASETS = {
    "cifar10": DatasetSpec(
        dataset_type=datasets.CIFAR10,
        num_classes=10,
        image_size=32,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ),
    "cifar100": DatasetSpec(
        dataset_type=datasets.CIFAR100,
        num_classes=100,
        image_size=32,
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    ),
}
```

The checkpoint and embedding cache digest must include the dataset spec and
representation configuration. A CIFAR-10 checkpoint must never satisfy a
CIFAR-100 cache request unless the representation was explicitly trained as a
shared representation and the effective config says so.

## 13. Test strategy

Tests are written before each production chunk. No test downloads CIFAR or
requires a GPU.

### Unit tests

- Test evaluation-call counts with tiny monkeypatched models and in-memory
  `TensorDataset` loaders.
- Test stable seed derivation against fixed expected integers.
- Test DataLoader batch ordering across two constructions.
- Test every artifact-validation failure mode.
- Test selectors on arrays small enough for hand-calculated expected outputs.
- Test diagnostics against explicit distance and cosine calculations.
- Test bootstrap repeatability and Holm adjustment against fixed fixtures.

### Integration tests

Create a synthetic 3-class dataset and fixed 8-dimensional embeddings. Run two
rounds for Random, TypiClust, CoreSet, and CCFL. Monkeypatch the expensive model
trainer with a deterministic tiny classifier but retain artifact writing,
resumption, aggregation, and statistics.

### Static guards

The verification command must fail if protocol-v2 source contains forbidden
metric names outside migration tests:

```powershell
$matches = rg -n "best_test_accuracy|final_test_accuracy" src scripts configs
if ($LASTEXITCODE -eq 0) { $matches; exit 1 }
```

Legacy report files and the specification itself are excluded from this guard.

## 14. Verification commands

```powershell
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
python -m pytest -q
python -m pytest --cov=src.protocol --cov=src.artifacts --cov=src.diagnostics --cov=src.statistics --cov-fail-under=90
python -m compileall -q src scripts tests
python -m scripts.run_experiments --config configs/protocol_v2_pilot.yaml --dry-run
python -m scripts.run_experiments --config configs/protocol_v2_pilot.yaml
python -m scripts.aggregate_results --root results/protocol_v2
python -m scripts.run_stats --root results/protocol_v2
python -m scripts.generate_cv_evidence --root results/protocol_v2
```

`compileall` is a CI command; local implementation verification may use an
AST-only parser if bytecode files are undesirable.

## 15. Requirement traceability

| Requirements | Primary implementation/tests |
|---|---|
| FR-001–FR-005 | T011–T017 |
| FR-006–FR-009 | T018–T024 |
| FR-010–FR-015 | T005–T010, T023, T032 |
| FR-016–FR-021 | T025–T032, T050–T053 |
| FR-022–FR-026 | T019, T033–T046 |
| FR-027–FR-031 | T047–T053, T055–T060 |
| QR-001–QR-007 | T002–T004, every phase checkpoint, T067–T071 |
| SC-001–SC-004 | Gate A checkpoint |
| SC-005–SC-007 | T055–T060 and generated evidence validation |

## 16. Rollback and compatibility

- No legacy result file is rewritten.
- Protocol-v2 outputs use a new root and can be deleted independently if a run
  is invalidated.
- Existing public APIs may retain wrappers for one release, but wrappers must
  not emit legacy metrics.
- If strict CUDA determinism fails because an operation lacks a deterministic
  implementation, the run fails. The maintainer may use CPU or revise the
  protocol version; silently enabling `warn_only=True` is forbidden.
