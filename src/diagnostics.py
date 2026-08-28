"""Label-free diagnostics for active-learning selections."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import ParamSpec, TypeVar

import numpy as np

P = ParamSpec("P")
T = TypeVar("T")


def _validated_embeddings(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings)
    if values.ndim != 2 or len(values) == 0 or values.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty two-dimensional array")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("embeddings must contain only finite numeric values")
    return values


def _validated_global_indices(
    indices: np.ndarray,
    *,
    name: str,
    num_samples: int,
) -> np.ndarray:
    values = np.asarray(indices)
    if values.ndim != 1 or not len(values):
        raise ValueError(f"{name} cannot be empty and must be one-dimensional")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must contain integer global indices")
    values = values.astype(int, copy=False)
    if len(np.unique(values)) != len(values):
        raise ValueError(f"{name} must contain unique global indices")
    if np.any(values < 0) or np.any(values >= num_samples):
        raise ValueError(f"{name} contains an out-of-range global index")
    return values


def selection_diagnostics(
    embeddings: np.ndarray,
    selected_indices: np.ndarray,
    *,
    evaluation_indices: np.ndarray,
    selected_typicality_scores: np.ndarray | None = None,
) -> dict[str, float]:
    """Measure cumulative selection quality over explicit remaining pool indices."""
    values = _validated_embeddings(embeddings)
    selected = _validated_global_indices(
        selected_indices,
        name="selected_indices",
        num_samples=len(values),
    )
    evaluation = _validated_global_indices(
        evaluation_indices,
        name="evaluation_indices",
        num_samples=len(values),
    )
    if np.intersect1d(selected, evaluation).size:
        raise ValueError("selected_indices and evaluation_indices must be disjoint")

    selected_embeddings = values[selected]
    evaluation_embeddings = values[evaluation]
    squared_distances = (
        np.sum(evaluation_embeddings**2, axis=1, keepdims=True)
        - 2.0 * evaluation_embeddings @ selected_embeddings.T
        + np.sum(selected_embeddings**2, axis=1)[None, :]
    )
    nearest_distances = np.sqrt(np.maximum(squared_distances.min(axis=1), 0.0))

    norms = np.linalg.norm(selected_embeddings, axis=1, keepdims=True)
    normalized = np.divide(
        selected_embeddings,
        norms,
        out=np.zeros_like(selected_embeddings, dtype=np.float64),
        where=norms > 0.0,
    )
    cosine = np.clip(normalized @ normalized.T, -1.0, 1.0)
    if len(selected) > 1:
        off_diagonal = ~np.eye(len(selected), dtype=bool)
        mean_pairwise_cosine = float(cosine[off_diagonal].mean())
    else:
        mean_pairwise_cosine = 0.0

    output = {
        "nearest_distance_mean": float(nearest_distances.mean()),
        "nearest_distance_p95": float(np.quantile(nearest_distances, 0.95)),
        "nearest_distance_max": float(nearest_distances.max()),
        "selected_pairwise_cosine_mean": mean_pairwise_cosine,
    }
    if selected_typicality_scores is not None:
        typicality = np.asarray(selected_typicality_scores, dtype=np.float64)
        if typicality.ndim != 1 or len(typicality) != len(selected):
            raise ValueError("selected_typicality_scores must align with selected_indices")
        if not np.isfinite(typicality).all():
            raise ValueError("selected_typicality_scores must contain only finite values")
        if np.any(typicality < 0.0):
            raise ValueError("selected_typicality_scores must be non-negative")
        output["selected_typicality_mean"] = float(typicality.mean())
    return output


def timed_selection(
    callable_: Callable[P, T],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> tuple[T, float]:
    """Run one new-query selection call and return its monotonic elapsed seconds."""
    started = perf_counter()
    selected = callable_(*args, **kwargs)
    elapsed = perf_counter() - started
    if elapsed < 0.0:
        raise RuntimeError("monotonic selection timer returned a negative duration")
    return selected, elapsed
