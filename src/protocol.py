"""Shared contracts for the deterministic evaluation protocol."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256
from numbers import Integral
from typing import Any

import numpy as np

PROTOCOL_VERSION = "2.0"
TEST_POLICY = "once_after_fixed_epochs"


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable 31-bit component seed from explicit identifiers."""
    material = "\x1f".join([str(base_seed), *(str(part) for part in parts)])
    digest = sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def validate_round_query_sizes(values: Iterable[object]) -> list[int]:
    """Return an explicit acquisition schedule of strictly positive integers."""
    try:
        raw_values = list(values)
    except TypeError as exc:
        raise ValueError("selection.round_query_sizes must contain positive integers") from exc

    if not raw_values or any(
        isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0
        for value in raw_values
    ):
        raise ValueError("selection.round_query_sizes must contain positive integers")
    return [int(value) for value in raw_values]


def validate_query(
    query: np.ndarray,
    *,
    pool_indices: np.ndarray,
    query_size: int,
) -> None:
    """Validate unique global query indices against the current unlabeled pool."""
    query = np.asarray(query, dtype=int)
    pool_indices = np.asarray(pool_indices, dtype=int)
    if query.ndim != 1:
        raise ValueError("query must be one-dimensional")
    if pool_indices.ndim != 1:
        raise ValueError("pool_indices must be one-dimensional")
    if len(query) != query_size:
        raise ValueError(f"expected {query_size} query indices, got {len(query)}")
    if len(np.unique(query)) != query_size:
        raise ValueError("query contains duplicate indices")
    if not np.isin(query, pool_indices).all():
        raise ValueError("query contains indices outside the unlabeled pool")


@dataclass(frozen=True)
class SeedBundle:
    """Independent random seeds used by one active-learning round."""

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
    ) -> SeedBundle:
        """Create paired component seeds for a method and round."""
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
    """Test metrics measured once after a fixed training schedule."""

    trained_epochs: int
    test_loss: float
    test_accuracy: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a serialization-friendly metric mapping."""
        return asdict(self)


@dataclass
class TrainingOutcome:
    """A trained estimator, its final metrics, and training-only history."""

    model: Any
    metrics: EvaluationMetrics
    history: list[dict[str, int | float]]
