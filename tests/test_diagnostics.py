from __future__ import annotations

import numpy as np
import pytest

import src.diagnostics as diagnostics
from src.diagnostics import selection_diagnostics, timed_selection
from src.reporting import posthoc_class_balance
from src.typicality import compute_selected_typicality_scores


def test_selection_diagnostics_has_exact_distances_and_typicality() -> None:
    embeddings = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
        dtype=np.float64,
    )
    typicality = np.array([2.0, 3.0, 5.0, 7.0], dtype=np.float64)

    result = selection_diagnostics(
        embeddings,
        selected_indices=np.array([0]),
        evaluation_indices=np.array([1, 2, 3]),
        selected_typicality_scores=typicality[[0]],
    )

    assert result == {
        "nearest_distance_mean": pytest.approx(7.0 / 3.0),
        "nearest_distance_p95": pytest.approx(3.8),
        "nearest_distance_max": pytest.approx(4.0),
        "selected_pairwise_cosine_mean": 0.0,
        "selected_typicality_mean": 2.0,
    }


def test_selection_diagnostics_has_exact_mean_pairwise_cosine() -> None:
    embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 2.0]],
        dtype=np.float64,
    )

    result = selection_diagnostics(
        embeddings,
        selected_indices=np.array([0, 1, 2]),
        evaluation_indices=np.array([3]),
    )

    assert result["selected_pairwise_cosine_mean"] == pytest.approx(-1.0 / 3.0)
    assert result["nearest_distance_mean"] == pytest.approx(1.0)
    assert "selected_typicality_mean" not in result


@pytest.mark.parametrize(
    ("selected", "evaluation", "message"),
    [([], [0], "selected_indices"), ([0], [], "evaluation_indices"), ([0, 0], [1], "unique")],
)
def test_selection_diagnostics_rejects_invalid_index_sets(selected, evaluation, message) -> None:
    with pytest.raises(ValueError, match=message):
        selection_diagnostics(
            np.array([[0.0], [1.0]], dtype=np.float64),
            selected_indices=np.asarray(selected, dtype=int),
            evaluation_indices=np.asarray(evaluation, dtype=int),
        )


@pytest.mark.parametrize(
    ("embeddings", "selected", "evaluation", "typicality", "message"),
    [
        (np.array([]), [0], [1], None, "embeddings"),
        (np.array([[0.0], [np.nan]]), [0], [1], None, "finite"),
        (np.array([[0.0], [1.0]]), [0.5], [1], None, "integer"),
        (np.array([[0.0], [1.0]]), [2], [1], None, "out-of-range"),
        (np.array([[0.0], [1.0]]), [0], [0], None, "disjoint"),
        (np.array([[0.0], [1.0]]), [0], [1], [1.0, 2.0], "align"),
        (np.array([[0.0], [1.0]]), [0], [1], [np.inf], "finite"),
        (np.array([[0.0], [1.0]]), [0], [1], [-1.0], "non-negative"),
    ],
)
def test_selection_diagnostics_rejects_malformed_inputs(
    embeddings, selected, evaluation, typicality, message
) -> None:
    with pytest.raises(ValueError, match=message):
        selection_diagnostics(
            embeddings,
            selected_indices=np.asarray(selected),
            evaluation_indices=np.asarray(evaluation),
            selected_typicality_scores=None if typicality is None else np.asarray(typicality),
        )


def test_timed_selection_returns_value_and_monotonic_elapsed_time(monkeypatch) -> None:
    ticks = iter([10.0, 10.125])
    monkeypatch.setattr(diagnostics, "perf_counter", lambda: next(ticks))

    selected, elapsed = timed_selection(lambda value: value + 1, 4)

    assert selected == 5
    assert elapsed == pytest.approx(0.125)


def test_selected_typicality_queries_full_representation_for_selected_points_only() -> None:
    scores = compute_selected_typicality_scores(
        embeddings=np.array([[0.0], [1.0], [3.0]], dtype=np.float64),
        selected_indices=np.array([0, 2], dtype=int),
        k=1,
    )
    np.testing.assert_allclose(scores, np.array([1.0, 0.5]), rtol=0.0, atol=2e-12)


def test_timed_selection_rejects_negative_monotonic_duration(monkeypatch) -> None:
    ticks = iter([10.0, 9.0])
    monkeypatch.setattr(diagnostics, "perf_counter", lambda: next(ticks))
    with pytest.raises(RuntimeError, match="negative"):
        timed_selection(lambda: None)


def test_posthoc_class_balance_returns_json_compatible_uniform_summary() -> None:
    result = posthoc_class_balance(
        labels=np.array([0, 0, 1, 2], dtype=int),
        selected_indices=np.array([0, 2, 3], dtype=int),
        num_classes=3,
    )

    assert result == {
        "class_counts": [1, 1, 1],
        "class_proportions": pytest.approx([1.0 / 3.0] * 3),
        "observed_classes": 3,
        "class_coverage_fraction": 1.0,
        "normalized_entropy": pytest.approx(1.0),
    }


def test_posthoc_class_balance_rejects_invalid_labels() -> None:
    with pytest.raises(ValueError, match="outside"):
        posthoc_class_balance(
            labels=np.array([0, 3], dtype=int),
            selected_indices=np.array([0], dtype=int),
            num_classes=3,
        )


@pytest.mark.parametrize(
    ("labels", "selected", "num_classes", "message"),
    [
        ([0], [0], "2", "num_classes"),
        ([0], [0], 0, "num_classes"),
        ([0.0, 1.0], [0], 2, "labels"),
        ([0, 1], [], 2, "selected_indices"),
        ([0, 1], [0.5], 2, "integers"),
        ([0, 1], [0, 0], 2, "unique"),
        ([0, 1], [2], 2, "out-of-range"),
    ],
)
def test_posthoc_class_balance_rejects_malformed_inputs(
    labels, selected, num_classes, message
) -> None:
    with pytest.raises(ValueError, match=message):
        posthoc_class_balance(
            labels=np.asarray(labels),
            selected_indices=np.asarray(selected),
            num_classes=num_classes,
        )


def test_posthoc_class_balance_handles_single_class() -> None:
    result = posthoc_class_balance(
        labels=np.array([0, 0], dtype=int),
        selected_indices=np.array([1], dtype=int),
        num_classes=1,
    )
    assert result["normalized_entropy"] == 1.0
