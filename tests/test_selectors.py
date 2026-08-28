from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from src.selectors import (
    decimal_grid,
    estimate_probcover_delta,
    kcenter_selector,
    probcover_purity,
    probcover_selector,
    tpcrp_ccfl_selector,
)


def _pinned_official_probcover_oracle(fixture: dict) -> tuple[list[int], list[int]]:
    """Small CPU port of the pinned official sparse-graph greedy procedure."""
    embeddings = np.asarray(fixture["embeddings"], dtype=np.float64)
    labeled = np.asarray(fixture["labeled_indices"], dtype=int)
    pool = np.asarray(fixture["pool_indices"], dtype=int)
    relevant = np.concatenate([labeled, pool])
    relevant_embeddings = embeddings[relevant]
    distances = np.linalg.norm(
        relevant_embeddings[:, None, :] - relevant_embeddings[None, :, :],
        axis=2,
    )
    edges = distances < float(fixture["delta"])

    covered = np.any(edges[: len(labeled)], axis=0)
    covered_counts = [int(covered.sum())]
    selected: list[int] = []
    for _ in range(int(fixture["query_size"])):
        degrees = edges[:, ~covered].sum(axis=1)
        local_index = int(np.argmax(degrees))
        selected.append(int(relevant[local_index]))
        covered |= edges[local_index]
        covered_counts.append(int(covered.sum()))
    return selected, covered_counts


def test_kcenter_initial_round_uses_farthest_point_from_global_mean() -> None:
    embeddings = np.array([[0.0], [2.0], [5.0]], dtype=np.float64)

    selected = kcenter_selector(
        embeddings=embeddings,
        pool_indices=np.array([0, 1, 2]),
        labeled_indices=np.array([], dtype=int),
        query_size=2,
    )

    assert selected.tolist() == [2, 0]


def test_kcenter_iterative_round_is_conditioned_on_labeled_set() -> None:
    embeddings = np.array([[0.0], [1.0], [4.0], [10.0]], dtype=np.float64)

    selected = kcenter_selector(
        embeddings=embeddings,
        pool_indices=np.array([1, 2, 3]),
        labeled_indices=np.array([0]),
        query_size=2,
    )

    assert selected.tolist() == [3, 2]


def test_kcenter_uses_lowest_pool_position_to_break_exact_ties() -> None:
    embeddings = np.array([[0.0], [-2.0], [2.0]], dtype=np.float64)

    selected = kcenter_selector(
        embeddings=embeddings,
        pool_indices=np.array([1, 2]),
        labeled_indices=np.array([0]),
        query_size=1,
    )

    assert selected.tolist() == [1]


@pytest.mark.parametrize("query_size", [0, 3])
def test_kcenter_rejects_invalid_query_size(query_size: int) -> None:
    with pytest.raises(ValueError, match="invalid query_size"):
        kcenter_selector(
            embeddings=np.array([[0.0], [1.0]]),
            pool_indices=np.array([0, 1]),
            labeled_indices=np.array([], dtype=int),
            query_size=query_size,
        )


def test_decimal_grid_is_exact_and_inclusive() -> None:
    assert decimal_grid(0.05, 0.11, 0.01).tolist() == [
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.1,
        0.11,
    ]


@pytest.mark.parametrize(
    ("minimum", "maximum", "step"),
    [(0.2, 0.1, 0.01), (0.1, 0.2, 0.0), (0.1, 0.2, -0.01)],
)
def test_decimal_grid_rejects_invalid_ranges(minimum: float, maximum: float, step: float) -> None:
    with pytest.raises(ValueError, match="invalid delta search range"):
        decimal_grid(minimum, maximum, step)


def test_pseudo_label_ball_purity_treats_boundary_as_impure() -> None:
    embeddings = np.array([[0.0], [1.0], [4.0]], dtype=np.float64)
    pseudo_labels = np.array([0, 1, 1], dtype=int)

    purity = probcover_purity(
        embeddings,
        pseudo_labels=pseudo_labels,
        candidates=np.array([0.5, 1.0, 3.0], dtype=np.float64),
    )

    np.testing.assert_allclose(purity, np.array([1.0, 1.0 / 3.0, 1.0 / 3.0]))


def test_probcover_radius_is_largest_candidate_meeting_alpha_and_deterministic() -> None:
    embeddings = np.array([[0.0], [0.1], [5.0], [5.1]], dtype=np.float64)
    candidates = np.array([0.05, 0.2, 1.0], dtype=np.float64)

    first = estimate_probcover_delta(
        embeddings,
        num_classes=2,
        candidates=candidates,
        alpha=0.95,
        clustering_seed=123,
    )
    second = estimate_probcover_delta(
        embeddings,
        num_classes=2,
        candidates=candidates,
        alpha=0.95,
        clustering_seed=123,
    )

    assert first == second == 1.0


def test_probcover_greedy_ties_use_lowest_global_pool_index() -> None:
    embeddings = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=np.float64)

    selected = probcover_selector(
        embeddings=embeddings,
        pool_indices=np.array([3, 2, 1, 0]),
        labeled_indices=np.array([], dtype=int),
        query_size=2,
        delta=0.25,
    )

    assert selected.tolist() == [0, 2]


def test_probcover_matches_pinned_official_fixture_selection_and_coverage() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "probcover_official_fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    embeddings = np.asarray(fixture["embeddings"], dtype=np.float64)

    selected = probcover_selector(
        embeddings=embeddings,
        pool_indices=np.asarray(fixture["pool_indices"], dtype=int),
        labeled_indices=np.asarray(fixture["labeled_indices"], dtype=int),
        query_size=int(fixture["query_size"]),
        delta=float(fixture["delta"]),
    )

    official_selected, official_covered_counts = _pinned_official_probcover_oracle(fixture)
    assert official_selected == fixture["expected_selected_indices"]
    assert official_covered_counts == fixture["expected_covered_counts"]
    assert selected.tolist() == official_selected

    neighborhoods = (
        NearestNeighbors(radius=float(fixture["delta"]), metric="euclidean")
        .fit(embeddings)
        .radius_neighbors(embeddings, return_distance=False)
    )
    covered: set[int] = set()
    covered_counts: list[int] = []
    for index in [*fixture["labeled_indices"], *selected.tolist()]:
        covered.update(int(neighbor) for neighbor in neighborhoods[index])
        covered_counts.append(len(covered))
    assert covered_counts == official_covered_counts


@pytest.mark.parametrize(
    ("pool_indices", "labeled_indices"),
    [([0, 0], []), ([0], [0]), ([2], [])],
)
def test_probcover_rejects_invalid_index_sets(pool_indices, labeled_indices) -> None:
    with pytest.raises(ValueError):
        probcover_selector(
            embeddings=np.array([[0.0], [1.0]]),
            pool_indices=np.asarray(pool_indices, dtype=int),
            labeled_indices=np.asarray(labeled_indices, dtype=int),
            query_size=1,
            delta=0.5,
        )


def _ccfl_fixture_kwargs() -> dict:
    return {
        "embeddings": np.array(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
            dtype=np.float64,
        ),
        "cluster_labels": np.array([0, 0, 1, 1], dtype=int),
        "centroids": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        "selected_cluster_ids": [0, 1],
        "pool_indices": np.array([0, 1, 2, 3], dtype=int),
        "knn_k": 1,
        "candidates_per_cluster": 2,
        "min_cluster_size": 1,
    }


def test_ccfl_refine_steps_zero_is_exact_candidate_only_output(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.selectors.compute_typicality_scores",
        lambda _embeddings, _knn_k: np.array([2.0, 1.0]),
    )

    selected = tpcrp_ccfl_selector(
        **_ccfl_fixture_kwargs(),
        refine_steps=0,
        use_cluster_weights=False,
        cluster_sizes=np.array([100, 1]),
        rng=np.random.default_rng(7),
    )

    assert selected.tolist() == [0, 2]


def test_ccfl_disabled_weights_are_exactly_unit_weights(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.selectors.compute_typicality_scores",
        lambda _embeddings, _knn_k: np.array([2.0, 1.0]),
    )
    kwargs = _ccfl_fixture_kwargs()

    unweighted = tpcrp_ccfl_selector(
        **kwargs,
        refine_steps=1,
        use_cluster_weights=False,
        cluster_sizes=np.array([100, 1]),
        rng=np.random.default_rng(7),
    )
    explicit_units = tpcrp_ccfl_selector(
        **kwargs,
        refine_steps=1,
        use_cluster_weights=True,
        cluster_sizes=np.ones(2, dtype=int),
        rng=np.random.default_rng(7),
    )

    np.testing.assert_array_equal(unweighted, explicit_units)


def test_ccfl_scientific_parameters_have_no_function_defaults() -> None:
    signature = inspect.signature(tpcrp_ccfl_selector)
    for name in (
        "candidates_per_cluster",
        "refine_steps",
        "use_cluster_weights",
        "cluster_sizes",
        "min_cluster_size",
        "rng",
    ):
        assert signature.parameters[name].default is inspect.Parameter.empty
