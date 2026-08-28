from decimal import Decimal
from numbers import Integral, Real

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from .typicality import compute_cluster_aware_scores, compute_typicality_scores


def random_selector(num_samples: int, budget: int, rng: np.random.Generator) -> np.ndarray:
    """Randomly sample unique pool indices for the query batch."""
    selected = rng.choice(num_samples, size=budget, replace=False)
    if len(np.unique(selected)) != len(selected):
        raise RuntimeError("random selector produced duplicate indices")
    return np.sort(selected)


def tpcrand_selector(
    cluster_labels: np.ndarray, budget: int, rng: np.random.Generator
) -> np.ndarray:
    """Select one random sample per cluster for TPCRand ablation."""
    selected = []
    for cluster_id in range(budget):
        members = np.where(cluster_labels == cluster_id)[0]
        if len(members) == 0:
            continue
        selected.append(rng.choice(members))

    if len(np.unique(selected)) != len(selected):
        raise RuntimeError("TPCRand produced duplicate indices")
    return np.array(sorted(selected), dtype=int)


def tpcrp_selector(
    embeddings: np.ndarray, cluster_labels: np.ndarray, budget: int, knn_k: int
) -> np.ndarray:
    """Select the most typical sample in each cluster (TPCRP)."""
    selected = []

    for cluster_id in range(budget):
        members = np.where(cluster_labels == cluster_id)[0]
        if len(members) == 0:
            continue

        cluster_emb = embeddings[members]
        scores = compute_typicality_scores(cluster_emb, knn_k)
        best_local_idx = int(np.argmax(scores))
        selected.append(members[best_local_idx])

    if len(np.unique(selected)) != len(selected):
        raise RuntimeError("TPCRP produced duplicate indices")
    return np.array(sorted(selected), dtype=int)


def tpcrp_modified_selector(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    centroids: np.ndarray,
    budget: int,
    knn_k: int,
    alpha: float,
) -> np.ndarray:
    """Select per-cluster samples using cluster-aware typicality scoring"""
    selected = []

    for cluster_id in range(budget):
        members = np.where(cluster_labels == cluster_id)[0]
        if len(members) == 0:
            continue

        cluster_emb = embeddings[members]
        scores = compute_cluster_aware_scores(
            cluster_embeddings=cluster_emb,
            centroid=centroids[cluster_id],
            k=knn_k,
            alpha=alpha,
        )
        best_local_idx = int(np.argmax(scores))
        selected.append(members[best_local_idx])

    if len(np.unique(selected)) != len(selected):
        raise RuntimeError("modified TPCRP produced duplicate indices")
    return np.array(sorted(selected), dtype=int)


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between two embedding sets"""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_n @ b_n.T


def tpcrp_ccfl_selector(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    centroids: np.ndarray,
    selected_cluster_ids: list[int],
    pool_indices: np.ndarray,
    knn_k: int,
    *,
    candidates_per_cluster: int,
    refine_steps: int,
    use_cluster_weights: bool,
    cluster_sizes: np.ndarray | None,
    min_cluster_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    TPCRP-CCFL:
    1) choose top-typical candidates per selected cluster
    2) initialize with most typical candidate in each cluster
    3) refine globally with a light facility-location objective over selected centroids
    """
    values = _validated_embeddings(embeddings)
    labels = np.asarray(cluster_labels)
    centroid_values = np.asarray(centroids)
    pool = np.asarray(pool_indices)
    if labels.ndim != 1 or len(labels) != len(values):
        raise ValueError("cluster_labels must contain one label per embedding")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("cluster_labels must contain integers")
    if (
        centroid_values.ndim != 2
        or centroid_values.shape[1] != values.shape[1]
        or not np.isfinite(centroid_values).all()
    ):
        raise ValueError("centroids must be a finite matrix matching embedding dimensions")
    if pool.ndim != 1 or not np.issubdtype(pool.dtype, np.integer):
        raise ValueError("pool_indices must be a one-dimensional integer vector")
    pool = pool.astype(int, copy=False)
    if len(np.unique(pool)) != len(pool) or np.any(pool < 0) or np.any(pool >= len(values)):
        raise ValueError("pool_indices must contain unique in-range global indices")
    if not selected_cluster_ids or len(set(selected_cluster_ids)) != len(selected_cluster_ids):
        raise ValueError("selected_cluster_ids must be non-empty and unique")
    if any(
        not isinstance(cluster_id, Integral)
        or isinstance(cluster_id, (bool, np.bool_))
        or cluster_id < 0
        or cluster_id >= len(centroid_values)
        for cluster_id in selected_cluster_ids
    ):
        raise ValueError("selected_cluster_ids contain an invalid cluster")
    if not isinstance(knn_k, Integral) or isinstance(knn_k, (bool, np.bool_)) or knn_k <= 0:
        raise ValueError("knn_k must be a positive integer")
    if (
        not isinstance(candidates_per_cluster, Integral)
        or isinstance(candidates_per_cluster, (bool, np.bool_))
        or candidates_per_cluster <= 0
    ):
        raise ValueError("candidates_per_cluster must be a positive integer")
    if (
        not isinstance(refine_steps, Integral)
        or isinstance(refine_steps, (bool, np.bool_))
        or refine_steps < 0
    ):
        raise ValueError("refine_steps must be a non-negative integer")
    if not isinstance(use_cluster_weights, bool):
        raise ValueError("use_cluster_weights must be boolean")
    if (
        not isinstance(min_cluster_size, Integral)
        or isinstance(min_cluster_size, (bool, np.bool_))
        or min_cluster_size <= 0
    ):
        raise ValueError("min_cluster_size must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit numpy Generator")

    pool_set = set(pool.tolist())
    candidate_sets: list[np.ndarray] = []
    valid_clusters: list[int] = []
    current: list[int] = []

    for cluster_id in selected_cluster_ids:
        members = np.where(cluster_labels == cluster_id)[0]
        if len(members) == 0:
            continue
        candidate_members = np.array([idx for idx in members if idx in pool_set], dtype=int)
        if len(candidate_members) == 0:
            continue

        if len(members) < min_cluster_size:
            candidates = np.array([int(rng.choice(candidate_members))], dtype=int)
        else:
            cluster_emb = embeddings[members]
            scores = compute_typicality_scores(cluster_emb, knn_k)
            member_to_local = {m: i for i, m in enumerate(members.tolist())}
            candidate_locals = np.array(
                [member_to_local[m] for m in candidate_members.tolist()], dtype=int
            )
            ranking = np.argsort(-scores[candidate_locals], kind="stable")
            ranked = candidate_members[ranking]
            keep = max(1, min(candidates_per_cluster, len(ranked)))
            candidates = ranked[:keep].astype(int)

        candidate_sets.append(candidates)
        valid_clusters.append(cluster_id)
        current.append(int(candidates[0]))

    if not current:
        return np.array([], dtype=int)

    target_centroids = centroid_values[np.array(valid_clusters, dtype=int)]
    if use_cluster_weights:
        if cluster_sizes is None:
            raise ValueError("cluster_sizes are required when use_cluster_weights is true")
        size_values = np.asarray(cluster_sizes)
        if (
            size_values.ndim != 1
            or len(size_values) != len(centroid_values)
            or not np.issubdtype(size_values.dtype, np.integer)
            or np.any(size_values < 0)
        ):
            raise ValueError("cluster_sizes must be a non-negative integer per centroid")
        weights = size_values[np.array(valid_clusters, dtype=int)].astype(float)
    else:
        weights = np.ones(len(valid_clusters), dtype=float)
    weights = np.maximum(weights, 1e-12)

    def objective(chosen: list[int]) -> float:
        facilities = values[np.array(chosen, dtype=int)]
        sim = _cosine_similarity_matrix(target_centroids, facilities)
        sim = np.clip((sim + 1.0) * 0.5, 0.0, 1.0)
        best = sim.max(axis=1)
        return float((weights * best).sum())

    best_score = objective(current)
    for _ in range(refine_steps):
        improved = False
        for i, candidates in enumerate(candidate_sets):
            base_pick = current[i]
            local_best_pick = base_pick
            local_best_score = best_score
            for cand in candidates:
                cand = int(cand)
                if cand == base_pick:
                    continue
                trial = current.copy()
                trial[i] = cand
                score = objective(trial)
                if score > local_best_score + 1e-12:
                    local_best_score = score
                    local_best_pick = cand
            if local_best_pick != base_pick:
                current[i] = local_best_pick
                best_score = local_best_score
                improved = True
        if not improved:
            break

    return np.array(current, dtype=int)


def tpcinv_selector(
    embeddings: np.ndarray, cluster_labels: np.ndarray, budget: int, knn_k: int
) -> np.ndarray:
    """
    Paper ablation: choose the most atypical point in each cluster.
    Equivalent to selecting the minimum-typicality point per cluster.
    """
    selected: list[int] = []

    for cluster_id in range(budget):
        members = np.where(cluster_labels == cluster_id)[0]
        if len(members) == 0:
            continue

        cluster_emb = embeddings[members]
        scores = compute_typicality_scores(cluster_emb, knn_k)
        worst_local_idx = int(np.argmin(scores))
        selected.append(int(members[worst_local_idx]))

    return np.array(sorted(selected), dtype=int)


def tpcnoclust_selector(embeddings: np.ndarray, budget: int, knn_k: int) -> np.ndarray:
    """
    Paper ablation: choose globally most typical points without clustering.
    This removes the diversity component.
    """
    scores = compute_typicality_scores(embeddings, knn_k)
    selected = np.argsort(scores)[-budget:]
    return np.sort(selected.astype(int))


def _validated_embeddings(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings)
    if values.ndim != 2 or len(values) == 0 or values.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty two-dimensional array")
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise ValueError("embeddings must contain only finite numeric values")
    return values


def _validated_index_sets(
    *,
    num_samples: int,
    pool_indices: np.ndarray,
    labeled_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pool = np.asarray(pool_indices)
    labeled = np.asarray(labeled_indices)
    if pool.ndim != 1 or labeled.ndim != 1:
        raise ValueError("pool_indices and labeled_indices must be one-dimensional")
    if (len(pool) and not np.issubdtype(pool.dtype, np.integer)) or (
        len(labeled) and not np.issubdtype(labeled.dtype, np.integer)
    ):
        raise ValueError("pool_indices and labeled_indices must contain integers")
    pool = pool.astype(int, copy=False)
    labeled = labeled.astype(int, copy=False)
    if len(np.unique(pool)) != len(pool) or len(np.unique(labeled)) != len(labeled):
        raise ValueError("pool_indices and labeled_indices must not contain duplicates")
    if np.any(pool < 0) or np.any(pool >= num_samples):
        raise ValueError("pool_indices contain an out-of-range index")
    if np.any(labeled < 0) or np.any(labeled >= num_samples):
        raise ValueError("labeled_indices contain an out-of-range index")
    if np.intersect1d(pool, labeled).size:
        raise ValueError("pool_indices and labeled_indices must be disjoint")
    return pool, labeled


def _validate_query_size(query_size: int, pool_size: int) -> int:
    if (
        not isinstance(query_size, Integral)
        or isinstance(query_size, (bool, np.bool_))
        or query_size <= 0
        or query_size > pool_size
    ):
        raise ValueError("invalid query_size")
    return int(query_size)


def kcenter_selector(
    embeddings: np.ndarray,
    pool_indices: np.ndarray,
    labeled_indices: np.ndarray,
    query_size: int,
) -> np.ndarray:
    """Select a global-index CoreSet batch by iterative farthest-first traversal."""
    values = _validated_embeddings(embeddings)
    pool, labeled = _validated_index_sets(
        num_samples=len(values),
        pool_indices=pool_indices,
        labeled_indices=labeled_indices,
    )
    query_size = _validate_query_size(query_size, len(pool))

    pool_embeddings = values[pool]
    if len(labeled):
        anchors = values[labeled]
        min_d2 = np.full(len(pool), np.inf, dtype=np.float64)
        for anchor in anchors:
            anchor_d2 = np.sum((pool_embeddings - anchor) ** 2, axis=1)
            min_d2 = np.minimum(min_d2, anchor_d2)
    else:
        center = values.mean(axis=0, keepdims=True)
        min_d2 = np.sum((pool_embeddings - center) ** 2, axis=1)

    chosen_local: list[int] = []
    available = np.ones(len(pool), dtype=bool)
    for _ in range(query_size):
        scores = np.where(available, min_d2, -np.inf)
        next_local = int(np.argmax(scores))
        chosen_local.append(next_local)
        available[next_local] = False
        new_d2 = np.sum(
            (pool_embeddings - pool_embeddings[next_local]) ** 2,
            axis=1,
        )
        min_d2 = np.minimum(min_d2, new_d2)

    return pool[np.asarray(chosen_local, dtype=int)]


def decimal_grid(minimum: float, maximum: float, step: float) -> np.ndarray:
    """Build an inclusive decimal radius grid without binary-float drift."""
    low = Decimal(str(minimum))
    high = Decimal(str(maximum))
    increment = Decimal(str(step))
    if not low.is_finite() or not high.is_finite() or not increment.is_finite():
        raise ValueError("invalid delta search range")
    if increment <= 0 or high < low:
        raise ValueError("invalid delta search range")

    values: list[float] = []
    current = low
    while current <= high:
        values.append(float(current))
        current += increment
    return np.asarray(values, dtype=np.float64)


def _validated_probcover_candidates(candidates: np.ndarray) -> np.ndarray:
    values = np.asarray(candidates, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("candidates must be a non-empty vector")
    if not np.all(np.isfinite(values)) or values[0] <= 0 or np.any(np.diff(values) <= 0):
        raise ValueError("candidates must be strictly increasing and positive")
    return values


def probcover_purity(
    embeddings: np.ndarray,
    *,
    pseudo_labels: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """Return the fraction of pseudo-label-pure balls for every radius candidate."""
    values = _validated_embeddings(embeddings)
    labels = np.asarray(pseudo_labels)
    radii = _validated_probcover_candidates(candidates)
    if labels.ndim != 1 or len(labels) != len(values):
        raise ValueError("pseudo_labels must contain one label per embedding")

    neighbors = NearestNeighbors(
        radius=float(radii[-1]),
        metric="euclidean",
    ).fit(values)
    distances, indices = neighbors.radius_neighbors(
        values,
        return_distance=True,
        sort_results=True,
    )

    first_impure = np.full(len(values), np.inf, dtype=np.float64)
    for center, (center_distances, center_indices) in enumerate(
        zip(distances, indices, strict=True)
    ):
        different = labels[center_indices] != labels[center]
        if np.any(different):
            first_different = int(np.flatnonzero(different)[0])
            first_impure[center] = float(center_distances[first_different])

    return np.asarray(
        [np.mean(first_impure > radius) for radius in radii],
        dtype=np.float64,
    )


def estimate_probcover_delta(
    embeddings: np.ndarray,
    *,
    num_classes: int,
    candidates: np.ndarray,
    alpha: float,
    clustering_seed: int,
) -> float:
    """Estimate ProbCover's radius using only deterministic K-Means pseudo-labels."""
    values = _validated_embeddings(embeddings)
    radii = _validated_probcover_candidates(candidates)
    if (
        not isinstance(alpha, Real)
        or isinstance(alpha, (bool, np.bool_))
        or not np.isfinite(alpha)
        or not 0.0 < alpha <= 1.0
    ):
        raise ValueError("alpha must be in (0, 1]")
    if not isinstance(num_classes, Integral) or isinstance(num_classes, (bool, np.bool_)):
        raise ValueError("num_classes must be an integer")
    if num_classes <= 0 or num_classes > len(values):
        raise ValueError("num_classes must be between 1 and the number of embeddings")
    if not isinstance(clustering_seed, Integral) or isinstance(clustering_seed, (bool, np.bool_)):
        raise ValueError("clustering_seed must be an integer")

    pseudo_labels = KMeans(
        n_clusters=int(num_classes),
        random_state=int(clustering_seed),
        n_init=10,
    ).fit_predict(values)
    purity = probcover_purity(
        values,
        pseudo_labels=pseudo_labels,
        candidates=radii,
    )
    valid = np.flatnonzero(purity >= alpha)
    if not len(valid):
        raise ValueError("no ProbCover radius satisfies the configured purity threshold")
    return float(radii[valid[-1]])


def probcover_selector(
    embeddings: np.ndarray,
    pool_indices: np.ndarray,
    labeled_indices: np.ndarray,
    query_size: int,
    *,
    delta: float,
) -> np.ndarray:
    """Greedily select pool-only global indices that maximize uncovered points."""
    values = _validated_embeddings(embeddings)
    pool, labeled = _validated_index_sets(
        num_samples=len(values),
        pool_indices=pool_indices,
        labeled_indices=labeled_indices,
    )
    pool = np.sort(pool)
    query_size = _validate_query_size(query_size, len(pool))
    if (
        not isinstance(delta, Real)
        or isinstance(delta, (bool, np.bool_))
        or not np.isfinite(delta)
        or float(delta) <= 0
    ):
        raise ValueError("delta must be a finite positive number")

    graph = NearestNeighbors(radius=float(delta), metric="euclidean").fit(values)
    neighborhoods = graph.radius_neighbors(values, return_distance=False)
    uncovered = np.ones(len(values), dtype=bool)
    for index in labeled:
        uncovered[neighborhoods[index]] = False

    available = np.ones(len(pool), dtype=bool)
    selected: list[int] = []
    for _ in range(query_size):
        gains = np.asarray(
            [
                uncovered[neighborhoods[index]].sum() if available[position] else -1
                for position, index in enumerate(pool)
            ],
            dtype=np.int64,
        )
        position = int(np.argmax(gains))
        chosen = int(pool[position])
        selected.append(chosen)
        available[position] = False
        uncovered[neighborhoods[chosen]] = False

    return np.asarray(selected, dtype=int)
