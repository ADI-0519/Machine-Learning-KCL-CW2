import numpy as np

from .typicality import compute_typicality_scores, compute_cluster_aware_scores


def random_selector(num_samples: int, budget: int, rng: np.random.Generator) -> np.ndarray:
    """Randomly sample unique pool indices for the query batch."""
    selected = rng.choice(num_samples, size=budget, replace=False)
    assert len(np.unique(selected)) == len(selected)
    return np.sort(selected)
    


def tpcrand_selector(cluster_labels: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
    """Select one random sample per cluster for TPCRand ablation."""
    selected = []
    for cluster_id in range(budget):
        members = np.where(cluster_labels == cluster_id)[0]
        if len(members) == 0:
            continue
        selected.append(rng.choice(members))

    assert len(np.unique(selected)) == len(selected)
    return np.array(sorted(selected), dtype=int)


def tpcrp_selector(embeddings: np.ndarray,cluster_labels: np.ndarray,budget: int,knn_k: int) -> np.ndarray:
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

    assert len(np.unique(selected)) == len(selected)
    return np.array(sorted(selected), dtype=int)


def tpcrp_modified_selector(embeddings: np.ndarray,cluster_labels: np.ndarray,centroids: np.ndarray,budget: int,knn_k: int,alpha: float) -> np.ndarray:
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

    assert len(np.unique(selected)) == len(selected)
    return np.array(sorted(selected), dtype=int)


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between two embedding sets"""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_n @ b_n.T


def tpcrp_ccfl_selector(embeddings: np.ndarray,cluster_labels: np.ndarray,centroids: np.ndarray,selected_cluster_ids: list[int],pool_indices: np.ndarray,knn_k: int,candidates_per_cluster: int = 5,refine_steps: int = 1,cluster_sizes: np.ndarray | None = None,min_cluster_size: int = 5,rng: np.random.Generator | None = None) -> np.ndarray:
    """
    TPCRP-CCFL:
    1) choose top-typical candidates per selected cluster
    2) initialize with most typical candidate in each cluster
    3) refine globally with a light facility-location objective over selected centroids
    """
    if rng is None:
        rng = np.random.default_rng()

    pool_set = set(pool_indices.tolist())
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
            candidate_locals = np.array([member_to_local[m] for m in candidate_members.tolist()], dtype=int)
            ranked = candidate_members[np.argsort(scores[candidate_locals])[::-1]]
            keep = max(1, min(candidates_per_cluster, len(ranked)))
            candidates = ranked[:keep].astype(int)

        candidate_sets.append(candidates)
        valid_clusters.append(cluster_id)
        current.append(int(candidates[0]))

    if not current:
        return np.array([], dtype=int)

    target_centroids = centroids[np.array(valid_clusters, dtype=int)]
    if cluster_sizes is not None:
        weights = cluster_sizes[np.array(valid_clusters, dtype=int)].astype(float)
    else:
        weights = np.ones(len(valid_clusters), dtype=float)
    weights = np.maximum(weights, 1e-12)

    def objective(chosen: list[int]) -> float:
        facilities = embeddings[np.array(chosen, dtype=int)]
        sim = _cosine_similarity_matrix(target_centroids, facilities)
        sim = np.clip((sim + 1.0) * 0.5, 0.0, 1.0)
        best = sim.max(axis=1)
        return float((weights * best).sum())

    best_score = objective(current)
    for _ in range(max(0, refine_steps)):
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

def tpcinv_selector(embeddings: np.ndarray,cluster_labels: np.ndarray,budget: int,knn_k: int) -> np.ndarray:
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


def tpcnoclust_selector(embeddings: np.ndarray,budget: int,knn_k: int) -> np.ndarray:
    """
    Paper ablation: choose globally most typical points without clustering.
    This removes the diversity component.
    """
    scores = compute_typicality_scores(embeddings, knn_k)
    selected = np.argsort(scores)[-budget:]
    return np.sort(selected.astype(int))


def _pairwise_squared_distances_to_set(embeddings: np.ndarray,selected_indices: list[int]) -> np.ndarray:
    """
    Return, for every point, the squared distance to its nearest selected point.
    """
    if not selected_indices:
        return np.full(len(embeddings), np.inf, dtype=np.float64)

    selected_emb = embeddings[selected_indices]
    # Shape: (n_points, n_selected)
    dists = (
        np.sum(embeddings**2, axis=1, keepdims=True)
        - 2.0 * embeddings @ selected_emb.T
        + np.sum(selected_emb**2, axis=1)[None, :]
    )
    return np.min(dists, axis=1)


def kcenter_selector(embeddings: np.ndarray,budget: int) -> np.ndarray:
    """
    CoreSet-style k-center greedy / farthest-first traversal in embedding space.
    Starts from the point closest to the global mean, then greedily adds the
    point farthest from the current selected set.
    """
    n = len(embeddings)
    if budget <= 0:
        return np.array([], dtype=int)
    if budget >= n:
        return np.arange(n, dtype=int)

    mean_emb = embeddings.mean(axis=0, keepdims=True)
    dists_to_mean = np.sum((embeddings - mean_emb) ** 2, axis=1)
    first_idx = int(np.argmin(dists_to_mean))

    selected: list[int] = [first_idx]
    min_dists = _pairwise_squared_distances_to_set(embeddings, selected)

    for _ in range(1, budget):
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)

        new_dists = np.sum((embeddings - embeddings[next_idx]) ** 2, axis=1)
        min_dists = np.minimum(min_dists, new_dists)

    return np.array(sorted(selected), dtype=int)
