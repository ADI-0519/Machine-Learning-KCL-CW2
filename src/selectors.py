import numpy as np

from .typicality import compute_typicality_scores, compute_cluster_aware_scores


def random_selector(num_samples: int, budget: int, rng: np.random.Generator) -> np.ndarray:
    selected = rng.choice(num_samples, size=budget, replace=False)
    assert len(np.unique(selected)) == len(selected)
    return np.sort(selected)
    


def tpcrand_selector(cluster_labels: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
    selected = []
    for cluster_id in range(budget):
        members = np.where(cluster_labels == cluster_id)[0]
        if len(members) == 0:
            continue
        selected.append(rng.choice(members))

    assert len(np.unique(selected)) == len(selected)
    return np.array(sorted(selected), dtype=int)


def tpcrp_selector(embeddings: np.ndarray,cluster_labels: np.ndarray,budget: int,knn_k: int) -> np.ndarray:
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