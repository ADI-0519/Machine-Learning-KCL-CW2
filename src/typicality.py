from numbers import Integral, Real

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _validated_embeddings(embeddings: np.ndarray, *, minimum_samples: int = 1) -> np.ndarray:
    values = np.asarray(embeddings)
    if values.ndim != 2 or len(values) < minimum_samples or values.shape[1] == 0:
        raise ValueError(
            f"embeddings must contain at least {minimum_samples} non-empty feature vectors"
        )
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("embeddings must contain only finite numeric values")
    return values


def _validated_k(k: int) -> int:
    if not isinstance(k, Integral) or isinstance(k, (bool, np.bool_)) or k <= 0:
        raise ValueError("k must be a positive integer")
    return int(k)


def compute_typicality_scores(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Compute inverse average KNN distance as a typicality score."""
    values = _validated_embeddings(embeddings)
    k = _validated_k(k)
    k_eff = min(k + 1, len(values))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(values)
    distances, _ = nn.kneighbors(values)

    neighbor_distances = distances[:, 1:] if k_eff > 1 else distances
    avg_dist = neighbor_distances.mean(axis=1)
    return 1.0 / (avg_dist + 1e-12)


def compute_selected_typicality_scores(
    embeddings: np.ndarray,
    selected_indices: np.ndarray,
    k: int,
) -> np.ndarray:
    """Compute exact full-representation k-NN typicality only for selected points."""
    values = _validated_embeddings(embeddings, minimum_samples=2)
    selected = np.asarray(selected_indices)
    if selected.ndim != 1 or not len(selected) or not np.issubdtype(selected.dtype, np.integer):
        raise ValueError("selected_indices must be a non-empty integer vector")
    selected = selected.astype(int, copy=False)
    if len(np.unique(selected)) != len(selected):
        raise ValueError("selected_indices must be unique")
    if np.any(selected < 0) or np.any(selected >= len(values)):
        raise ValueError("selected_indices contain an out-of-range index")
    k = _validated_k(k)

    neighbors_to_request = min(k + 1, len(values))
    model = NearestNeighbors(n_neighbors=neighbors_to_request, metric="euclidean").fit(values)
    distances, indices = model.kneighbors(values[selected])
    scores = np.empty(len(selected), dtype=np.float64)
    for row, global_index in enumerate(selected):
        nonself_distances = distances[row][indices[row] != global_index]
        used_distances = nonself_distances[: min(k, len(nonself_distances))]
        if not len(used_distances):
            raise ValueError("typicality requires at least one non-self neighbor")
        scores[row] = 1.0 / (float(used_distances.mean()) + 1e-12)
    return scores


def compute_cluster_aware_scores(
    cluster_embeddings: np.ndarray, centroid: np.ndarray, k: int, alpha: float
) -> np.ndarray:
    """Blend typicality with centroid proximity for cluster-aware ranking."""
    values = _validated_embeddings(cluster_embeddings)
    center = np.asarray(centroid)
    if center.ndim != 1 or center.shape[0] != values.shape[1] or not np.isfinite(center).all():
        raise ValueError("centroid must be a finite vector matching the embedding width")
    if (
        not isinstance(alpha, Real)
        or isinstance(alpha, (bool, np.bool_))
        or not np.isfinite(alpha)
        or not 0.0 <= float(alpha) <= 1.0
    ):
        raise ValueError("alpha must be in [0, 1]")
    typ = compute_typicality_scores(values, k)

    dists = np.linalg.norm(values - center[None, :], axis=1)
    centrality = 1.0 / (dists + 1e-12)

    def normalize(x: np.ndarray) -> np.ndarray:
        if np.allclose(x.max(), x.min()):
            return np.ones_like(x)
        return (x - x.min()) / (x.max() - x.min())

    typ_n = normalize(typ)
    cen_n = normalize(centrality)
    return alpha * typ_n + (1.0 - alpha) * cen_n
