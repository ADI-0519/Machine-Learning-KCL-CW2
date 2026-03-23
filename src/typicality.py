import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_typicality_scores(embeddings: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(k + 1, len(embeddings))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)

    neighbor_distances = distances[:, 1:] if k_eff > 1 else distances
    avg_dist = neighbor_distances.mean(axis=1)
    return 1.0 / (avg_dist + 1e-12)


def compute_cluster_aware_scores(
    cluster_embeddings: np.ndarray,
    centroid: np.ndarray,
    k: int,
    alpha: float,
) -> np.ndarray:
    typ = compute_typicality_scores(cluster_embeddings, k)

    dists = np.linalg.norm(cluster_embeddings - centroid[None, :], axis=1)
    centrality = 1.0 / (dists + 1e-12)

    def normalize(x: np.ndarray) -> np.ndarray:
        if np.allclose(x.max(), x.min()):
            return np.ones_like(x)
        return (x - x.min()) / (x.max() - x.min())

    typ_n = normalize(typ)
    cen_n = normalize(centrality)
    return alpha * typ_n + (1.0 - alpha) * cen_n