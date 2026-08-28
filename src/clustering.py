import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


def cluster_embeddings(
    embeddings: np.ndarray, n_clusters: int, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster finite embeddings and return integer labels plus centroids."""
    values = np.asarray(embeddings)
    if values.ndim != 2 or len(values) == 0 or values.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty two-dimensional array")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("embeddings must contain only finite numeric values")
    if not isinstance(n_clusters, int) or isinstance(n_clusters, bool):
        raise ValueError("n_clusters must be a positive integer")
    if n_clusters <= 0 or n_clusters > len(values):
        raise ValueError("n_clusters must be between 1 and the number of embeddings")
    if n_clusters <= 50:
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    else:
        model = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=10, batch_size=1024
        )

    return model.fit_predict(values), model.cluster_centers_
