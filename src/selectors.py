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