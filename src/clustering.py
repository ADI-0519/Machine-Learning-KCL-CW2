import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

def cluster_embeddings(embeddings: np.ndarray,n_clusters:int, random_state:int) -> tuple[np.ndarray,np.ndarray]:
    if n_clusters <= 50:
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    else:
        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, batch_size=1024)

    return model.fit_predict(embeddings), model.cluster_centers_