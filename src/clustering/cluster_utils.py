
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from ..config import N_CLUSTERS, LOG_DIR

def cluster_latents(Z: np.ndarray | None = None, n_clusters: int = N_CLUSTERS, save: bool = True):
    """Cluster latent vectors and optionally save the labels."""
    save_output = Z is None
    if Z is None:
        Z = np.load(LOG_DIR / 'ticker_latent.npy')
    Z = StandardScaler().fit_transform(Z)
    clust = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = clust.fit_predict(Z)
    if save and save_output:
        np.save(LOG_DIR / 'labels.npy', labels)
    return Z, labels
