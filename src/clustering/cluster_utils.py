
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from ..config import N_CLUSTERS, LOG_DIR

def cluster_latents():
    Z = np.load(LOG_DIR / 'ticker_latent.npy')
    clust = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward')
    labels = clust.fit_predict(Z)
    np.save(LOG_DIR/'labels.npy', labels)
    return Z, labels
