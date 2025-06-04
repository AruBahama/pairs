
import numpy as np, json
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from ..config import N_CLUSTERS, LOG_DIR

def cluster_latents():
    Z = np.load(LOG_DIR/'latent.npy')
    clust = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward')
    labels = clust.fit_predict(Z)
    np.save(LOG_DIR/'labels.npy', labels)
    return Z, labels
