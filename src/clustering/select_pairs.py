
import numpy as np
from scipy.spatial.distance import cdist
from ..config import PAIRS_PER_CLUST, LOG_DIR

def select_pairs():
    Z = np.load(LOG_DIR/'latent.npy')
    labels = np.load(LOG_DIR/'labels.npy')
    pairs=[]
    for c in set(labels):
        idx = np.where(labels==c)[0]
        dists = cdist(Z[idx], Z[idx], metric='euclidean')
        triu = np.triu_indices_from(dists, k=1)
        sorted_pairs = sorted(zip(triu[0],triu[1],dists[triu]), key=lambda x:x[2])
        top = sorted_pairs[:PAIRS_PER_CLUST]
        pairs.extend([(idx[i], idx[j]) for i,j,_ in top])
    np.save(LOG_DIR/'pairs.npy', pairs)
    return pairs
