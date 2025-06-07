
import json
import numpy as np
from scipy.spatial.distance import cdist
from ..config import PAIRS_PER_CLUST, LOG_DIR

def select_pairs():
    # Load standardized latent vectors for consistent distance calculations
    Z = np.load(LOG_DIR / 'ticker_latent_scaled.npy')
    labels = np.load(LOG_DIR / 'labels.npy')
    with open(LOG_DIR / 'ticker_index.json') as f:
        tickers = json.load(f)

    pairs = []
    for c in set(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            print(f"Cluster {c} has <2 tickers; skipped.")
            continue

        dists = cdist(Z[idx], Z[idx], metric='euclidean')
        triu = np.triu_indices_from(dists, k=1)
        sorted_pairs = sorted(zip(triu[0], triu[1], dists[triu]), key=lambda x: x[2])
        top = sorted_pairs[:min(PAIRS_PER_CLUST, len(sorted_pairs))]
        pairs.extend([(tickers[idx[i]], tickers[idx[j]]) for i, j, _ in top])

    # Save as numpy array of strings for later steps
    np.save(LOG_DIR / 'pairs.npy', np.array(pairs, dtype=object))
    return pairs
