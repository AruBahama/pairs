import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from statsmodels.tsa.stattools import adfuller
from ..config import PAIRS_PER_CLUST, LOG_DIR, PROC_DIR


def _load_prices(ticker: str) -> pd.Series:
    """Return a price series for ``ticker`` from ``PROC_DIR``."""
    df = pd.read_parquet(PROC_DIR / f"{ticker}.parquet")
    for col in ["Close", "close", "Adj Close", "adjclose"]:
        if col in df.columns:
            return df[col]
    return df.select_dtypes(include="number").iloc[:, 0]


def _is_cointegrated(t1: str, t2: str, alpha: float = 0.05) -> bool:
    """Return ``True`` if the ADF p-value of ``t1``-``t2`` is below ``alpha``."""
    try:
        price1 = _load_prices(t1)
        price2 = _load_prices(t2)
        pval = adfuller(price1 - price2)[1]
    except Exception:
        return False
    return pval < alpha

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

        dists = cdist(Z[idx], Z[idx], metric="euclidean")
        triu = np.triu_indices_from(dists, k=1)
        sorted_pairs = sorted(zip(triu[0], triu[1], dists[triu]), key=lambda x: x[2])

        count = 0
        for i, j, _ in sorted_pairs:
            if count >= PAIRS_PER_CLUST:
                break
            t1, t2 = tickers[idx[i]], tickers[idx[j]]
            if _is_cointegrated(t1, t2):
                pairs.append((t1, t2))
                count += 1

    # Save as numpy array of strings for later steps
    np.save(LOG_DIR / 'pairs.npy', np.array(pairs, dtype=object))
    return pairs
