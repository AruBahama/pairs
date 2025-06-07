import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from statsmodels.tsa.stattools import adfuller

from ..config import PAIRS_PER_CLUST, LOG_DIR, PROC_DIR


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _load_prices(ticker: str) -> pd.Series:
    """Return the price series for *ticker* (tries Close/Adj Close first)."""
    df = pd.read_parquet(Path(PROC_DIR) / f"{ticker}.parquet")

    for col in ("Close", "close", "Adj Close", "adjclose"):
        if col in df.columns:
            return df[col]

    # fallback: first numeric column
    return df.select_dtypes(include="number").iloc[:, 0]


def _is_cointegrated(t1: str, t2: str, alpha: float = 0.05) -> bool:
    """ADF test on (price1 – price2); returns True if p-value < *alpha*."""
    try:
        p1 = _load_prices(t1)
        p2 = _load_prices(t2)
        p_val = adfuller(p1 - p2)[1]
    except Exception:
        return False
    return p_val < alpha


# ---------------------------------------------------------------------- #
# Main selection routine
# ---------------------------------------------------------------------- #
def select_pairs():
    """Pick the closest, cointegrated pairs in each cluster."""
    # latent vectors & labels
    Z = np.load(Path(LOG_DIR) / "ticker_latent_scaled.npy")
    labels = np.load(Path(LOG_DIR) / "labels.npy")
    with open(Path(LOG_DIR) / "ticker_index.json") as f:
        tickers = json.load(f)

    pairs = []

    for c in sorted(set(labels)):
        # indices belonging to this cluster
        idx = np.where(labels == c)[0]

        if len(idx) < 2:
            print(f"[cluster {c}] <2 tickers – skipped")
            continue

        # pairwise Euclidean distances inside the cluster
        dists = cdist(Z[idx], Z[idx], metric="euclidean")
        i_upper = np.triu_indices_from(dists, k=1)
        ranked = sorted(zip(i_upper[0], i_upper[1], dists[i_upper]),
                        key=lambda x: x[2])

        # walk down the ranked list, keep cointegrated pairs until quota met
        count = 0
        for i, j, _ in ranked:
            if count >= PAIRS_PER_CLUST:
                break

            t1, t2 = tickers[idx[i]], tickers[idx[j]]
            if _is_cointegrated(t1, t2):
                pairs.append((t1, t2))
                count += 1

        if count < PAIRS_PER_CLUST:
            print(f"[cluster {c}] only {count}/{PAIRS_PER_CLUST} cointegrated pairs found.")

    # save & return
    out = np.array(pairs, dtype=object)
    np.save(Path(LOG_DIR) / "pairs.npy", out)
    return pairs
