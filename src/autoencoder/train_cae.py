import json
import numpy as np
import pandas as pd
from ..config import PROC_DIR, CAE_EPOCHS, CAE_BATCH_SIZE, LOG_DIR, WINDOW_LENGTH, LATENT_DIM
from .cae_model import build_cae
from ..data.window_builder import build_windows
from ..data.scaler import fit_scaler


def _exp_weights(n: int, alpha: float) -> np.ndarray:
    """Return exponentially decaying weights with emphasis on recent samples."""
    w = alpha ** np.arange(n)[::-1]
    return w / w.sum()

def train_cae(
    window_length: int = WINDOW_LENGTH,
    latent_dim: int = LATENT_DIM,
    save: bool = True,
    recency_alpha: float = 0.9,
):
    """Train the CAE and return latent representations.

    Parameters
    ----------
    window_length : int
        Length of each input window.
    latent_dim : int
        Dimension of the latent space.
    save : bool
        If ``True`` saves the model and embeddings to ``LOG_DIR``.
    recency_alpha : float
        Exponential decay factor for recency weighting. Values closer to 1 give
        more emphasis to recent windows.

    Returns
    -------
    list[np.ndarray]
        Window level latent vectors for each ticker.
    np.ndarray
        Array of shape ``(n_tickers, latent_dim * 2)`` containing the weighted
        mean and variance for each ticker.
    History
        Training history returned by ``model.fit``.
    """
    scaler = fit_scaler()
    files = sorted(PROC_DIR.glob("*.parquet"))
    dfs = [pd.read_parquet(p) for p in files]
    X = pd.concat(dfs).dropna()
    X = scaler.transform(X)
    n_features = X.shape[1]

    windows = []
    lengths = []
    idx = 0
    for df in dfs:
        w = build_windows(
            pd.DataFrame(X[idx: idx + len(df)], index=df.index),
            window_length
        )
        windows.append(w)
        lengths.append(len(w))
        idx += len(df)
    Xw = np.concatenate(windows, axis=0)

    model, encoder = build_cae(n_features, window_length, latent_dim)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    history = model.fit(
        Xw,
        Xw,
        epochs=CAE_EPOCHS,
        batch_size=CAE_BATCH_SIZE,
        validation_split=0.1,
    )

    if save:
        model.save(LOG_DIR / "cae.h5")
        encoder.save(LOG_DIR / "encoder.h5")

    latent = encoder.predict(Xw)
    if save:
        np.save(LOG_DIR / "latent.npy", latent)

    # Keep window level embeddings grouped by ticker
    start = 0
    ticker_windows_latent: list[np.ndarray] = []
    agg = []
    for l in lengths:
        lw = latent[start : start + l]
        ticker_windows_latent.append(lw)
        weights = _exp_weights(l, recency_alpha)
        mean = np.average(lw, axis=0, weights=weights)
        var = np.average((lw - mean) ** 2, axis=0, weights=weights)
        agg.append(np.concatenate([mean, var]))
        start += l
    ticker_latent = np.stack(agg, axis=0)
    if save:
        np.save(LOG_DIR / "ticker_latent.npy", ticker_latent)

    tickers = [p.stem for p in files]
    if save:
        with open(LOG_DIR / "ticker_index.json", "w") as f:
            json.dump(tickers, f)

    return ticker_windows_latent, ticker_latent, history
