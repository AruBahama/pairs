
import json
import numpy as np
import pandas as pd
from ..config import PROC_DIR, CAE_EPOCHS, CAE_BATCH_SIZE, LOG_DIR
from .cae_model import build_cae
from ..data.window_builder import build_windows
from ..data.scaler import fit_scaler

def train_cae():
    """Train the convolutional autoencoder and save latent factors."""
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
        w = build_windows(pd.DataFrame(X[idx:idx + len(df)], index=df.index))
        windows.append(w)
        lengths.append(len(w))
        idx += len(df)
    Xw = np.concatenate(windows, axis=0)

    model, encoder = build_cae(n_features)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    history = model.fit(
        Xw,
        Xw,
        epochs=CAE_EPOCHS,
        batch_size=CAE_BATCH_SIZE,
        validation_split=0.1,
    )
    model.save(LOG_DIR / "cae.h5")
    encoder.save(LOG_DIR / "encoder.h5")

    latent = encoder.predict(Xw)
    np.save(LOG_DIR / "latent.npy", latent)

    # Aggregate latent vectors per ticker to obtain one vector per stock
    start = 0
    agg = []
    for l in lengths:
        agg.append(latent[start : start + l].mean(axis=0))
        start += l
    ticker_latent = np.stack(agg, axis=0)
    np.save(LOG_DIR / "ticker_latent.npy", ticker_latent)

    tickers = [p.stem for p in files]
    with open(LOG_DIR / "ticker_index.json", "w") as f:
        json.dump(tickers, f)

    return history
