
import numpy as np
import pandas as pd
from ..config import PROC_DIR, CAE_EPOCHS, CAE_BATCH_SIZE, LOG_DIR
from .cae_model import build_cae
from ..data.window_builder import build_windows
from ..data.scaler import fit_scaler

def train_cae():
    scaler = fit_scaler()
    dfs = [pd.read_parquet(p) for p in PROC_DIR.glob("*.parquet")]
    X = pd.concat(dfs).dropna()
    X = scaler.transform(X)
    n_features = X.shape[1]
    windows=[]
    idx=0
    for df in dfs:
        w=build_windows(pd.DataFrame(X[idx:idx+len(df)],index=df.index))
        windows.append(w)
        idx+=len(df)
    Xw = np.concatenate(windows,axis=0)
    Xw = Xw[...,np.newaxis]  # add channel dim

    model, encoder = build_cae(n_features)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    history = model.fit(Xw, Xw, epochs=CAE_EPOCHS, batch_size=CAE_BATCH_SIZE, validation_split=0.1)
    model.save(LOG_DIR/'cae.h5')
    encoder.save(LOG_DIR/'encoder.h5')
    np.save(LOG_DIR/'latent.npy', encoder.predict(Xw))
    return history
