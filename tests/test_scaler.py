import pandas as pd
import numpy as np
import importlib

from src import config
from src.data import scaler as scaler_mod


def test_fit_and_load_scaler(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    df = pd.DataFrame({"a": [1,2,3,4], "b": [4,3,2,1]})
    for i in range(2):
        df.to_parquet(proc / f"t{i}.parquet")
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(config, "PROC_DIR", proc)
    monkeypatch.setattr(config, "LOG_DIR", log_dir)
    monkeypatch.setattr(scaler_mod, "PROC_DIR", proc)
    monkeypatch.setattr(scaler_mod, "LOG_DIR", log_dir)
    scaler = scaler_mod.fit_scaler()
    assert scaler.mean_.shape[0] == 2
    loaded = scaler_mod.load_scaler()
    assert np.allclose(loaded.mean_, scaler.mean_)
