import numpy as np
import pandas as pd
import json
from src import config
from src.clustering import select_pairs as sp


def test_select_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    monkeypatch.setattr(sp, "LOG_DIR", tmp_path)
    monkeypatch.setattr(config, "PROC_DIR", tmp_path)
    monkeypatch.setattr(sp, "PROC_DIR", tmp_path)
    monkeypatch.setattr(config, "PAIRS_PER_CLUST", 2)
    monkeypatch.setattr(sp, "PAIRS_PER_CLUST", 2)

    np.save(tmp_path / "ticker_latent_scaled.npy", np.random.rand(4, 3))
    np.save(tmp_path / "labels.npy", np.array([0, 0, 0, 0]))
    with open(tmp_path / "ticker_index.json", "w") as f:
        json.dump(["A", "B", "C", "D"], f)

    rng = np.random.default_rng(0)
    prices = {
        "A": rng.normal(size=100).cumsum(),
        "B": None,
        "C": rng.normal(size=100).cumsum(),
        "D": rng.normal(size=100).cumsum(),
    }
    prices["B"] = prices["A"] + rng.normal(scale=0.01, size=100)
    for t, p in prices.items():
        pd.DataFrame({"Close": p}).to_parquet(tmp_path / f"{t}.parquet")

    pairs = sp.select_pairs()
    assert ("A", "B") in pairs
    assert (tmp_path / "pairs.npy").exists()
