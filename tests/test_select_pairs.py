import numpy as np
import json
from src import config
from src.clustering import select_pairs as sp


def test_select_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    monkeypatch.setattr(sp, "LOG_DIR", tmp_path)
    np.save(tmp_path / "ticker_latent_scaled.npy", np.random.rand(4, 3))
    np.save(tmp_path / "labels.npy", np.array([0, 0, 1, 1]))
    with open(tmp_path / "ticker_index.json", "w") as f:
        json.dump(["A", "B", "C", "D"], f)
    pairs = sp.select_pairs()
    assert isinstance(pairs, list)
    assert (tmp_path / "pairs.npy").exists()
