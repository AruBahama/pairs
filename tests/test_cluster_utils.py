import numpy as np
from src import config
from src.clustering import cluster_utils


def test_cluster_latents(tmp_path, monkeypatch):
    data = np.random.rand(10, 4)
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    # patch module-level LOG_DIR if present
    monkeypatch.setattr(cluster_utils, "LOG_DIR", tmp_path)
    Zs, labels = cluster_utils.cluster_latents(data, n_clusters=2, save=True)
    assert Zs.shape == data.shape
    assert len(labels) == 10
