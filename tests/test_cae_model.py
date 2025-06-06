import numpy as np
import pytest
pytest.importorskip('tensorflow')

from src.autoencoder.cae_model import build_cae


def test_build_cae_forward_pass():
    model, encoder = build_cae(n_features=2, window_length=4, latent_dim=2)
    x = np.random.rand(1, 4, 2).astype(np.float32)
    out = model.predict(x)
    assert out.shape == x.shape
    latent = encoder.predict(x)
    assert latent.shape == (1, 2)
