import numpy as np
import pandas as pd
import importlib
import pytest

pytest.importorskip('gymnasium')

from src.rl import envs as env_mod
from src import config
PairTradingEnv = env_mod.PairTradingEnv


def test_env_basic_step(monkeypatch):
    monkeypatch.setattr(config, "WINDOW_LENGTH", 3)
    monkeypatch.setattr(env_mod, "WINDOW_LENGTH", 3)
    price1 = pd.Series(np.arange(10, dtype=float))
    price2 = pd.Series(np.arange(10, dtype=float))
    env = PairTradingEnv(price1, price2)
    obs, _ = env.reset()
    assert obs.shape[0] == env.observation_space.shape[0]
    obs, reward, done, trunc, info = env.step(1)
    assert isinstance(reward, float)
    assert obs.shape == env.observation_space.shape
