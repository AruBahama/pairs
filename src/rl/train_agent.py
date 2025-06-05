
"""Train PPO agents on each selected trading pair."""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from .envs import PairTradingEnv
from ..config import LOG_DIR, PROC_DIR, NUM_WORKERS


def _load_prices(ticker: str) -> pd.Series:
    """Return a price series for ``ticker`` from ``PROC_DIR``.

    The function tries common column names and defaults to the first numeric
    column if none are found.
    """

    df = pd.read_parquet(PROC_DIR / f"{ticker}.parquet")
    for col in ["Close", "close", "Adj Close", "adjclose"]:
        if col in df.columns:
            return df[col]
    return df.select_dtypes(include="number").iloc[:, 0]


def _train_pair(t1: str, t2: str) -> None:
    """Train a PPO agent for a single pair of tickers."""

    price1 = _load_prices(t1)
    price2 = _load_prices(t2)
    env_id = f"PairTradingEnv_{t1}_{t2}"

    def env_creator(env_config=None):
        return PairTradingEnv(price1, price2)

    register_env(env_id, env_creator)
    config = (
        PPOConfig()
        .environment(env_id)
        .framework("tf2")
        .rollouts(num_rollout_workers=NUM_WORKERS)
    )
    algo = config.build()
    for _ in range(10):
        algo.train()

    ckpt_dir = Path(LOG_DIR) / f"ppo_{t1}_{t2}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    algo.save(str(ckpt_dir))
    algo.stop()


def train_all_pairs(pairs: Iterable[tuple[str, str]] | None = None) -> None:
    """Train PPO agents for each pair listed in ``pairs``.

    If ``pairs`` is ``None`` the function loads ``pairs.npy`` from ``LOG_DIR``.
    """

    ray.init(ignore_reinit_error=True)
    if pairs is None:
        pairs_file = LOG_DIR / "pairs.npy"
        if not pairs_file.exists():
            print("pairs.npy not found; nothing to train.")
            ray.shutdown()
            return
        pairs = np.load(pairs_file, allow_pickle=True)

    for t1, t2 in pairs:
        print(f"Training pair {t1}-{t2}")
        _train_pair(t1, t2)

    ray.shutdown()


if __name__ == "__main__":
    train_all_pairs()
