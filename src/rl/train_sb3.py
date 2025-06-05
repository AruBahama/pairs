"""Training utilities using Stable-Baselines3 PPO."""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .envs import PairTradingEnv
from ..config import LOG_DIR, PROC_DIR


def _load_prices(ticker: str) -> pd.Series:
    """Return a price series for ``ticker`` from ``PROC_DIR``."""
    df = pd.read_parquet(PROC_DIR / f"{ticker}.parquet")
    for col in ["Close", "close", "Adj Close", "adjclose"]:
        if col in df.columns:
            return df[col]
    return df.select_dtypes(include="number").iloc[:, 0]


def train_pair_sb3(
    t1: str,
    t2: str,
    total_timesteps: int = 100_000,
    gamma: float = 0.99,
) -> None:
    """Train PPO on a single ticker pair using SB3."""
    price1 = _load_prices(t1)
    price2 = _load_prices(t2)

    def make_env():
        return PairTradingEnv(price1, price2)

    env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", env, gamma=gamma, verbose=1)
    model.learn(total_timesteps=total_timesteps)

    save_dir = Path(LOG_DIR) / f"sb3_ppo_{t1}_{t2}"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(save_dir / "model"))


def train_all_pairs_sb3(
    pairs: Iterable[tuple[str, str]] | None = None,
    total_timesteps: int = 100_000,
    gamma: float = 0.99,
) -> None:
    """Train PPO agents with SB3 for each pair listed in ``pairs``."""

    if pairs is None:
        pairs_file = LOG_DIR / "pairs.npy"
        if not pairs_file.exists():
            print("pairs.npy not found; nothing to train.")
            return
        pairs = np.load(pairs_file, allow_pickle=True)

    for t1, t2 in pairs:
        print(f"Training pair {t1}-{t2}")
        train_pair_sb3(t1, t2, total_timesteps=total_timesteps, gamma=gamma)


if __name__ == "__main__":
    train_all_pairs_sb3()
