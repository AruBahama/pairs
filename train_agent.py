
"""Train a PPO agent on all selected pairs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import ray
from ray.rllib.algorithms.ppo import PPOConfig

from ..config import LOG_DIR
from .envs import PairTradingEnv


def _load_spread(pair_idx: int) -> pd.Series:
    """Return a spread series for the given pair index.

    This helper first looks for ``spread_{idx}.csv`` in ``LOG_DIR``. When
    missing, a random walk series is generated as a simple placeholder so that
    training can run without real data.
    """

    csv_path = LOG_DIR / f"spread_{pair_idx}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0)
        return df.iloc[:, 0]

    # fallback random walk
    rng = np.random.default_rng(pair_idx)
    data = rng.standard_normal(500).cumsum()
    return pd.Series(data)


def train_all_pairs(num_iters: int = 10) -> None:
    """Train a PPO agent for each selected pair.

    Parameters
    ----------
    num_iters:
        Number of training iterations per pair.
    """

    ray.init(ignore_reinit_error=True)

    pairs_file = LOG_DIR / "pairs.npy"
    pairs = np.load(pairs_file, allow_pickle=True)

    for idx, pair in enumerate(pairs):
        spread = _load_spread(idx)

        def env_creator(env_config=None, spread=spread):
            return PairTradingEnv(spread)

        config = PPOConfig().environment(env_creator)
        algo = config.build()

        for _ in range(num_iters):
            algo.train()

        ckpt_dir = Path(LOG_DIR) / f"pair_{idx}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        algo.save(ckpt_dir)

        algo.stop()

    ray.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO agents for all pairs")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of training iterations per pair")
    args = parser.parse_args()

    train_all_pairs(args.iterations)
