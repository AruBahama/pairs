
"""Train PPO agents on each selected trading pair."""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import ray
import logging
from ray.rllib.algorithms.ppo import PPO

from ..backtest import metrics

from .envs import PairTradingEnv
from ..config import LOG_DIR, PROC_DIR, NUM_WORKERS, WINDOW_LENGTH

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Early stopping parameters
TEST_FRACTION = 0.2     # fraction of data used for evaluation
MAX_ITERS = 100         # hard cap on training iterations
PATIENCE = 5            # window size for plateau detection
TOLERANCE = 0.01        # minimum Sharpe improvement to continue


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
    """Train a PPO agent for a single pair of tickers with early stopping."""

    price1 = _load_prices(t1)
    price2 = _load_prices(t2)

    split = int(len(price1) * (1 - TEST_FRACTION))
    train1, test1 = price1.iloc[:split], price1.iloc[split:]
    train2, test2 = price2.iloc[:split], price2.iloc[split:]

    env_cls = lambda cfg: PairTradingEnv(train1, train2)
    eval_env = PairTradingEnv(test1, test2)

    trainer = PPO(
        env=env_cls,
        config={
            "framework": "torch",
            "num_workers": NUM_WORKERS,
            "train_batch_size": 4000,
            # Use an LSTM-based policy and value network to retain a hidden state
            # of previous observations when optimising the policy and Q-values.
            "model": {
                "use_lstm": True,
                "max_seq_len": WINDOW_LENGTH,
                "lstm_cell_size": 128,
            },
        },
    )

    def _evaluate() -> float:
        obs, _ = eval_env.reset()
        state = trainer.get_policy().get_initial_state()
        done = False
        rewards = []
        while not done:
            action, state, _ = trainer.compute_single_action(obs, state=state)
            obs, _, done, _, info = eval_env.step(action)
            rewards.append(info["pnl"])
        return metrics.sharpe(np.array(rewards)) if rewards else 0.0

    sharpes: list[float] = []
    for i in range(MAX_ITERS):
        result = trainer.train()
        test_sharpe = _evaluate()
        sharpes.append(test_sharpe)
        logger.info(
            f"{t1}-{t2} iter {i}: reward {result['episode_reward_mean']:.3f} test Sharpe {test_sharpe:.3f}"
        )

        if len(sharpes) >= 2 * PATIENCE:
            prev = np.mean(sharpes[-2 * PATIENCE : -PATIENCE])
            curr = np.mean(sharpes[-PATIENCE:])
            if curr - prev < TOLERANCE:
                logger.info(
                    f"Early stopping at iter {i}: Sharpe plateaued at {curr:.3f}"
                )
                break

    checkpoint = trainer.save(LOG_DIR / f"{t1}_{t2}")
    trainer.cleanup()


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
        logger.info(f"Training pair {t1}-{t2}")
        _train_pair(t1, t2)

    ray.shutdown()


if __name__ == "__main__":
    from ..preflight import run_checks

    run_checks()
    train_all_pairs()
