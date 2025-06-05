
"""Utilities for running RL policy backtests with Backtesting.py."""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from ray.rllib.algorithms.algorithm import Algorithm

from ..config import LOG_DIR, PROC_DIR, INIT_CAPITAL, WINDOW_LENGTH, METRICS
from . import metrics as m


def _load_prices(ticker: str) -> pd.Series:
    """Return a processed price series for ``ticker``."""

    df = pd.read_parquet(PROC_DIR / f"{ticker}.parquet")
    for col in ["Close", "close", "Adj Close", "adjclose"]:
        if col in df.columns:
            return df[col]
    return df.select_dtypes(include="number").iloc[:, 0]


def _load_checkpoint(directory: Path) -> Algorithm:
    """Load the most recent RLlib checkpoint from ``directory``."""

    candidates = sorted(directory.glob("checkpoint_*/"), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {directory}")
    return Algorithm.from_checkpoint(str(candidates[0]))


def run_backtests(
    policy_dir: str | Path | None = None,
    pair_list: Iterable[tuple[str, str]] | None = None,
) -> dict[str, dict[str, float]]:
    """Run offline backtests for trained policies and return metrics.

    Parameters
    ----------
    policy_dir : path-like, optional
        Directory containing saved RL policy checkpoints. Defaults to ``LOG_DIR``.
    pair_list : iterable of ``(ticker1, ticker2)`` tuples, optional
        Pairs to evaluate. If ``None``, the function attempts to load
        ``pairs.npy`` from ``policy_dir``.
    """

    policy_dir = Path(policy_dir or LOG_DIR)

    if pair_list is None:
        pairs_file = policy_dir / "pairs.npy"
        if not pairs_file.exists():
            print("pairs.npy not found; nothing to backtest.")
            return {}
        pair_list = np.load(pairs_file, allow_pickle=True)

    results: dict[str, dict[str, float]] = {}

    for t1, t2 in pair_list:
        pair_name = f"{t1}-{t2}"
        ckpt_dir = policy_dir / f"ppo_{t1}_{t2}"
        try:
            algo = _load_checkpoint(ckpt_dir)
        except Exception as e:
            print(f"Skipping {pair_name}: {e}")
            continue

        price1 = _load_prices(t1)
        price2 = _load_prices(t2)
        spread = price1 - price2
        df = pd.DataFrame(
            {
                "Open": spread,
                "High": spread,
                "Low": spread,
                "Close": spread,
            }
        )

        class RLPairStrategy(Strategy):
            def init(self):
                self.t = WINDOW_LENGTH
                self.state = algo.get_initial_state()

            def next(self):
                if self.t >= len(spread):
                    return
                obs = spread.iloc[self.t - WINDOW_LENGTH : self.t].values.astype(
                    "float32"
                )
                action, self.state, _ = algo.compute_single_action(
                    obs, state=self.state
                )
                if action == 1:  # long spread
                    if not self.position.is_long:
                        self.position.close()
                        self.buy()
                elif action == 2:  # short spread
                    if not self.position.is_short:
                        self.position.close()
                        self.sell()
                else:  # flat
                    if self.position:
                        self.position.close()
                self.t += 1

        bt = Backtest(
            df,
            RLPairStrategy,
            cash=INIT_CAPITAL,
            commission=0.0,
            exclusive_orders=True,
        )

        stats = bt.run()
        equity = stats["_equity_curve"]["Equity"]
        returns = equity.pct_change().dropna()
        metrics = {name: getattr(m, name)(returns) for name in METRICS}
        metrics["Final Equity"] = float(equity.iloc[-1])
        results[pair_name] = metrics
        print(f"Backtested {pair_name}: {metrics}")

    return results
