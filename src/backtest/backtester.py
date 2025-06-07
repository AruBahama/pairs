
"""Utilities for running RL policy backtests with Backtesting.py."""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from ray.rllib.algorithms.algorithm import Algorithm

from ..config import (
    LOG_DIR,
    PROC_DIR,
    INIT_CAPITAL,
    WINDOW_LENGTH,
    METRICS,
    STOP_LOSS_LEVEL,
)
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


class RLPairStrategy(Strategy):
    """Backtesting strategy that executes a trained RL policy."""

    ckpt_dir: str | Path

    def init(self) -> None:
        self.agent = _load_checkpoint(Path(self.ckpt_dir))
        self.state = self.agent.get_initial_state()
        self.stop_trading = False

    def next(self) -> None:
        if self.stop_trading:
            return

        if self.equity < STOP_LOSS_LEVEL:
            if self.position:
                self.position.close()
            self.stop_trading = True
            return

        obs = self.data.Close[-WINDOW_LENGTH:].values.astype("float32")
        action, self.state, _ = self.agent.compute_single_action(
            obs, state=self.state
        )
        if action == 1:
            if not self.position.is_long:
                if self.position:
                    self.position.close()
                self.buy()
        elif action == 2:
            if not self.position.is_short:
                if self.position:
                    self.position.close()
                self.sell()
        else:
            if self.position:
                self.position.close()


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

        price1 = _load_prices(t1)
        price2 = _load_prices(t2)
        df = pd.DataFrame({"spread": price1 - price2})
        # Feed the spread into Backtesting.py by duplicating it across the
        # OHLC columns.  Backtesting.py expects an OHLC dataframe even if the
        # strategy trades a synthetic instrument such as a spread.
        df_bt = pd.DataFrame(
            {
                "Open": df["spread"],
                "High": df["spread"],
                "Low": df["spread"],
                "Close": df["spread"],
                "Volume": 0,
            }
        )

        bt = Backtest(
            df_bt,
            RLPairStrategy,
            cash=INIT_CAPITAL,
            commission=0.0005,
            slippage=0.0005,
            strategy_kwargs={"ckpt_dir": LOG_DIR / f"{t1}_{t2}"},
        )

        stats = bt.run()
        equity = stats["_equity_curve"]["Equity"]
        returns = equity.pct_change().dropna()

        # Use the spread's own returns as a benchmark for metrics that
        # require one (e.g. beta and alpha).
        benchmark = df["spread"].pct_change().dropna().loc[returns.index]

        metrics = {}
        for name in METRICS:
            func = getattr(m, name)
            if name in {"beta", "alpha"}:
                metrics[name] = func(returns, benchmark)
            else:
                metrics[name] = func(returns)
        metrics["Final Equity"] = float(equity.iloc[-1])
        results[pair_name] = metrics
        print(f"Backtested {pair_name}: {metrics}")

    return results
