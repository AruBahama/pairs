
"""Simple backtesting utilities for trained pair‑trading agents."""

from pathlib import Path
import pandas as pd
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm

from .envs import PairTradingEnv
from .metrics import sharpe
from .config import (
    RAW_DIR,
    TICKER_FILE,
    LOG_DIR,
    INIT_CAPITAL,
    WINDOW_LENGTH,
)


def _load_prices(ticker: str) -> pd.Series:
    """Load close prices for a ticker."""
    csv = RAW_DIR / f"{ticker}.csv"
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    # assume a standard OHLCV file
    return df["Close"].dropna()


def run_backtests() -> pd.DataFrame:
    """Run backtests for all trained pairs.

    Returns
    -------
    pandas.DataFrame
        Summary statistics with total PnL and Sharpe ratio for each pair.
    """

    tickers = pd.read_csv(TICKER_FILE, header=None)[0].tolist()
    pairs = np.load(LOG_DIR / "pairs.npy")

    results = []

    for idx, (i, j) in enumerate(pairs):
        t1, t2 = tickers[i], tickers[j]

        # load prices and compute simple spread
        p1 = _load_prices(t1)
        p2 = _load_prices(t2)
        df = pd.DataFrame({"p1": p1, "p2": p2}).dropna()
        spread = df["p1"] - df["p2"]

        # initialise env and agent
        env = PairTradingEnv(spread)
        agent_dir = LOG_DIR / f"agent_{idx}"
        checkpoint_paths = list(agent_dir.glob("checkpoint_*/"))
        checkpoint = None
        if checkpoint_paths:
            checkpoint = max(
                checkpoint_paths, key=lambda p: int(p.name.split("_")[1])
            )
        if checkpoint is None:
            print(f"No checkpoint for pair {t1}-{t2}; skipping")
            continue
        algo = Algorithm.from_checkpoint(checkpoint)

        obs, _ = env.reset()
        capital = INIT_CAPITAL
        pnl_series = []

        done = False
        while not done:
            action = algo.compute_single_action(obs)
            obs, reward, done, _, info = env.step(action)
            capital += reward
            pnl_series.append(capital - INIT_CAPITAL)

        pnl = capital - INIT_CAPITAL
        sr = sharpe(np.array(pnl_series))

        results.append({
            "pair": f"{t1}-{t2}",
            "total_pnl": pnl,
            "sharpe": sr,
        })

    df_res = pd.DataFrame(results)
    out = LOG_DIR / "backtest_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out, index=False)
    return df_res


if __name__ == "__main__":
    summary = run_backtests()
    print(summary)
