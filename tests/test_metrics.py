import numpy as np
from src.backtest import metrics


def test_sharpe_and_others():
    returns = np.array([0.01, -0.02, 0.03, -0.01])
    bench = np.array([0.02, -0.01, 0.01, 0.0])
    assert np.isfinite(metrics.sharpe(returns))
    assert np.isfinite(metrics.total_pnl(returns))
    assert np.isfinite(metrics.annual_return(returns))
    assert np.isfinite(metrics.beta(returns, bench))
    assert np.isfinite(metrics.alpha(returns, bench))
    assert np.isfinite(metrics.max_drawdown(returns))
    assert np.isfinite(metrics.sortino(returns))
    assert np.isfinite(metrics.calmar(returns))
