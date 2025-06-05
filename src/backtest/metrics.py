"""Financial metrics utilities (currently placeholders)."""


def sharpe(returns,rf=0):
    import numpy as np
    excess = returns - rf
    return np.mean(excess)/np.std(excess,ddof=1)

# add other metrics...


def total_pnl(trades):
    """Total profit and loss across all trades."""
    raise NotImplementedError("total_pnl metric not implemented")


def annual_return(equity_curve):
    """Annualized return based on an equity curve."""
    raise NotImplementedError("annual_return metric not implemented")


def beta(returns, benchmark_returns):
    """Risk beta relative to a benchmark."""
    raise NotImplementedError("beta metric not implemented")


def alpha(returns, benchmark_returns, rf=0):
    """Risk-adjusted alpha relative to a benchmark."""
    raise NotImplementedError("alpha metric not implemented")


def max_drawdown(equity_curve):
    """Maximum observed drawdown."""
    raise NotImplementedError("max_drawdown metric not implemented")


def sortino(returns, rf=0):
    """Sortino ratio using downside volatility."""
    raise NotImplementedError("sortino metric not implemented")


def calmar(returns):
    """Calmar ratio based on annual return and drawdown."""
    raise NotImplementedError("calmar metric not implemented")
