
def sharpe(returns, rf=0):
    """Sharpe ratio of a returns series."""
    import numpy as np
    excess = returns - rf
    return np.mean(excess) / np.std(excess, ddof=1)


def total_pnl(returns):
    """Total profit and loss."""
    import numpy as np
    return np.sum(returns)


def annual_return(returns, periods_per_year=252):
    """Annualized return based on compounding."""
    import numpy as np
    if len(returns) == 0:
        return 0.0
    cumulative = np.prod(1 + returns)
    return cumulative ** (periods_per_year / len(returns)) - 1


def beta(returns, benchmark):
    """Beta of the strategy relative to a benchmark."""
    import numpy as np
    cov = np.cov(returns, benchmark, ddof=1)
    return cov[0, 1] / cov[1, 1]


def alpha(returns, benchmark):
    """Alpha of the strategy after adjusting for beta."""
    import numpy as np
    b = beta(returns, benchmark)
    return np.mean(returns) - b * np.mean(benchmark)


def max_drawdown(returns):
    """Maximum drawdown of the cumulative returns curve."""
    import numpy as np
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def sortino(returns, rf=0):
    """Sortino ratio using downside deviation."""
    import numpy as np
    excess = returns - rf
    negative = excess[excess < 0]
    if negative.size == 0:
        return float('inf')
    downside = np.std(negative, ddof=1)
    return np.mean(excess) / downside


def calmar(returns, periods_per_year=252):
    """Calmar ratio of annual return to maximum drawdown."""
    ar = annual_return(returns, periods_per_year)
    dd = abs(max_drawdown(returns))
    if dd == 0:
        return float('inf')
    return ar / dd

