import numpy as np
import pandas as pd


def calc_hedge_ratio(price1: pd.Series, price2: pd.Series) -> float:
    """Return hedge ratio using correlation-adjusted volatility scaling.

    The hedge ratio quantifies how many shares of ``price2`` are required to
    neutralize price risk in ``price1``. It equals ``rho * sigma_A / sigma_B``
    where ``rho`` is the correlation between ``Δprice1`` and ``Δprice2``,
    ``sigma_A`` is the standard deviation of ``Δprice1`` and ``sigma_B`` the
    standard deviation of ``Δprice2``.
    """
    d1 = price1.diff().dropna()
    d2 = price2.diff().dropna()
    if len(d1) == 0 or len(d2) == 0:
        raise ValueError("price series must contain at least two observations")

    rho = np.corrcoef(d1, d2)[0, 1]
    sigma_a = np.std(d1, ddof=1)
    sigma_b = np.std(d2, ddof=1)
    return float(rho * sigma_a / sigma_b)
