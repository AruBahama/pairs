import numpy as np
import pandas as pd
from src.rl.hedge_utils import calc_hedge_ratio


def test_calc_hedge_ratio():
    price1 = pd.Series([1, 2, 4, 7, 11], dtype=float)
    price2 = pd.Series([2, 4, 8, 14, 22], dtype=float)
    ratio = calc_hedge_ratio(price1, price2)
    assert np.isclose(ratio, 0.5)
