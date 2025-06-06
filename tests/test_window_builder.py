import numpy as np
import pandas as pd
from src.data.window_builder import build_windows


def test_build_windows_basic():
    df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
    windows = build_windows(df, window_length=3)
    assert windows.shape == (2, 3, 2)
    np.testing.assert_array_equal(windows[0], df.iloc[0:3].values)
