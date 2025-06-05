import numpy as np
import pandas as pd
from ..config import WINDOW_LENGTH

def build_windows(
    df: pd.DataFrame, window_length: int = WINDOW_LENGTH
) -> np.ndarray:
    """Convert a feature dataframe into overlapping windows."""
    if len(df) <= window_length:
        return np.empty((0, window_length, df.shape[1]))
    return np.lib.stride_tricks.sliding_window_view(
        df.values, (window_length, df.shape[1])
    )[:-1, 0]
