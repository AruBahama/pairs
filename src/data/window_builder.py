
import numpy as np, pandas as pd
from ..config import WINDOW_LENGTH

def build_windows(df: pd.DataFrame, window_length: int = WINDOW_LENGTH) -> np.ndarray:
    """Convert a feature dataframe into overlapping windows."""
    X = []
    for i in range(len(df) - window_length):
        X.append(df.iloc[i : i + window_length].values)
    return np.array(X)
