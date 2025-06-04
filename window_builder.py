
import numpy as np, pandas as pd
from ..config import WINDOW_LENGTH

def build_windows(df:pd.DataFrame):
    X=[]
    for i in range(len(df)-WINDOW_LENGTH):
        X.append(df.iloc[i:i+WINDOW_LENGTH].values)
    return np.array(X)
