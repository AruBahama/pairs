
from sklearn.preprocessing import StandardScaler
import joblib, pandas as pd
from ..config import PROC_DIR, LOG_DIR

def fit_scaler():
    dfs = [pd.read_parquet(p) for p in PROC_DIR.glob("*.parquet")]
    X = pd.concat(dfs).dropna()
    scaler = StandardScaler().fit(X)
    joblib.dump(scaler, LOG_DIR/'scaler.joblib')
    return scaler
