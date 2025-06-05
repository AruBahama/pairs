
from sklearn.preprocessing import StandardScaler
import joblib, pandas as pd
from ..config import PROC_DIR, LOG_DIR

def fit_scaler():
    """Fit a :class:`StandardScaler` on all processed data."""
    files = list(PROC_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No processed parquet files found in {PROC_DIR}"
        )
    dfs = [pd.read_parquet(p) for p in files]
    X = pd.concat(dfs).dropna()
    scaler = StandardScaler().fit(X)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, LOG_DIR / "scaler.joblib")
    return scaler


def load_scaler():
    """Load the cached scaler or fit a new one if missing."""
    path = LOG_DIR / "scaler.joblib"
    if path.exists():
        return joblib.load(path)
    return fit_scaler()
