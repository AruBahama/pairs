
"""Download daily OHLCV data."""
import pandas as pd, yfinance as yf
from pathlib import Path
from ..config import RAW_DIR, START_DATE, END_DATE, TICKER_FILE

def _fill_missing_days(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure continuous business-day index for later alignment."""
    return df.resample("B").ffill()

def download(ticker:str):
    data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    if data.empty:
        print(f"No data for {ticker}")
        return
    data = _fill_missing_days(data)
    out = RAW_DIR/f"{ticker}.csv"
    out.parent.mkdir(parents=True,exist_ok=True)
    data.to_csv(out)

def batch():
    # The snp.csv file may contain additional columns; only the first column
    # with ticker symbols should be used.
    tickers = (
        pd.read_csv(TICKER_FILE, usecols=[0])
          .iloc[:, 0]
          .dropna()
          .tolist()
    )
    for t in tickers:
        print('↓',t)
        download(t.strip())

if __name__ == "__main__":
    batch()
