"""Technical indicators + fundamentals"""
import pandas as pd, pandas_ta as ta
from financetoolkit import Toolkit
from ..config import RAW_DIR, PROC_DIR

def engineer(ticker:str):
    df = pd.read_csv(RAW_DIR/f"{ticker}.csv", index_col=0, parse_dates=True)
    df.ta.strategy(ta.Strategy(name="all", talib=False))
    df = df.dropna()

    tk = Toolkit(
        [ticker],  # FinanceToolkit expects a list
        start=df.index.min(),
        end=df.index.max(),
        enforce_source="YahooFinance",
    )

    funda = tk.ratios.collect()
    # Up-sample quarterly fundamentals to business days and
    # interpolate between reporting dates
    funda = (funda.resample("B").interpolate().ffill()
                   .reindex(df.index).ffill())

    df = pd.concat([df, funda], axis=1).dropna()
    out = PROC_DIR/f"{ticker}.parquet"
    out.parent.mkdir(parents=True,exist_ok=True)
    df.to_parquet(out)

def batch():
    for p in RAW_DIR.glob("*.csv"):
        engineer(p.stem)

if __name__ == "__main__":
    batch()
