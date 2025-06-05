
"""Global configuration"""
import os
from pathlib import Path

ROOT_DIR        = Path(__file__).resolve().parents[1]
DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
PROC_DIR        = DATA_DIR / "processed"
LOG_DIR         = ROOT_DIR / "logs"

# CSV listing tickers to download. Only the first column is used.
TICKER_FILE     = ROOT_DIR / "snp.csv"
START_DATE      = "2015-01-01"
END_DATE        = "2024-12-31"

WINDOW_LENGTH   = 60
LATENT_DIM      = 10
CAE_EPOCHS      = 500
CAE_BATCH_SIZE  = 256

N_CLUSTERS      = 10
PAIRS_PER_CLUST = 15

INIT_CAPITAL    = 1_000
RL_ALGO         = "PPO"
SWITCH_PENALTY  = 0.1

TRADE_FREQUENCY = "1D"
METRICS = ["total_pnl","annual_return","beta","alpha","max_drawdown","sharpe","sortino","calmar"]

SEED = 42
NUM_WORKERS = max(1, os.cpu_count()-1)
