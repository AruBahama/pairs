
# Pairs‑Trading System (Deep RL + Clustering)

End‑to‑end research pipeline that discovers statistical‑arbitrage pairs in the S&P 500, clusters them with a convolutional auto‑encoder (CAE), and trains a PPO agent (Ray RLlib) to trade each pair with **$1 000** initial capital.

## Quick start

```bash
git clone <your‑fork> pairs
cd pairs
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/optimize_hyperparams.py  # optional hyperparameter tuning
python scripts/run_full_pipeline.py
```
### Repository layout
- `src/` – production Python package with submodules for data, autoencoder, clustering, RL and backtesting
- `scripts/` – entry points and utilities
- `notebooks/` – Jupyter notebooks demonstrating the pipeline

## Pipeline Overview

1. **Data collection** → daily OHLCV + fundamentals (FinanceToolkit) aligned to fiscal quarter‑ends
2. **Pre‑processing** → technical indicators + scaling + 60‑day windows
3. **Hyperparameter search** (Optuna) → optimal window length, latent dimension and cluster count
4. **CAE training** (TensorFlow/Keras, 500 epochs) → 10‑D latent factors
5. **Clustering** (Ward, k=10) → pick 15 closest pairs/cluster
6. **RL training** (PPO) – *reward = ΔPnL*
7. **Backtest** → PnL, annual %, beta, alpha, MDD, Sharpe, Sortino, Calmar

All notebooks under `/notebooks` are executable in order and call the production code in `src/`.

---
