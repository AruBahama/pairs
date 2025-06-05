
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


---
