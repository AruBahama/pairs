
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

## Pipeline Overview

1. **Download** OHLCV data using `src.data.downloader.batch` for tickers defined in `snp.csv`.
2. **Feature engineering** with `src.data.feature_engineer.batch` to build rolling windows of indicators.
3. **Train CAE** via `src.autoencoder.train_cae.train_cae` and save latent vectors.
4. **Cluster securities** using `src.clustering.cluster_utils.cluster_latents`.
5. **Select pairs** with `src.clustering.select_pairs.select_pairs` from each cluster.
6. **Train agents** on each pair through `src.rl.train_agent.train_all_pairs`.
7. **Backtest results** with `src.backtest.backtester.run_backtests`.

The RL agent training and backtesting steps currently contain placeholder
implementations and are intended as starting points for further development.

---
