
# Pairs‑Trading System (Deep RL + Clustering)

End‑to‑end research pipeline that discovers statistical‑arbitrage pairs in the S&P 500, clusters them with a convolutional auto‑encoder (CAE), and trains a PPO agent (Ray RLlib) to trade each pair with **$1 000** initial capital. The RL agent now employs an LSTM‑based recurrent policy and critic so that past observations influence current decisions.

## Quick Start

Clone the repository, install the Python dependencies and run the pipeline:

```bash
git clone <your-fork> pairs
cd pairs
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_full_pipeline.py
```

This single command downloads data, trains the ML models and runs a
backtest. A Mermaid diagram describing how each module connects is available in
[docs/pipeline_flow.html](docs/pipeline_flow.html).
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
   The policy and value networks use an LSTM backbone so previous observations
   are carried across timesteps. Alternatively run `scripts/train_agents_sb3.py`
   to use Stable-Baselines3 PPO (``gamma=0.99``).
7. **Backtest results** with `src.backtest.backtester.run_backtests`.

The RL training script uses a lightweight configuration suitable for quick
experiments. Backtests leverage **Backtesting.py** via
`src.backtest.backtester.run_backtests`, and a collection of common financial
metrics is provided in `src.backtest.metrics`.

---

### Hedge Ratio Definition

The system uses a volatility- and correlation-adjusted hedge ratio.  Given two
price series ``A`` and ``B`` the ratio is

``beta = rho * sigma_A / sigma_B``

where ``rho`` is the correlation between the expected price changes
``ΔA`` and ``ΔB`` and ``sigma_A``/``sigma_B`` are the standard deviations of
those changes.  This value indicates how many shares of the short leg (``B``)
neutralize the price risk in the long leg (``A``), allowing the spread to
capture relative mispricing rather than market drift.

