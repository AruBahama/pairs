
# Pairs‑Trading System (Deep RL + Clustering)

End‑to‑end research pipeline that discovers statistical‑arbitrage pairs in the S&P 500, clusters them with a convolutional auto‑encoder (CAE), and trains a PPO agent (Ray RLlib) to trade each pair with **$1 000** initial capital.

## Quick start

```bash
git clone <your‑fork> pairs_trading_system
cd pairs_trading_system
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_full_pipeline.py
```

## Pipeline Overview

1. **Data collection** → daily OHLCV + fundamentals (FinanceToolkit) aligned to quarter‑ends  
2. **Pre‑processing** → technical indicators + scaling + 60‑day windows  
3. **CAE training** (TensorFlow/Keras, 500 epochs) → 10‑D latent factors  
4. **Clustering** (Ward, k=10) → pick 15 closest pairs/cluster  
5. **RL training** (PPO) – *reward = ΔPnL*  
6. **Backtest** → PnL, annual %, beta, alpha, MDD, Sharpe, Sortino, Calmar  

All notebooks under `/notebooks` are executable in order and call the production code in `src/`.

---
