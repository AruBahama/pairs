
#!/usr/bin/env bash
# Example SLURM submission or local launch
set -euo pipefail
# Default RLlib training
python -m src.rl.train_agent
# Or run Stable-Baselines3 PPO
# python scripts/train_agents_sb3.py
