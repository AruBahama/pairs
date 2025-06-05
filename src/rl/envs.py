
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ..config import INIT_CAPITAL, WINDOW_LENGTH

class PairTradingEnv(gym.Env):
    """Simple PnL reward: Δ(PnL) given discrete actions."""
    metadata={'render_modes':['human']}

    def __init__(self, spread_series):
        super().__init__()
        self.spread = spread_series.values
        self.action_space = spaces.Discrete(3)  # 0=flat,1=long-spread,2=short-spread
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(WINDOW_LENGTH,), dtype=np.float32)
        self.position = 0  # -1 short, 0 flat, 1 long
        self.capital = INIT_CAPITAL
        self.t = WINDOW_LENGTH

    def step(self, action):
        prev_spread = self.spread[self.t-1]
        self.position = {0:0,1:1,2:-1}[action]
        self.t += 1
        done = self.t >= len(self.spread)
        obs = self.spread[self.t-WINDOW_LENGTH:self.t]
        pnl = self.position * (self.spread[self.t-1] - prev_spread)
        reward = pnl  # ΔPnL as reward
        info={'pnl':pnl}
        return obs.astype(np.float32), reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position=0
        self.t=WINDOW_LENGTH
        obs=self.spread[:WINDOW_LENGTH]
        return obs.astype(np.float32), {}
