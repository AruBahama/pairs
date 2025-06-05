
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ..config import INIT_CAPITAL, WINDOW_LENGTH, SWITCH_PENALTY

class PairTradingEnv(gym.Env):
    """Simple PnL reward: Δ(PnL) given discrete actions."""
    metadata = {"render_modes": ["human"]}

    def __init__(self, price1_series, price2_series, hedge_ratio=None):
        """Environment using two price series.

        Parameters
        ----------
        price1_series, price2_series : pandas.Series
            Aligned price series for each asset.
        hedge_ratio : array-like or callable, optional
            Static array of hedge ratios or function ``f(t)`` returning the
            hedge ratio at index ``t``. If ``None`` a ratio of ``1`` is used.
        """
        super().__init__()
        self.price1 = price1_series.values
        self.price2 = price2_series.values
        if callable(hedge_ratio):
            self._hr_fn = hedge_ratio
            self._hr_arr = None
        elif hedge_ratio is not None:
            hr = hedge_ratio.values if hasattr(hedge_ratio, "values") else np.asarray(hedge_ratio)
            self._hr_arr = hr
            self._hr_fn = None
            if len(hr) != len(self.price1):
                raise ValueError("hedge_ratio length must match price series")
        else:
            self._hr_fn = None
            self._hr_arr = np.ones_like(self.price1)

        self.action_space = spaces.Discrete(3)  # 0=flat,1=long-spread,2=short-spread
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(WINDOW_LENGTH,), dtype=np.float32
        )
        self.position = 0  # -1 short, 0 flat, 1 long
        self.capital = INIT_CAPITAL
        self.t = WINDOW_LENGTH
        self.prev_hr = self._get_hr(self.t - 1)

    def _get_hr(self, idx):
        return (
            self._hr_fn(idx)
            if self._hr_fn is not None
            else self._hr_arr[idx]
        )

    def _get_hr_array(self, start, end):
        if self._hr_fn is not None:
            return np.array([self._hr_fn(i) for i in range(start, end)])
        return self._hr_arr[start:end]

    def _compute_spread(self, start, end):
        hr = self._get_hr_array(start, end)
        return self.price1[start:end] - hr * self.price2[start:end]

    def step(self, action):
        prev_position = self.position
        hr_prev = self.prev_hr
        prev_spread = self.price1[self.t - 1] - hr_prev * self.price2[self.t - 1]

        self.t += 1
        done = self.t >= len(self.price1)

        current_spread = (
            self.price1[self.t - 1] - hr_prev * self.price2[self.t - 1]
        )
        pnl = prev_position * (current_spread - prev_spread)
        reward = pnl
        new_position = {0:0,1:1,2:-1}[action]
        if prev_position * new_position == -1:
            reward -= SWITCH_PENALTY
        self.position = new_position
        obs = self._compute_spread(self.t - WINDOW_LENGTH, self.t)
        self.prev_hr = self._get_hr(self.t - 1)
        info = {"pnl": pnl}
        return obs.astype(np.float32), reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 0
        self.t = WINDOW_LENGTH
        self.prev_hr = self._get_hr(self.t - 1)
        obs = self._compute_spread(0, self.t)
        return obs.astype(np.float32), {}
