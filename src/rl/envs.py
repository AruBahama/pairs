"""RL trading environments."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from ..config import INIT_CAPITAL, WINDOW_LENGTH, SWITCH_PENALTY
from .hedge_utils import calc_hedge_ratio


class PairTradingEnv(gym.Env):
    """Simple pair-trading environment with Δ-PnL reward."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, price1_series, price2_series, hedge_ratio=None):
        """
        Parameters
        ----------
        price1_series, price2_series : pandas.Series
            Synchronized price series of the two assets.
        hedge_ratio : array-like or callable, optional
            Static array of hedge ratios or a function ``f(t)`` returning the
            ratio at index ``t``.  If ``None`` the ratio is estimated as
            ``rho * sigma_A / sigma_B`` where ``rho`` is the correlation between
            ``Δprice1`` and ``Δprice2`` and ``sigma_A``/``sigma_B`` are their
            respective standard deviations.
        """
        super().__init__()

        # Cache prices as numpy for speed
        self.price1 = price1_series.values
        self.price2 = price2_series.values

        # Handle hedge-ratio input
        if callable(hedge_ratio):
            self._hr_fn, self._hr_arr = hedge_ratio, None
        elif hedge_ratio is not None:
            hr = hedge_ratio.values if hasattr(hedge_ratio, "values") else np.asarray(hedge_ratio)
            if len(hr) != len(self.price1):
                raise ValueError("hedge_ratio length must match price series length")
            self._hr_fn, self._hr_arr = None, hr
        else:
            hr = calc_hedge_ratio(price1_series, price2_series)
            self._hr_fn, self._hr_arr = None, np.full_like(self.price1, hr, dtype=float)

        # Action mapping: 0 = short spread, 1 = flat, 2 = long spread
        self.action_space = spaces.Discrete(3)

        # Observation = last WINDOW_LENGTH normalised spread values
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(WINDOW_LENGTH,), dtype=np.float32
        )

        # Internal state
        self.position = 0            # −1 short, 0 flat, +1 long
        self.capital  = INIT_CAPITAL
        self.t        = WINDOW_LENGTH
        self.prev_hr  = self._get_hr(self.t - 1)

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _get_hr(self, idx: int):
        return self._hr_fn(idx) if self._hr_fn is not None else self._hr_arr[idx]

    def _get_hr_array(self, start: int, end: int):
        if self._hr_fn is not None:
            return np.array([self._hr_fn(i) for i in range(start, end)])
        return self._hr_arr[start:end]

    def _compute_spread(self, start: int, end: int):
        """Return normalised spread for the window [start, end)."""
        hr      = self._get_hr_array(start, end)
        spread  = self.price1[start:end] - hr * self.price2[start:end]
        denom   = max(abs(spread[0]), 1e-6)           # avoid /0
        return spread / denom

    # --------------------------------------------------------------------- #
    # Gym API
    # --------------------------------------------------------------------- #
    def step(self, action: int):
        """Advance one time-step and return (obs, reward, done, trunc, info)."""
        prev_position = self.position
        self.position = {-1: -1, 0: 0, 1: 1}.get({0: -1, 1: 0, 2: 1}[action], 0)

        # Spread at t-1 with previous hedge ratio
        prev_spread = self.price1[self.t - 1] - self.prev_hr * self.price2[self.t - 1]

        # Advance time index
        self.t += 1
        done = self.t >= len(self.price1)

        # Current spread (use *current* hedge ratio)
        self.prev_hr = self._get_hr(self.t - 1)
        curr_spread  = self.price1[self.t - 1] - self.prev_hr * self.price2[self.t - 1]

        # PnL for this step, scaled by |prev_spread| to keep rewards magnitude-stable
        pnl     = self.position * (curr_spread - prev_spread)
        reward  = pnl / max(abs(prev_spread), 1e-6)

        # Switching penalty
        if prev_position != self.position and prev_position != 0:
            reward -= SWITCH_PENALTY

        # Observation = most-recent WINDOW_LENGTH normalised spreads
        obs = self._compute_spread(self.t - WINDOW_LENGTH, self.t)

        info = {"pnl": pnl}
        return obs.astype(np.float32), reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 0
        self.t        = WINDOW_LENGTH
        self.prev_hr  = self._get_hr(self.t - 1)
        obs = self._compute_spread(0, self.t)
        return obs.astype(np.float32), {}
