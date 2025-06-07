"""RL trading environments."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from ..config import INIT_CAPITAL, WINDOW_LENGTH, SWITCH_PENALTY, STOP_LOSS_LEVEL
from .hedge_utils import calc_hedge_ratio


class PairTradingEnv(gym.Env):
    """Simple pair-trading environment with Δ-PnL reward."""

    metadata = {"render_modes": ["human"]}

    # ------------------------------------------------------------------ #
    # Constructor
    # ------------------------------------------------------------------ #
    def __init__(self, price1_series: pd.Series, price2_series: pd.Series,
                 hedge_ratio=None):
        """
        Parameters
        ----------
        price1_series, price2_series : pandas.Series
            Synchronized price series of the two assets.
        hedge_ratio : array-like | callable | None
            • array-like  – a pre-computed hedge-ratio series  
            • callable    – a function f(t) that returns the ratio at time t  
            • None        – estimate a static Kelly-style ratio
        """
        super().__init__()

        # --- Cache prices as numpy for speed --------------------------------
        self.price1 = price1_series.values
        self.price2 = price2_series.values
        self.window_length = min(WINDOW_LENGTH, len(self.price1))

        # --- Hedge-ratio handling ------------------------------------------
        if callable(hedge_ratio):                      # dynamic function
            self._hr_fn, self._hr_arr = hedge_ratio, None
        elif hedge_ratio is not None:                  # user-supplied array
            hr = hedge_ratio.values if hasattr(hedge_ratio, "values") else np.asarray(hedge_ratio)
            if len(hr) != len(self.price1):
                raise ValueError("hedge_ratio length must match price series length")
            self._hr_fn, self._hr_arr = None, hr.astype(float)
        else:                                          # static estimate
            static_hr = calc_hedge_ratio(price1_series, price2_series)
            self._hr_fn, self._hr_arr = None, np.full_like(self.price1, static_hr, dtype=float)

        # --- Gym spaces -----------------------------------------------------
        # 0 = short spread, 1 = flat, 2 = long spread
        self.action_space = spaces.Discrete(3)

        # observation is the last `window_length` normalised spreads
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_length,),
            dtype=np.float32,
        )

        # --- Episode state --------------------------------------------------
        self.position: int = 0               # −1 short, 0 flat, +1 long
        self.capital:  float = INIT_CAPITAL
        self.stop_loss = STOP_LOSS_LEVEL
        self.t:        int   = self.window_length  # index of *next* bar
        self.prev_hr:  float = self._get_hr(self.t - 1)

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #
    def _get_hr(self, idx: int) -> float:
        """Return hedge ratio at index *idx*."""
        return self._hr_fn(idx) if self._hr_fn is not None else self._hr_arr[idx]

    def _get_hr_array(self, start: int, end: int) -> np.ndarray:
        """Return hedge-ratio slice for [start, end)."""
        start = max(start, 0)
        end   = min(end, len(self.price1))
        if self._hr_fn is not None:
            return np.array([self._hr_fn(i) for i in range(start, end)], dtype=float)
        return self._hr_arr[start:end]

    def _compute_spread(self, start: int, end: int) -> np.ndarray:
        """
        Normalised spread for window [start, end).
        Always returns a vector of length `self.window_length`,
        left-padded with zeros if the requested slice is shorter.
        """
        start = max(start, 0)
        end   = max(start, min(end, len(self.price1)))

        hr      = self._get_hr_array(start, end)
        spread  = self.price1[start:end] - hr * self.price2[start:end]

        if spread.size == 0:
            spread = np.zeros(1, dtype=float)

        denom   = max(abs(spread[0]), 1e-6)        # avoid division by 0
        spread  = spread / denom

        # left-pad if the slice is shorter than window_length
        pad = self.window_length - spread.size
        if pad > 0:
            spread = np.pad(spread, (pad, 0), mode="constant")

        return spread

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def step(self, action: int):
        """Advance one bar and return (obs, reward, done, trunc, info)."""
        if action not in (0, 1, 2):
            raise ValueError(f"Invalid action {action}; must be 0, 1 or 2")

        prev_position = self.position
        self.position = {0: -1, 1: 0, 2: 1}[action]

        # --- Calculate previous spread (t-1) using previous hedge ratio ------
        prev_spread = self.price1[self.t - 1] - self.prev_hr * self.price2[self.t - 1]

        # --- Advance time ----------------------------------------------------
        self.t += 1
        done = self.t >= len(self.price1)

        # --- Current spread --------------------------------------------------
        if done:
            # We stepped past the last bar; reuse previous spread
            curr_spread = prev_spread
        else:
            self.prev_hr = self._get_hr(self.t - 1)
            curr_spread  = self.price1[self.t - 1] - self.prev_hr * self.price2[self.t - 1]

        # --- Reward: Δ-PnL normalised by |prev_spread| -----------------------
        pnl     = self.position * (curr_spread - prev_spread)
        self.capital += pnl
        reward  = pnl / max(abs(prev_spread), 1e-6)

        # switching penalty
        if prev_position != self.position and prev_position != 0:
            reward -= SWITCH_PENALTY

        # --- Stop-loss -------------------------------------------------------
        hit_stop = self.capital < self.stop_loss
        if hit_stop:
            self.position = 0
            done = True

        # --- Build observation ----------------------------------------------
        end   = min(self.t, len(self.price1))
        start = max(end - self.window_length, 0)
        obs   = self._compute_spread(start, end).astype(np.float32)

        info = {"pnl": pnl, "stop_loss": hit_stop}
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        """Gym reset; returns (obs, info)."""
        super().reset(seed=seed)

        self.position = 0
        self.capital  = INIT_CAPITAL
        self.stop_loss = STOP_LOSS_LEVEL
        self.t        = self.window_length
        self.prev_hr  = self._get_hr(self.t - 1)

        obs = self._compute_spread(0, self.t).astype(np.float32)
        return obs, {}
