from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym

from app.learning.features import build_observation, observation_size


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int = 50,
        initial_cash: float = 100000.0,
        commission_pct: float = 0.05,
        slippage_bps: float = 2.0,
        time_penalty_per_step: float = 0.0,
        feature_config: dict | None = None,
    ):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.window_size = window_size
        self.initial_cash = float(initial_cash)
        self.commission_pct = float(commission_pct)
        self.slippage_bps = float(slippage_bps)
        self.time_penalty_per_step = float(time_penalty_per_step)
        self.feature_config = feature_config or {}

        self.action_space = gym.spaces.Discrete(3)
        obs_len = observation_size(window_size, self.feature_config)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_len,),
            dtype=np.float32,
        )

        self._reset_state()

    def _reset_state(self) -> None:
        self.step_index = self.window_size
        self.position = 0
        self.cash = self.initial_cash
        self.position_qty = 0.0
        self.last_value = self.initial_cash

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_state()
        obs = self._get_obs()
        return obs, {}

    def _get_price(self, index: int) -> float:
        return float(self.data.loc[index, "Close"])

    def _get_obs(self) -> np.ndarray:
        closes = self.data.loc[: self.step_index, "Close"].tolist()
        volumes = self.data.loc[: self.step_index, "Volume"].tolist()
        cash_pct = self.cash / self.initial_cash if self.initial_cash else 1.0
        return build_observation(
            closes,
            volumes,
            self.window_size,
            float(self.position),
            float(cash_pct),
            feature_config=self.feature_config,
        )

    def _trade_cost(self, price: float) -> float:
        commission = price * (self.commission_pct / 100.0)
        slippage = price * (self.slippage_bps / 10000.0)
        return commission + slippage

    def step(self, action: int):
        done = False
        price = self._get_price(self.step_index)

        if action == 1 and self.position <= 0:
            if self.position < 0:
                self.cash += abs(self.position_qty) * price - self._trade_cost(price)
                self.position_qty = 0.0
            self.position = 1
            self.position_qty = 1.0
            self.cash -= price + self._trade_cost(price)
        elif action == 2 and self.position >= 0:
            if self.position > 0:
                self.cash += self.position_qty * price - self._trade_cost(price)
                self.position_qty = 0.0
            self.position = -1
            self.position_qty = -1.0
            self.cash += price - self._trade_cost(price)

        portfolio_value = self.cash + self.position_qty * price
        reward = portfolio_value - self.last_value - self.time_penalty_per_step
        self.last_value = portfolio_value

        self.step_index += 1
        if self.step_index >= len(self.data) - 1:
            done = True

        obs = self._get_obs()
        info = {
            "portfolio_value": portfolio_value,
            "position": self.position,
            "cash": self.cash,
        }
        return obs, float(reward), done, False, info
