# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

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
        if self.data.empty:
            raise ValueError("training data is empty")
        self.window_size = window_size
        self.initial_cash = float(initial_cash)
        self.commission_pct = float(commission_pct)
        self.slippage_bps = float(slippage_bps)
        self.time_penalty_per_step = float(time_penalty_per_step)
        self.feature_config = feature_config or {}

        self.position_entry_price = 0.0 # Track entry price for current position
        self.position_entry_qty = 0.0   # Track quantity for current position
        self.total_realized_pnl = 0.0   # Track total realized PnL
        self.last_portfolio_value = self.initial_cash # Keep track for overall change for flat periods

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
        last_index = max(len(self.data) - 1, 0)
        self.step_index = min(self.window_size, last_index)
        self.position = 0
        self.cash = self.initial_cash
        self.position_qty = 0.0
        self.last_value = self.initial_cash
        self.position_entry_price = 0.0
        self.position_entry_qty = 0.0
        self.total_realized_pnl = 0.0
        self.last_portfolio_value = self.initial_cash

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
        reward = -self.time_penalty_per_step # Start with time penalty

        # Variables to store realized PnL for the current step, if any
        realized_pnl_this_step = 0.0

        # Current position status before action
        current_position = self.position
        
        # Determine if a position is being closed
        is_closing_long = (action == 2 or action == 0) and current_position == 1
        is_closing_short = (action == 1 or action == 0) and current_position == -1

        if is_closing_long:
            # Calculate realized PnL for closed long trade
            closed_pnl = (price - self.position_entry_price) * self.position_entry_qty
            trade_cost = self._trade_cost(self.position_entry_price) * self.position_entry_qty + \
                         self._trade_cost(price) * self.position_entry_qty
            realized_pnl_this_step += closed_pnl - trade_cost
            self.total_realized_pnl += realized_pnl_this_step
            self.cash += self.position_qty * price - self._trade_cost(price)
            # Reset position tracking
            self.position = 0
            self.position_qty = 0.0
            self.position_entry_price = 0.0
            
        elif is_closing_short:
            # Calculate realized PnL for closed short trade
            closed_pnl = (self.position_entry_price - price) * abs(self.position_entry_qty)
            trade_cost = self._trade_cost(self.position_entry_price) * abs(self.position_entry_qty) + \
                         self._trade_cost(price) * abs(self.position_entry_qty)
            realized_pnl_this_step += closed_pnl - trade_cost
            self.total_realized_pnl += realized_pnl_this_step
            self.cash += abs(self.position_qty) * price - self._trade_cost(price)
            # Reset position tracking
            self.position = 0
            self.position_qty = 0.0
            self.position_entry_price = 0.0

        # Apply action to open new position or change existing one
        if action == 1:  # Go long
            if self.position == 0: # Only open if currently flat
                self.position = 1
                self.position_qty = 1.0 # Assume fixed quantity
                self.position_entry_price = price
                self.cash -= price + self._trade_cost(price)
            # If already long, do nothing (hold long)
        elif action == 2:  # Go short
            if self.position == 0: # Only open if currently flat
                self.position = -1
                self.position_qty = -1.0 # Assume fixed quantity
                self.position_entry_price = price
                self.cash += price - self._trade_cost(price)
            # If already short, do nothing (hold short)
        # If action is 0 (flat), and position was closed above, we are now flat.
        # If action is 0 and we were already flat, we remain flat.

        # Add realized PnL from closed trades to the reward for this step
        reward += realized_pnl_this_step

        # Update last_value for portfolio_value tracking, though not directly used for reward in this scheme
        portfolio_value = self.cash + (self.position_qty * price if self.position != 0 else 0)
        self.last_value = portfolio_value # Keep for consistency or future use

        self.step_index += 1
        if self.step_index >= len(self.data) - 1:
            done = True

        obs = self._get_obs()
        info = {
            "portfolio_value": portfolio_value,
            "position": self.position,
            "cash": self.cash,
            "total_realized_pnl": self.total_realized_pnl # Add to info for monitoring
        }
        return obs, float(reward), done, False, info
