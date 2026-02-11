# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from collections import deque
from dataclasses import dataclass

from app.learning.features import build_observation, observation_size


@dataclass
class RewardConfig:
    """Single source of truth for reward parameters.

    Configured at ``learning.reward`` in config.yaml.  Per-account overrides
    (in ``brokers.alpaca.accounts[].reward``) are deep-merged onto these
    defaults at startup.  The orchestrator's ``time_penalty_per_bar`` is
    derived from this config when not explicitly set:
        time_penalty_per_bar = time_penalty_weight * bar_interval_minutes / time_normalizer
    """

    nav_weight: float = 1.0            # w1 - differential NAV
    time_penalty_weight: float = 0.5   # w2 - position holding penalty
    profit_bonus_weight: float = 2.0   # w3 - realized profit bonus
    velocity_weight: float = 0.3       # w4 - equity velocity
    nav_normalizer: float = 100_000.0  # normalizes all components
    time_normalizer: float = 390.0     # minutes in a trading day
    velocity_window: int = 20          # lookback steps for equity velocity
    bar_interval_minutes: float = 5.0  # minutes per bar


@dataclass
class RewardBreakdown:
    nav_change: float = 0.0
    time_penalty: float = 0.0
    profit_bonus: float = 0.0
    velocity: float = 0.0
    override: float = 0.0
    total: float = 0.0


class TradingRewardCalculator:
    def __init__(self, config: RewardConfig):
        self.cfg = config
        self.prev_nav: float = 0.0
        self.equity_buffer: deque[float] = deque(maxlen=config.velocity_window)
        self.position_entry_step: int = 0
        self.position_entry_price: float = 0.0
        self.in_position: bool = False

    def reset(self, initial_nav: float) -> None:
        self.prev_nav = initial_nav
        self.equity_buffer.clear()
        self.equity_buffer.append(initial_nav)
        self.position_entry_step = 0
        self.position_entry_price = 0.0
        self.in_position = False

    def on_position_open(self, step: int, price: float) -> None:
        self.in_position = True
        self.position_entry_step = step
        self.position_entry_price = price

    def on_position_close(self) -> None:
        self.in_position = False
        self.position_entry_step = 0
        self.position_entry_price = 0.0

    def compute(
        self,
        nav: float,
        step: int,
        realized_pnl: float,
        override: float = 0.0,
        override_mode: str = "add",
    ) -> RewardBreakdown:
        norm = max(self.cfg.nav_normalizer, 1e-6)
        bd = RewardBreakdown()

        # 1. Differential NAV
        bd.nav_change = self.cfg.nav_weight * (nav - self.prev_nav) / norm

        # 2. Time penalty (only while in position)
        if self.in_position:
            steps_held = max(step - self.position_entry_step, 0)
            minutes_held = steps_held * self.cfg.bar_interval_minutes
            bd.time_penalty = -self.cfg.time_penalty_weight * minutes_held / max(self.cfg.time_normalizer, 1e-6)

        # 3. Transaction costs - implicit in NAV change (already deducted)

        # 4. Realized profit bonus (only on close with profit)
        if realized_pnl > 0:
            bd.profit_bonus = self.cfg.profit_bonus_weight * realized_pnl / norm

        # 5. Equity velocity
        self.equity_buffer.append(nav)
        if len(self.equity_buffer) >= 2:
            changes = []
            buf = list(self.equity_buffer)
            for i in range(1, len(buf)):
                changes.append(buf[i] - buf[i - 1])
            avg_change = sum(changes) / len(changes)
            bd.velocity = self.cfg.velocity_weight * avg_change / norm

        # Override handling
        bd.override = override

        # Total
        bd.total = bd.nav_change + bd.time_penalty + bd.profit_bonus + bd.velocity
        if override:
            if override_mode == "override":
                bd.total = override
            else:
                bd.total += override

        self.prev_nav = nav
        return bd


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int = 50,
        initial_cash: float = 100000.0,
        commission_pct: float = 0.05,
        slippage_bps: float = 2.0,
        feature_config: dict | None = None,
        reward_config: RewardConfig | dict | None = None,
        symbol: str | None = None,
        reward_overrides: dict[str, float] | None = None,
        reward_override_mode: str = "add",
    ):
        super().__init__()
        self.data = data.reset_index(drop=True)
        if self.data.empty:
            raise ValueError("training data is empty")
        self.window_size = window_size
        self.initial_cash = float(initial_cash)
        self.commission_pct = float(commission_pct)
        self.slippage_bps = float(slippage_bps)
        self.feature_config = feature_config or {}

        if isinstance(reward_config, dict):
            self.reward_config = RewardConfig(**{
                k: v for k, v in reward_config.items()
                if k in RewardConfig.__dataclass_fields__
            })
        elif reward_config is None:
            self.reward_config = RewardConfig()
        else:
            self.reward_config = reward_config
        self.reward_calculator = TradingRewardCalculator(self.reward_config)

        self.symbol = str(symbol) if symbol else None
        self.reward_overrides = reward_overrides or {}
        self.reward_override_mode = str(reward_override_mode or "add").lower()

        self.position_entry_price = 0.0
        self.total_realized_pnl = 0.0
        self.gross_profits = 0.0
        self.gross_losses = 0.0
        self.trades_this_episode = 0

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
        self.position_entry_price = 0.0
        self.total_realized_pnl = 0.0
        self.last_value = self.initial_cash
        self.trades_this_episode = 0
        self.gross_profits = 0.0
        self.gross_losses = 0.0
        self.reward_calculator.reset(self.initial_cash)

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
        highs = self.data.loc[: self.step_index, "High"].tolist() if "High" in self.data.columns else None
        lows = self.data.loc[: self.step_index, "Low"].tolist() if "Low" in self.data.columns else None
        cash_pct = self.cash / self.initial_cash if self.initial_cash else 1.0
        return build_observation(
            closes,
            volumes,
            self.window_size,
            float(self.position),
            float(cash_pct),
            float(cash_pct),
            feature_config=self.feature_config,
            highs=highs,
            lows=lows,
        )

    def _trade_cost(self, price: float) -> float:
        commission = price * (self.commission_pct / 100.0)
        slippage = price * (self.slippage_bps / 10000.0)
        return commission + slippage

    def step(self, action: int):
        done = False
        price = self._get_price(self.step_index)

        # Resolve live reward override
        reward_override = 0.0
        if self.reward_overrides and "datetime" in self.data.columns:
            try:
                ts = self.data.loc[self.step_index, "datetime"]
            except Exception:
                ts = None
            if ts is not None:
                try:
                    ts_val = pd.to_datetime(ts, errors="coerce")
                except Exception:
                    ts_val = None
                if ts_val is not None and not pd.isna(ts_val):
                    bucket = ts_val.floor(f"{int(self.reward_config.bar_interval_minutes)}min")
                    key = bucket.isoformat()
                    reward_override = float(self.reward_overrides.get(key, 0.0) or 0.0)

        # Variables to store realized PnL for the current step
        realized_pnl_this_step = 0.0

        # Current position status before action
        current_position = self.position

        # Determine if a position is being closed
        is_closing_long = (action == 2 or action == 0) and current_position == 1
        is_closing_short = (action == 1 or action == 0) and current_position == -1

        if is_closing_long:
            closed_pnl = (price - self.position_entry_price) * self.position_qty
            trade_cost = self._trade_cost(self.position_entry_price) * self.position_qty + \
                         self._trade_cost(price) * self.position_qty
            realized_pnl_this_step += closed_pnl - trade_cost
            self.total_realized_pnl += realized_pnl_this_step
            self.cash += self.position_qty * price - self._trade_cost(price)
            self.position = 0
            self.position_qty = 0.0
            self.position_entry_price = 0.0
            self.reward_calculator.on_position_close()

        elif is_closing_short:
            closed_pnl = (self.position_entry_price - price) * abs(self.position_qty)
            trade_cost = self._trade_cost(self.position_entry_price) * abs(self.position_qty) + \
                         self._trade_cost(price) * abs(self.position_qty)
            realized_pnl_this_step += closed_pnl - trade_cost
            self.total_realized_pnl += realized_pnl_this_step
            self.cash += abs(self.position_qty) * price - self._trade_cost(price)
            self.position = 0
            self.position_qty = 0.0
            self.position_entry_price = 0.0
            self.reward_calculator.on_position_close()

        # Track wins/losses for info dict
        if realized_pnl_this_step != 0:
            self.trades_this_episode += 1
            if realized_pnl_this_step > 0:
                self.gross_profits += realized_pnl_this_step
            else:
                self.gross_losses += abs(realized_pnl_this_step)

        # Apply action to open new position
        if action == 1:  # Go long
            if self.position == 0:
                self.position = 1
                self.position_qty = 1.0
                self.position_entry_price = price
                self.cash -= price + self._trade_cost(price)
                self.reward_calculator.on_position_open(self.step_index, price)
        elif action == 2:  # Go short
            if self.position == 0:
                self.position = -1
                self.position_qty = -1.0
                self.position_entry_price = price
                self.cash += price - self._trade_cost(price)
                self.reward_calculator.on_position_open(self.step_index, price)

        # Compute NAV = cash + mark-to-market position value
        portfolio_value = self.cash + (self.position_qty * price if self.position != 0 else 0)
        self.last_value = portfolio_value

        # Compute reward via calculator
        bd = self.reward_calculator.compute(
            nav=portfolio_value,
            step=self.step_index,
            realized_pnl=realized_pnl_this_step,
            override=reward_override,
            override_mode=self.reward_override_mode,
        )
        reward = bd.total

        self.step_index += 1
        if self.step_index >= len(self.data) - 1:
            done = True

        obs = self._get_obs()

        profit_factor = self.gross_profits / max(self.gross_losses, 1e-6)

        info = {
            "portfolio_value": portfolio_value,
            "position": self.position,
            "cash": self.cash,
            "total_realized_pnl": self.total_realized_pnl,
            "gross_profits": self.gross_profits,
            "gross_losses": self.gross_losses,
            "profit_factor": profit_factor,
            "trades_count": self.trades_this_episode,
        }
        return obs, float(reward), done, False, info
