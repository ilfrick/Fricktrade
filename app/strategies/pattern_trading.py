# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.learning.indicators import atr as compute_atr
from app.strategies.base import Strategy


@dataclass
class _PositionState:
    entry_price: float | None = None
    stop_price: float | None = None
    trailing_stop: float | None = None
    took_partial: bool = False


class PatternTradingStrategy(Strategy):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.state = _PositionState()

    def generate_signal(self, market_state: dict) -> dict:
        prices = market_state.get("prices", [])
        highs = market_state.get("highs", [])
        lows = market_state.get("lows", [])
        volumes = market_state.get("volumes", [])
        last_price = market_state.get("last_price")

        if not prices or last_price is None:
            return {"action": "hold"}

        if not self._passes_filters(market_state):
            return {"action": "hold"}

        if self.state.entry_price is None:
            if self._entry_signal(prices, highs, lows, volumes, last_price):
                stop_pct = float(self.cfg["risk"].get("stop_loss_pct", 0.05))
                # ATR-based adaptive stop (2x ATR) with fixed stop as floor
                atr_stop = last_price
                if len(highs) >= 2 and len(lows) >= 2 and len(prices) >= 2:
                    n = min(14, len(highs))
                    h_arr = np.array(highs[-n:], dtype=float)
                    l_arr = np.array(lows[-n:], dtype=float)
                    c_arr = np.array(prices[-n:], dtype=float)
                    atr_val = compute_atr(h_arr, l_arr, c_arr, period=min(14, n))
                    if atr_val > 0:
                        atr_stop = last_price - 2.0 * atr_val
                fixed_stop = last_price * (1.0 - stop_pct)
                stop_price = max(atr_stop, fixed_stop)
                self.state = _PositionState(entry_price=last_price, stop_price=stop_price)
                return {"action": "buy"}
            return {"action": "hold"}

        return self._manage_position(last_price)

    def _passes_filters(self, market_state: dict) -> bool:
        selection = self.cfg["selection"]
        price = market_state.get("last_price") or 0.0
        rel_vol = market_state.get("relative_volume") or 0.0
        session_gain = market_state.get("session_gain_pct") or 0.0
        total_volume = market_state.get("session_volume") or 0.0
        spread_pct = market_state.get("spread_pct")
        catalyst = market_state.get("catalyst", False)

        price_min = selection.get("price_min", 0.0)
        price_max = selection.get("price_max")
        if price < price_min:
            return False
        if price_max is not None and price > price_max:
            return False
        if rel_vol < selection["relative_volume_min"]:
            return False
        if session_gain < selection["premarket_gain_min_pct"]:
            return False
        if total_volume < selection["min_shares_traded"]:
            return False
        if selection.get("require_catalyst", True) and not catalyst:
            return False
        max_spread = selection.get("max_spread_pct", 1.0)
        if spread_pct is not None and spread_pct > max_spread:
            return False
        if selection.get("strict_spread", False) and spread_pct is None:
            return False
        return True

    def _entry_signal(
        self,
        prices: list[float],
        highs: list[float],
        lows: list[float],
        volumes: list[float],
        last_price: float,
    ) -> bool:
        if not highs or not lows or len(highs) < 2:
            return False
        lookback = self._lookback_bars()
        slice_data = highs[-lookback:-1] if len(highs) > lookback else highs[:-1]
        if not slice_data:
            return False
        recent_high = max(slice_data)
        if last_price <= recent_high:
            return False
        if not self._trend_ok(prices, highs, lows):
            return False
        if not self._pullback_ok(highs, lows, last_price):
            return False
        if not self._volume_ok(volumes):
            return False
        return True

    def _trend_ok(self, prices: list[float], highs: list[float], lows: list[float]) -> bool:
        ma_periods = self.cfg["pattern"].get("ma_periods", [9, 20])
        for period in ma_periods:
            if len(prices) < period:
                return False
            ma = float(np.mean(prices[-period:]))
            if prices[-1] < ma:
                return False
        if len(highs) >= 3 and len(lows) >= 3:
            if not (highs[-1] > highs[-2] > highs[-3]):
                return False
            if not (lows[-1] > lows[-2] > lows[-3]):
                return False
        return True

    def _pullback_ok(self, highs: list[float], lows: list[float], last_price: float) -> bool:
        lookback = self._lookback_bars()
        recent_high = max(highs[-lookback:])
        recent_low = min(lows[-lookback:])
        if recent_high <= recent_low:
            return False
        retrace = (recent_high - last_price) / (recent_high - recent_low)
        max_retrace = self.cfg["pattern"].get("pullback_max_retrace_pct", 50) / 100.0
        return retrace <= max_retrace

    def _volume_ok(self, volumes: list[float]) -> bool:
        if len(volumes) < 5:
            return False
        avg_vol = float(np.mean(volumes[:-1]))
        if avg_vol <= 0:
            return False
        mult = float(self.cfg["entry"].get("volume_confirm_mult", 1.5))
        return volumes[-1] >= avg_vol * mult

    def _manage_position(self, last_price: float) -> dict:
        stop_pct = float(self.cfg["risk"].get("stop_loss_pct", 0.05))
        trail_pct = float(self.cfg["risk"].get("trailing_stop_pct", 0.02))
        take_profit_pct = float(self.cfg["risk"].get("partial_take_profit_pct", 0.10))

        if self.state.entry_price is None:
            return {"action": "hold"}

        if self.state.stop_price is None:
            self.state.stop_price = self.state.entry_price * (1.0 - stop_pct)

        if self.state.trailing_stop is None:
            self.state.trailing_stop = self.state.entry_price * (1.0 - trail_pct)
        else:
            self.state.trailing_stop = max(self.state.trailing_stop, last_price * (1.0 - trail_pct))

        if not self.state.took_partial and last_price >= self.state.entry_price * (1.0 + take_profit_pct):
            self.state.took_partial = True
            return {"action": "sell", "reduce_pct": 0.5}

        if last_price <= self.state.stop_price or last_price <= (self.state.trailing_stop or 0.0):
            self.state = _PositionState()
            return {"action": "exit"}

        return {"action": "hold"}

    def _lookback_bars(self) -> int:
        return int(self.cfg["entry"].get("breakout_lookback_bars", 20))
