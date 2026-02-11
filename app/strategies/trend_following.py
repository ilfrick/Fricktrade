# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.strategies.base import Strategy


@dataclass
class TrendParams:
    fast_window: int = 10
    slow_window: int = 30
    breakout_pct: float = 0.3
    exit_pct: float = 0.2


class TrendFollowingStrategy(Strategy):
    def __init__(self, params: dict):
        cfg = params.get("trend_following", {}) if isinstance(params, dict) else {}
        self.params = TrendParams(
            fast_window=int(cfg.get("fast_window", 10)),
            slow_window=int(cfg.get("slow_window", 30)),
            breakout_pct=float(cfg.get("breakout_pct", 0.3)),
            exit_pct=float(cfg.get("exit_pct", 0.2)),
        )

    def generate_signal(self, market_state: dict) -> dict:
        prices = market_state.get("prices", []) or []
        if len(prices) < max(self.params.fast_window, self.params.slow_window) + 1:
            return {"action": "hold"}
        close = np.array(prices, dtype=float)
        fast = float(close[-self.params.fast_window :].mean())
        slow = float(close[-self.params.slow_window :].mean())
        last = float(close[-1])
        if slow <= 0:
            return {"action": "hold"}
        trend_strength = (fast - slow) / slow * 100.0
        confidence = float(min(abs(trend_strength) / max(self.params.breakout_pct * 3.0, 0.01), 1.0))
        if trend_strength >= self.params.breakout_pct and last >= fast:
            return {"action": "buy", "confidence": confidence, "trend_strength": trend_strength}
        if trend_strength <= -self.params.exit_pct and last <= fast:
            return {"action": "sell", "confidence": confidence, "trend_strength": trend_strength}
        return {"action": "hold", "confidence": 0.0, "trend_strength": trend_strength}
