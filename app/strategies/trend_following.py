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

        # RSI(14) filter
        rsi_period = 14
        if len(close) >= rsi_period + 1:
            deltas = np.diff(close[-(rsi_period + 1):])
            gains = np.where(deltas > 0, deltas, 0.0)
            losses = np.where(deltas < 0, -deltas, 0.0)
            avg_gain = float(gains.mean())
            avg_loss = float(losses.mean())
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
            else:
                rsi = 100.0
        else:
            rsi = 50.0  # neutral default

        # Volume confirmation
        volumes = market_state.get("volumes", []) or []
        vol_ok = True
        if len(volumes) >= 5:
            avg_vol = float(np.mean(volumes[-5:-1])) if len(volumes) > 1 else 0.0
            vol_ok = avg_vol <= 0 or volumes[-1] >= avg_vol * 1.5

        if trend_strength >= self.params.breakout_pct and last >= fast and rsi < 70 and vol_ok:
            return {"action": "buy", "confidence": confidence, "trend_strength": trend_strength, "rsi": rsi}
        if trend_strength <= -self.params.exit_pct and last <= fast and rsi > 30:
            return {"action": "sell", "confidence": confidence, "trend_strength": trend_strength, "rsi": rsi}
        return {"action": "hold", "confidence": 0.0, "trend_strength": trend_strength, "rsi": rsi}
