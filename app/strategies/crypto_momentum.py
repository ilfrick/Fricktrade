# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.strategies.base import Strategy


@dataclass
class CryptoMomentumParams:
    fast_window: int = 5
    medium_window: int = 15
    slow_window: int = 60
    volume_mult: float = 2.0
    min_return_pct: float = 0.3
    trailing_stop_pct: float = 2.0


class CryptoMomentumStrategy(Strategy):
    """Short-term momentum optimised for crypto.

    Signals:
    - BUY  when 5m/15m/60m returns all positive AND volume > volume_mult × avg
    - HOLD otherwise (no short — Alpaca crypto is long-only)

    Confidence is proportional to the minimum return across timeframes (capped at 1.0).
    Only activates for crypto symbols (containing '/').
    """

    def __init__(self, params: dict):
        cfg = params.get("crypto_momentum", {}) if isinstance(params, dict) else {}
        self.params = CryptoMomentumParams(
            fast_window=int(cfg.get("fast_window", 5)),
            medium_window=int(cfg.get("medium_window", 15)),
            slow_window=int(cfg.get("slow_window", 60)),
            volume_mult=float(cfg.get("volume_mult", 2.0)),
            min_return_pct=float(cfg.get("min_return_pct", 0.3)),
            trailing_stop_pct=float(cfg.get("trailing_stop_pct", 2.0)),
        )

    def generate_signal(self, market_state: dict) -> dict:
        symbol = market_state.get("symbol", "")
        if "/" not in symbol:
            return {"action": "hold", "confidence": 0.0, "name": "crypto_momentum"}

        prices = market_state.get("prices", []) or []
        min_len = self.params.slow_window + 2
        if len(prices) < min_len:
            return {"action": "hold", "confidence": 0.0, "name": "crypto_momentum"}

        close = np.array(prices, dtype=float)

        # Returns over each timeframe
        def _ret(n: int) -> float:
            if close[-n - 1] <= 0:
                return 0.0
            return float((close[-1] - close[-n - 1]) / close[-n - 1] * 100.0)

        ret_fast = _ret(self.params.fast_window)
        ret_med = _ret(self.params.medium_window)
        ret_slow = _ret(self.params.slow_window)

        # Volume confirmation
        volumes = market_state.get("volumes", []) or []
        vol_ok = False
        if len(volumes) >= self.params.medium_window + 1:
            avg_vol = float(np.mean(volumes[-self.params.medium_window - 1 : -1]))
            vol_ok = avg_vol > 0 and float(volumes[-1]) >= avg_vol * self.params.volume_mult

        all_positive = ret_fast > self.params.min_return_pct and ret_med > 0.0 and ret_slow > 0.0
        all_negative = ret_fast < -self.params.min_return_pct and ret_med < 0.0 and ret_slow < 0.0

        if all_positive and vol_ok:
            min_ret = min(ret_fast, ret_med, ret_slow)
            confidence = min(min_ret / (self.params.min_return_pct * 3.0), 1.0)
            return {
                "action": "buy",
                "confidence": float(confidence),
                "name": "crypto_momentum",
                "trailing_stop_pct": self.params.trailing_stop_pct,
            }

        if all_negative and vol_ok:
            # Long-only: signal to exit any existing position
            return {"action": "sell", "confidence": 0.6, "name": "crypto_momentum"}

        return {"action": "hold", "confidence": 0.0, "name": "crypto_momentum"}
