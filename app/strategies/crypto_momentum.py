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

        vel_fast = _ret(self.params.fast_window) / self.params.fast_window
        vel_med  = _ret(self.params.medium_window) / self.params.medium_window
        vel_slow = _ret(self.params.slow_window) / self.params.slow_window

        per_bar_thr = self.params.min_return_pct / self.params.fast_window
        slow_trend_min = per_bar_thr * 0.2  # require minimum slow trend strength (IMP-6)

        # Volume confirmation — used as a confidence boost, not a hard gate.
        # Alpaca crypto bar volume is platform-routed flow, not global exchange volume.
        # It's systematically thin on weekends and off-peak hours, so blocking on it
        # would silence signals during valid low-volume regimes.
        volumes = market_state.get("volumes", []) or []
        vol_boost = 1.0
        if len(volumes) >= self.params.medium_window + 1:
            avg_vol = float(np.mean(volumes[-self.params.medium_window - 1 : -1]))
            if avg_vol > 0:
                ratio = float(volumes[-1]) / avg_vol
                vol_boost = min(ratio / self.params.volume_mult, 1.5)  # caps at 1.5×

        all_positive = vel_fast > per_bar_thr and vel_med > 0.0 and vel_slow > slow_trend_min
        all_negative = vel_fast < -per_bar_thr and vel_med < 0.0 and vel_slow < -slow_trend_min

        if all_positive:
            weighted_vel = 0.5 * vel_fast + 0.3 * vel_med + 0.2 * vel_slow
            base_conf = min(weighted_vel / (per_bar_thr * 4.0), 1.0)
            # VWAP filter (IMP-4): vwap_dev is normalized to [-1, 1] (raw_pct / 5.0)
            indicators = market_state.get("indicators") or {}
            vwap_dev = indicators.get("vwap_dev")
            vwap_mult = 1.0
            if vwap_dev is not None:
                if vwap_dev > 0.3:   # overextended above VWAP (~1.5% raw) — dampen
                    vwap_mult = 0.7
                elif vwap_dev >= 0:  # at/near VWAP — ideal entry
                    vwap_mult = 1.1
                else:                # below VWAP in uptrend — caution
                    vwap_mult = 0.85
            # RSI overbought damper: crypto can stay overbought, but high RSI reduces
            # entry quality — dampen confidence rather than gate entirely
            rsi = self._wilder_rsi(list(close), 14)
            rsi_mult = 1.0
            if rsi > 80:
                rsi_mult = 0.5   # strongly overbought — halve confidence
            elif rsi > 72:
                rsi_mult = 0.75  # moderately overbought — reduce confidence
            confidence = min(base_conf * max(vol_boost, 0.5) * vwap_mult * rsi_mult, 1.0)
            return {
                "action": "buy",
                "confidence": float(confidence),
                "name": "crypto_momentum",
                "trailing_stop_pct": self.params.trailing_stop_pct,
            }

        if all_negative:
            # Long-only: signal to exit any existing position
            sell_conf = min(abs(vel_fast) / (per_bar_thr * 4.0), 1.0)
            return {"action": "sell", "confidence": float(sell_conf), "name": "crypto_momentum"}

        return {"action": "hold", "confidence": 0.0, "name": "crypto_momentum"}
