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

    @staticmethod
    def _ema(arr: np.ndarray, span: int) -> float:
        alpha = 2.0 / (span + 1.0)
        result = float(arr[0])
        for val in arr[1:]:
            result = alpha * float(val) + (1.0 - alpha) * result
        return result

    def generate_signal(self, market_state: dict) -> dict:
        prices = market_state.get("prices", []) or []
        if len(prices) < max(self.params.fast_window, self.params.slow_window) + 1:
            return {"action": "hold"}
        close = np.array(prices, dtype=float)
        # Use pre-computed EMA from indicator injection if available; otherwise compute inline
        indicators = market_state.get("indicators") or {}
        if indicators.get("ema_fast") and indicators.get("ema_slow"):
            fast = float(indicators["ema_fast"])
            slow = float(indicators["ema_slow"])
        else:
            prices_for_ema = close[-(self.params.slow_window * 3):]
            fast = self._ema(prices_for_ema, self.params.fast_window)
            slow = self._ema(prices_for_ema, self.params.slow_window)
        last = float(close[-1])
        if slow <= 0:
            return {"action": "hold"}
        trend_strength = (fast - slow) / slow * 100.0

        # RSI(14) — compute inline
        rsi = self._compute_rsi(close)

        # Volume confirmation — skipped for crypto (Alpaca bar volumes are platform-only,
        # not global exchange volume, and are unreliably thin on weekends).
        symbol = market_state.get("symbol", "")
        is_crypto = "/" in symbol
        volumes = market_state.get("volumes", []) or []
        vol_ok = True
        if not is_crypto and len(volumes) >= 5:
            avg_vol = float(np.mean(volumes[-5:-1])) if len(volumes) > 1 else 0.0
            vol_ok = avg_vol <= 0 or volumes[-1] >= avg_vol * 1.5

        # Extended indicators (when available from indicator injection; already fetched above)
        supertrend = indicators.get("supertrend")  # 1=bullish, -1=bearish
        vwap_dev = indicators.get("vwap_dev")  # >0 means price above VWAP

        # Regime filter
        regime_name = market_state.get("regime_name")
        in_crisis = regime_name == "high_vol_crisis"

        # --- BUY signal ---
        if trend_strength >= self.params.breakout_pct and last >= fast and rsi < 70 and vol_ok:
            if in_crisis:
                return {"action": "hold", "confidence": 0.0, "trend_strength": trend_strength, "rsi": rsi}
            confidence = self._compute_confidence(trend_strength, rsi, supertrend, vwap_dev, is_buy=True)
            return {"action": "buy", "confidence": confidence, "trend_strength": trend_strength, "rsi": rsi}

        # --- SELL signal ---
        if trend_strength <= -self.params.exit_pct and last <= fast and rsi > 30:
            confidence = self._compute_confidence(trend_strength, rsi, supertrend, vwap_dev, is_buy=False)
            return {"action": "sell", "confidence": confidence, "trend_strength": trend_strength, "rsi": rsi}

        # --- Extended exit signals (indicator-based, only when holding a position) ---
        if indicators:
            _portfolio = market_state.get("portfolio") or {}
            _held_qty = float((_portfolio.get("positions") or {}).get(symbol, {}).get("qty", 0.0))
            if _held_qty > 0:
                # Overbought RSI exit
                if rsi > 75:
                    return {"action": "sell", "confidence": 0.6, "trend_strength": trend_strength, "rsi": rsi}
                # Bearish supertrend flip
                if supertrend is not None and supertrend == -1 and trend_strength < 0:
                    return {"action": "sell", "confidence": 0.5, "trend_strength": trend_strength, "rsi": rsi}

        return {"action": "hold", "confidence": 0.0, "trend_strength": trend_strength, "rsi": rsi}

    @staticmethod
    def _compute_rsi(close: np.ndarray, period: int = 14) -> float:
        return Strategy._wilder_rsi(list(close), period)

    def _compute_confidence(
        self,
        trend_strength: float,
        rsi: float,
        supertrend: float | None,
        vwap_dev: float | None,
        is_buy: bool,
    ) -> float:
        # Base confidence from trend strength
        base = float(min(abs(trend_strength) / max(self.params.breakout_pct * 3.0, 0.01), 1.0))
        score = base

        # RSI distance from extremes (higher confidence when not near overbought/oversold)
        if is_buy:
            rsi_bonus = max(0.0, (70.0 - rsi) / 70.0) * 0.15
        else:
            rsi_bonus = max(0.0, (rsi - 30.0) / 70.0) * 0.15
        score += rsi_bonus

        # Supertrend alignment bonus
        if supertrend is not None:
            if (is_buy and supertrend == 1) or (not is_buy and supertrend == -1):
                score += 0.15  # aligned
            elif (is_buy and supertrend == -1) or (not is_buy and supertrend == 1):
                score -= 0.10  # conflicting

        # VWAP deviation bonus
        if vwap_dev is not None:
            if (is_buy and vwap_dev > 0) or (not is_buy and vwap_dev < 0):
                score += 0.10  # price on right side of VWAP

        return float(np.clip(score, 0.0, 1.0))
