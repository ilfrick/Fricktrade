# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from app.strategies.base import Strategy


@dataclass
class GapReversalParams:
    min_gap_pct: float = 2.0
    target_fill_pct: float = 50.0
    stop_pct: float = 0.5
    volume_confirm_mult: float = 2.0
    max_trade_window_minutes: int = 60
    rsi_period: int = 5
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0


class GapReversalStrategy(Strategy):
    """Opening gap reversal for US equities.

    Gap-down reversals (BUY):
    - Gap down > min_gap_pct vs previous close
    - Volume in first bars > volume_confirm_mult × avg
    - RSI(rsi_period) < rsi_oversold
    - Trade window: first max_trade_window_minutes minutes after open
    - Target: target_fill_pct % of the gap. Stop: stop_pct below open.

    Gap-up reversals (EXIT):
    - Gap up > min_gap_pct with RSI > rsi_overbought → signal sell on held position.

    Equity-only: skips crypto symbols (containing '/').
    """

    def __init__(self, params: dict):
        cfg = params.get("gap_reversal", {}) if isinstance(params, dict) else {}
        self.params = GapReversalParams(
            min_gap_pct=float(cfg.get("min_gap_pct", 2.0)),
            target_fill_pct=float(cfg.get("target_fill_pct", 50.0)),
            stop_pct=float(cfg.get("stop_pct", 0.5)),
            volume_confirm_mult=float(cfg.get("volume_confirm_mult", 2.0)),
            max_trade_window_minutes=int(cfg.get("max_trade_window_minutes", 60)),
            rsi_period=int(cfg.get("rsi_period", 5)),
            rsi_oversold=float(cfg.get("rsi_oversold", 30.0)),
            rsi_overbought=float(cfg.get("rsi_overbought", 70.0)),
        )

    def generate_signal(self, market_state: dict) -> dict:
        symbol = market_state.get("symbol", "")
        if "/" in symbol:
            # Crypto has no defined open — skip
            return {"action": "hold", "confidence": 0.0, "name": "gap_reversal"}

        prices = market_state.get("prices", []) or []
        if len(prices) < self.params.rsi_period + 2:
            return {"action": "hold", "confidence": 0.0, "name": "gap_reversal"}

        close = np.array(prices, dtype=float)
        last = float(close[-1])

        # Previous session close (first price in today's bar sequence is the open)
        prev_close = market_state.get("prev_close")
        session_open = market_state.get("session_open")
        if prev_close is None or session_open is None or float(prev_close) <= 0:
            return {"action": "hold", "confidence": 0.0, "name": "gap_reversal"}

        prev_close = float(prev_close)
        session_open = float(session_open)
        gap_pct = (session_open - prev_close) / prev_close * 100.0

        # Trade window check (must be within first max_trade_window_minutes of session)
        now_utc = datetime.now(timezone.utc)
        session_open_time = market_state.get("session_open_time")
        if session_open_time is not None:
            try:
                elapsed_minutes = (now_utc - session_open_time).total_seconds() / 60.0
                if elapsed_minutes > self.params.max_trade_window_minutes:
                    return {"action": "hold", "confidence": 0.0, "name": "gap_reversal"}
            except Exception:
                pass

        # RSI on short period for intraday
        rsi = self._compute_rsi(close, self.params.rsi_period)

        # Volume confirmation
        volumes = market_state.get("volumes", []) or []
        vol_ok = False
        if len(volumes) >= 5:
            avg_vol = float(np.mean(volumes[-6:-1])) if len(volumes) > 5 else float(np.mean(volumes[:-1]))
            vol_ok = avg_vol > 0 and float(volumes[-1]) >= avg_vol * self.params.volume_confirm_mult

        # Gap DOWN reversal → BUY signal
        if gap_pct <= -self.params.min_gap_pct and rsi < self.params.rsi_oversold and vol_ok:
            gap_size = abs(gap_pct)
            confidence = min(0.4 + (gap_size - self.params.min_gap_pct) / self.params.min_gap_pct * 0.3, 0.9)
            target = session_open + abs(session_open - prev_close) * (self.params.target_fill_pct / 100.0)
            return {
                "action": "buy",
                "confidence": float(confidence),
                "name": "gap_reversal",
                "take_profit_price": target,
                "hard_stop_pct": self.params.stop_pct,
            }

        # Gap UP reversal → EXIT signal (if holding a position)
        position_qty = float((market_state.get("positions") or {}).get(symbol, {}).get("qty", 0.0))
        if gap_pct >= self.params.min_gap_pct and rsi > self.params.rsi_overbought and position_qty > 0 and vol_ok:
            return {"action": "sell", "confidence": 0.65, "name": "gap_reversal"}

        return {"action": "hold", "confidence": 0.0, "name": "gap_reversal"}

    @staticmethod
    def _compute_rsi(close: np.ndarray, period: int = 14) -> float:
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close[-(period + 1) :])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - 100.0 / (1.0 + rs))
