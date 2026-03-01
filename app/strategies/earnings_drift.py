# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""
Post-Earnings Announcement Drift (PEAD) strategy.

Buys stocks that gap up on the day after an earnings surprise, riding the
momentum that tends to persist for 2-4 weeks after a positive surprise.

Entry conditions:
  - earnings_window == 'post'  (within 30 days after earnings date)
  - session_gain_pct >= min_gap_pct  (opening gap on the announcement day)
  - volume >= min_volume_mult × average volume  (confirms institutional participation)
  - Crypto symbols (containing '/') are skipped

Exit: trailing stop or hold_days time-based exit (handled by risk manager).
"""

from __future__ import annotations

from app.strategies.base import Strategy


class EarningsDriftStrategy(Strategy):
    """PEAD: buy stocks with positive gap after earnings surprise."""

    def __init__(self, params: dict):
        cfg = params.get("earnings_drift", {}) if isinstance(params, dict) else {}
        self._min_gap_pct = float(cfg.get("min_gap_pct", 5.0))
        self._min_volume_mult = float(cfg.get("min_volume_mult", 1.5))
        self._hold_days = int(cfg.get("hold_days", 20))
        self._trailing_stop_pct = float(cfg.get("trailing_stop_pct", 4.0))

    def generate_signal(self, market_state: dict) -> dict:
        symbol = market_state.get("symbol", "")
        # Equity-only
        if "/" in symbol:
            return {"action": "hold", "confidence": 0.0, "name": "earnings_drift"}

        earnings_window = market_state.get("earnings_window", "none")
        if earnings_window != "post":
            return {"action": "hold", "confidence": 0.0, "name": "earnings_drift",
                    "reason": f"earnings_window={earnings_window}"}

        gap_pct = float(market_state.get("session_gain_pct", 0.0) or 0.0)
        rel_volume = float(market_state.get("relative_volume", 1.0) or 1.0)

        if gap_pct < self._min_gap_pct:
            return {"action": "hold", "confidence": 0.0, "name": "earnings_drift",
                    "reason": f"gap_pct={gap_pct:.2f} < {self._min_gap_pct}"}

        if rel_volume < self._min_volume_mult:
            return {"action": "hold", "confidence": 0.0, "name": "earnings_drift",
                    "reason": f"rel_vol={rel_volume:.2f} < {self._min_volume_mult}"}

        # Confidence proportional to gap size, capped at 1.0
        # Larger gap → higher surprise → stronger drift signal
        raw_conf = min(gap_pct / (self._min_gap_pct * 3.0), 1.0)
        # Volume boost: extra 20% confidence for very high volume
        if rel_volume >= self._min_volume_mult * 2:
            raw_conf = min(raw_conf + 0.1, 1.0)

        return {
            "action": "buy",
            "confidence": round(raw_conf, 3),
            "name": "earnings_drift",
            "reason": f"PEAD gap={gap_pct:.1f}% rel_vol={rel_volume:.1f}x",
            "hold_days": self._hold_days,
            "trailing_stop_pct": self._trailing_stop_pct,
        }
