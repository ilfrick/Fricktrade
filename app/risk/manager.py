# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import threading
from datetime import date


class RiskManager:
    """
    Manages trading risk limits including daily loss, position sizing, and circuit breakers.

    Thread-safe: All state mutations are protected by a lock.

    Config keys:
        - enabled: bool - Enable/disable risk checks
        - max_daily_loss_pct: float - Maximum daily loss percentage (e.g., 2.0 for 2%)
        - max_position_size_pct: float - Maximum single position size as % of portfolio
        - max_short_exposure_pct: float - Maximum short exposure as % of portfolio
        - max_portfolio_leverage: float - Maximum portfolio leverage
        - circuit_breaker_drawdown_pct: float - Drawdown % that triggers circuit breaker
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._daily_loss = 0.0
        self._last_reset_date: date | None = None
        self._lock = threading.Lock()

    def _enabled(self) -> bool:
        return bool(self.cfg.get("enabled", True))

    def can_open_trade(self, exposure_pct: float, short_exposure_pct: float, leverage: float) -> bool:
        """
        Check if a new trade can be opened given current risk parameters.

        Args:
            exposure_pct: Current position exposure as percentage
            short_exposure_pct: Current short exposure as percentage
            leverage: Current portfolio leverage

        Returns:
            True if trade is allowed, False otherwise
        """
        if not self._enabled():
            return True

        with self._lock:
            self._maybe_reset_daily()

            max_daily_loss = float(self.cfg.get("max_daily_loss_pct", 0.0) or 0.0)
            if max_daily_loss > 0 and self._daily_loss <= -max_daily_loss:
                return False

            max_position = float(self.cfg.get("max_position_size_pct", 100.0) or 100.0)
            if exposure_pct >= max_position:
                return False

            max_short = float(self.cfg.get("max_short_exposure_pct", 100.0) or 100.0)
            if short_exposure_pct >= max_short:
                return False

            max_leverage = float(self.cfg.get("max_portfolio_leverage", 10.0) or 10.0)
            if leverage >= max_leverage:
                return False

        return True

    def record_pnl(self, pnl_pct: float) -> None:
        """
        Record a P&L change for daily loss tracking.

        Args:
            pnl_pct: P&L change as percentage (negative for losses)
        """
        with self._lock:
            self._maybe_reset_daily()
            self._daily_loss = min(0.0, self._daily_loss + float(pnl_pct))

    def update_daily_loss(self, pnl_pct: float) -> None:
        """Alias for record_pnl for backward compatibility."""
        self.record_pnl(pnl_pct)

    def reset_daily(self) -> None:
        """Manually reset daily loss tracking to zero."""
        with self._lock:
            self._daily_loss = 0.0
            self._last_reset_date = date.today()

    def _maybe_reset_daily(self) -> None:
        """Auto-reset daily loss at start of new trading day. Must be called with lock held."""
        today = date.today()
        if self._last_reset_date != today:
            self._daily_loss = 0.0
            self._last_reset_date = today

    @property
    def daily_loss(self) -> float:
        """Current daily loss percentage (thread-safe read)."""
        with self._lock:
            return self._daily_loss

    def should_circuit_break(self, drawdown_pct: float) -> bool:
        """
        Check if circuit breaker should be triggered.

        Args:
            drawdown_pct: Current drawdown percentage

        Returns:
            True if circuit breaker should trigger
        """
        if not self._enabled():
            return False
        threshold = float(self.cfg.get("circuit_breaker_drawdown_pct", 100.0) or 100.0)
        return drawdown_pct >= threshold
