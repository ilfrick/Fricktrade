# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import threading
from datetime import date, datetime, timezone


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

    def __init__(self, cfg: dict, tz: timezone | None = None):
        self.cfg = cfg
        self._tz = tz or timezone.utc
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
            self._last_reset_date = datetime.now(self._tz).date()

    def _maybe_reset_daily(self) -> None:
        """Auto-reset daily loss at start of new trading day. Must be called with lock held."""
        today = datetime.now(self._tz).date()
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

    def check_cooldown(self, last_trade_at: datetime | None, now: datetime) -> tuple[bool, str | None]:
        """Check if cooldown period is active since last trade.

        Returns:
            (is_blocked, reason) — reason is "cooldown" if blocked, else None.
        """
        cooldown = int(self.cfg.get("cooldown_seconds", 0))
        if cooldown <= 0 or not last_trade_at:
            return False, None
        elapsed = (now - last_trade_at).total_seconds()
        if elapsed < cooldown:
            return True, "cooldown"
        return False, None

    def check_order_limits(
        self, qty: int, last_price: float, limits_cfg: dict
    ) -> tuple[bool, str | None]:
        """Check if an order violates qty/notional limits.

        Args:
            qty: Order quantity.
            last_price: Last price of the symbol.
            limits_cfg: The ``trading_limits`` config dict.

        Returns:
            (violates, reason) — reason is "order_limit" if violated.
        """
        if not limits_cfg.get("enabled", False):
            return False, None
        max_qty = limits_cfg.get("max_order_qty")
        if max_qty is not None and qty > float(max_qty):
            return True, "order_limit"
        notional = qty * last_price
        max_notional = limits_cfg.get("max_order_notional")
        if max_notional is not None and notional > float(max_notional):
            return True, "order_limit"
        min_notional = limits_cfg.get("min_order_notional")
        if min_notional is not None and notional < float(min_notional):
            return True, "order_limit"
        return False, None

    def risk_preflight(
        self,
        qty: int,
        last_price: float,
        limits_cfg: dict,
        last_trade_at: datetime | None,
        now: datetime,
        exposure_pct: float = 0.0,
        short_exposure_pct: float = 0.0,
        leverage: float = 1.0,
    ) -> tuple[bool, str | None]:
        """Combined entry risk check: order limits + cooldown + can_open_trade.

        Returns:
            (blocked, reason) — reason string if blocked, else None.
        """
        violates, reason = self.check_order_limits(qty, last_price, limits_cfg)
        if violates:
            return True, "order_limit"
        blocked, _ = self.check_cooldown(last_trade_at, now)
        if blocked:
            return True, "cooldown"
        if not self.can_open_trade(exposure_pct, short_exposure_pct, leverage):
            return True, "risk_block"
        return False, None

    @staticmethod
    def check_exposure_caps(
        symbol: str,
        action: str,
        qty: int,
        last_price: float,
        portfolio: dict,
        caps_cfg: dict,
        venue_fn=None,
        sector_fn=None,
    ) -> tuple[bool, str | None]:
        """Check if an order would violate venue/sector exposure caps.

        Args:
            symbol: The symbol being traded.
            action: "buy" or "sell".
            qty: Order quantity.
            last_price: Last price.
            portfolio: Portfolio snapshot dict.
            caps_cfg: The ``risk.exposure_caps`` config dict.
            venue_fn: Callable(symbol) -> venue name or None.
            sector_fn: Callable(symbol) -> sector name or None.

        Returns:
            (violates, reason) — reason is "exposure_cap" if violated.
        """
        if not caps_cfg.get("enabled", False):
            return False, None
        if action not in ("buy", "sell"):
            return False, None
        delta = qty * last_price
        positions = portfolio.get("positions", {})
        current_qty = float(positions.get(symbol, {}).get("qty", 0.0) or 0.0)
        if action == "sell" and current_qty > 0:
            delta = -min(delta, current_qty * last_price)
        equity = float(portfolio.get("equity", 0.0) or 0.0)
        if equity <= 0:
            return False, None
        venue_caps = caps_cfg.get("venues", {}) or {}
        sector_caps = caps_cfg.get("sectors", {}) or {}
        if venue_caps and venue_fn is not None:
            exposure = _group_exposure(portfolio, venue_fn)
            venue = venue_fn(symbol)
            if venue:
                exposure[venue] = exposure.get(venue, 0.0) + abs(delta)
            for venue_name, cap in venue_caps.items():
                if exposure.get(venue_name, 0.0) / equity * 100.0 > float(cap):
                    return True, "exposure_cap"
        if sector_caps and sector_fn is not None:
            exposure = _group_exposure(portfolio, sector_fn)
            sector = sector_fn(symbol)
            if sector:
                exposure[sector] = exposure.get(sector, 0.0) + abs(delta)
            for sector_name, cap in sector_caps.items():
                if exposure.get(sector_name, 0.0) / equity * 100.0 > float(cap):
                    return True, "exposure_cap"
        return False, None


def _group_exposure(portfolio: dict, mapper) -> dict[str, float]:
    """Aggregate position exposure by group (venue/sector)."""
    positions = portfolio.get("positions", {})
    exposure: dict[str, float] = {}
    for symbol, pos in positions.items():
        group = mapper(symbol)
        if not group:
            continue
        value = abs(float(pos.get("value", 0.0) or 0.0))
        exposure[group] = exposure.get(group, 0.0) + value
    return exposure
