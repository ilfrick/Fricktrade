# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

class RiskManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.daily_loss = 0.0

    def _enabled(self) -> bool:
        return bool(self.cfg.get("enabled", True))

    def can_open_trade(self, exposure_pct: float, short_exposure_pct: float, leverage: float) -> bool:
        if not self._enabled():
            return True
        max_daily_loss = float(self.cfg.get("max_daily_loss_pct", 0.0) or 0.0)
        if max_daily_loss > 0 and self.daily_loss <= -max_daily_loss:
            return False
        if exposure_pct >= self.cfg["max_position_size_pct"]:
            return False
        if short_exposure_pct >= self.cfg["max_short_exposure_pct"]:
            return False
        if leverage >= self.cfg["max_portfolio_leverage"]:
            return False
        return True

    def record_pnl(self, pnl_pct: float) -> None:
        self.daily_loss = min(0.0, self.daily_loss + pnl_pct)

    def update_daily_loss(self, pnl_pct: float) -> None:
        self.daily_loss = min(0.0, float(pnl_pct))

    def should_circuit_break(self, drawdown_pct: float) -> bool:
        if not self._enabled():
            return False
        return drawdown_pct >= self.cfg["circuit_breaker_drawdown_pct"]
