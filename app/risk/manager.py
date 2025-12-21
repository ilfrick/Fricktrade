class RiskManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.daily_loss = 0.0

    def can_open_trade(self, exposure_pct: float, short_exposure_pct: float, leverage: float) -> bool:
        if self.daily_loss >= self.cfg["max_daily_loss_pct"]:
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

    def should_circuit_break(self, drawdown_pct: float) -> bool:
        return drawdown_pct >= self.cfg["circuit_breaker_drawdown_pct"]
