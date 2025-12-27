from __future__ import annotations

from dataclasses import dataclass

from app.strategies.base import Strategy


@dataclass
class MarketMakerParams:
    base_spread_pct: float = 0.4
    inventory_target_pct: float = 0.0
    skew_pct: float = 0.15
    min_qty: int = 1


class MarketMakerStrategy(Strategy):
    def __init__(self, params: dict):
        cfg = params.get("market_maker", {}) if isinstance(params, dict) else {}
        self.params = MarketMakerParams(
            base_spread_pct=float(cfg.get("base_spread_pct", 0.4)),
            inventory_target_pct=float(cfg.get("inventory_target_pct", 0.0)),
            skew_pct=float(cfg.get("skew_pct", 0.15)),
            min_qty=int(cfg.get("min_qty", 1)),
        )

    def generate_signal(self, market_state: dict) -> dict:
        last_price = market_state.get("last_price")
        if not last_price:
            return {"action": "hold"}
        spread_pct = market_state.get("spread_pct")
        if spread_pct is None:
            spread_pct = self.params.base_spread_pct
        inventory_pct = float(market_state.get("exposure_pct", 0.0) or 0.0)
        inventory_delta = inventory_pct - self.params.inventory_target_pct
        skew = self.params.skew_pct * (1 if inventory_delta > 0 else -1 if inventory_delta < 0 else 0)
        bid = last_price * (1 - (spread_pct / 100.0) / 2 - skew / 100.0)
        ask = last_price * (1 + (spread_pct / 100.0) / 2 - skew / 100.0)
        if inventory_delta > 0:
            return {"action": "sell", "order_type": "limit", "limit_price": ask}
        if inventory_delta < 0:
            return {"action": "buy", "order_type": "limit", "limit_price": bid}
        return {"action": "hold"}
