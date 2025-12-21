from datetime import datetime, timedelta

from app.strategies.base import Strategy


class IntradayMomentumStrategy(Strategy):
    def __init__(self, lookback_minutes: int, entry_threshold_pct: float, exit_threshold_pct: float, allow_shorts: bool):
        self.lookback_minutes = lookback_minutes
        self.entry_threshold_pct = entry_threshold_pct
        self.exit_threshold_pct = exit_threshold_pct
        self.allow_shorts = allow_shorts

    def generate_signal(self, market_state: dict) -> dict:
        prices = market_state.get("prices", [])
        if len(prices) < 2:
            return {"action": "hold"}

        start_price = prices[0]
        latest_price = prices[-1]
        change_pct = (latest_price - start_price) / start_price * 100.0

        if change_pct >= self.entry_threshold_pct:
            return {"action": "buy", "confidence": min(change_pct / 5.0, 1.0)}
        if change_pct <= -self.entry_threshold_pct and self.allow_shorts:
            return {"action": "sell", "confidence": min(abs(change_pct) / 5.0, 1.0)}
        if abs(change_pct) <= self.exit_threshold_pct:
            return {"action": "exit"}
        return {"action": "hold"}
