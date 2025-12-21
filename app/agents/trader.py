import time

from app.execution.executor import ExecutionEngine
from app.monitoring.metrics import TRADES, PNL, DRAWDOWN
from app.risk.manager import RiskManager
from app.strategies.intraday_momentum import IntradayMomentumStrategy


class TradingAgent:
    def __init__(self, broker, cfg: dict):
        self.cfg = cfg
        self.broker = broker
        self.risk = RiskManager(cfg["risk"])
        params = cfg["strategy"]["params"]
        self.strategy = IntradayMomentumStrategy(
            params["lookback_minutes"],
            params["entry_threshold_pct"],
            params["exit_threshold_pct"],
            params["allow_shorts"],
        )
        self.executor = ExecutionEngine(broker)

    def run_once(self, symbol: str, market_state: dict):
        signal = self.strategy.generate_signal(market_state)
        action = signal.get("action", "hold")
        if action == "hold":
            return None

        if not self.risk.can_open_trade(
            exposure_pct=market_state.get("exposure_pct", 0.0),
            short_exposure_pct=market_state.get("short_exposure_pct", 0.0),
            leverage=market_state.get("leverage", 1.0),
        ):
            return None

        order_id = self.executor.execute(symbol, action, qty=market_state.get("qty", 1))
        if order_id and action in ("buy", "sell"):
            TRADES.labels(symbol=symbol, side=action).inc()
        return order_id

    def loop(self, symbol: str, market_data_provider, interval_seconds: int = 60):
        while True:
            market_state = market_data_provider(symbol)
            self.run_once(symbol, market_state)
            time.sleep(interval_seconds)
