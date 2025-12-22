import logging
import time

from app.execution.executor import ExecutionEngine
from app.monitoring.metrics import TRADES, PNL, DRAWDOWN
from app.risk.manager import RiskManager
from app.strategies.intraday_momentum import IntradayMomentumStrategy
from app.strategies.rl_policy import RLPolicyStrategy
from app.utils.market import is_market_open


class TradingAgent:
    def __init__(self, broker, cfg: dict):
        self.cfg = cfg
        self.broker = broker
        self.risk = RiskManager(cfg["risk"])
        self.learning_cfg = cfg.get("learning", {})
        params = cfg["strategy"]["params"]
        self.strategy = self._build_strategy(params)
        self.guardrail = self._build_guardrail(params)
        self.executor = ExecutionEngine(broker)
        self._last_market_open = None

    def _build_strategy(self, params: dict):
        if self.learning_cfg.get("enabled"):
            model_path = self.learning_cfg.get("model_path", "/app/models/ppo_policy.zip")
            window_size = int(self.learning_cfg.get("window_size", 50))
            device = self.learning_cfg.get("device", "auto")
            feature_config = self.learning_cfg.get("features", {})
            try:
                return RLPolicyStrategy(
                    model_path,
                    window_size=window_size,
                    device=device,
                    feature_config=feature_config,
                )
            except FileNotFoundError as exc:
                logging.warning("RL model unavailable, falling back to rule-based strategy: %s", exc)
        return IntradayMomentumStrategy(
            params["lookback_minutes"],
            params["entry_threshold_pct"],
            params["exit_threshold_pct"],
            params["allow_shorts"],
        )

    def _build_guardrail(self, params: dict):
        guard_cfg = self.learning_cfg.get("guardrail", {})
        if not guard_cfg.get("enabled"):
            return None
        guard_params = guard_cfg.get("params", params)
        return IntradayMomentumStrategy(
            guard_params.get("lookback_minutes", params["lookback_minutes"]),
            guard_params.get("entry_threshold_pct", params["entry_threshold_pct"]),
            guard_params.get("exit_threshold_pct", params["exit_threshold_pct"]),
            guard_params.get("allow_shorts", params["allow_shorts"]),
        )

    def _apply_guardrail(self, action: str, guard_action: str, mode: str) -> str:
        if action not in ("buy", "sell"):
            return action
        if mode == "confirm":
            return action if guard_action == action else "hold"
        if mode == "veto":
            if guard_action in ("hold", "exit") or guard_action != action:
                return "hold"
        return action

    def run_once(self, symbol: str, market_state: dict):
        signal = self.strategy.generate_signal(market_state)
        action = signal.get("action", "hold")
        if self.guardrail:
            guard_action = self.guardrail.generate_signal(market_state).get("action", "hold")
            mode = self.learning_cfg.get("guardrail", {}).get("mode", "confirm")
            action = self._apply_guardrail(action, guard_action, mode)
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
            market_open = is_market_open(self.cfg)
            if market_open != self._last_market_open:
                state = "open" if market_open else "closed"
                logging.info("Market is %s; %s trading loop.", state, "starting" if market_open else "waiting")
                self._last_market_open = market_open
            if not market_open:
                time.sleep(interval_seconds)
                continue
            market_state = market_data_provider(symbol)
            self.run_once(symbol, market_state)
            time.sleep(interval_seconds)
