import logging
import time
from datetime import datetime

from app.execution.executor import ExecutionEngine
from app.monitoring.metrics import (
    TRADES,
    PNL,
    DRAWDOWN,
    ACCOUNT_TOTAL,
    ACCOUNT_CASH,
    ACCOUNT_INVESTED,
    SYMBOL_ACTIVE,
)
from app.risk.manager import RiskManager
from app.strategies.intraday_momentum import IntradayMomentumStrategy
from app.strategies.rl_policy import RLPolicyStrategy
from app.utils.market import is_market_open
from app.utils.restart import should_restart


class TradingAgent:
    def __init__(self, broker, cfg: dict):
        self.cfg = cfg
        self.broker = broker
        self.risk = RiskManager(cfg["risk"])
        self.learning_cfg = cfg.get("learning", {})
        params = cfg["strategy"]["params"]
        self._strategy_params = params
        self._strategy_by_symbol: dict[str, object] = {}
        self._guardrail_by_symbol: dict[str, object] = {}
        self.executor = ExecutionEngine(broker)
        self._last_market_open = None
        self._started_at = datetime.utcnow()

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

    def _get_strategy(self, symbol: str):
        if symbol not in self._strategy_by_symbol:
            self._strategy_by_symbol[symbol] = self._build_strategy(self._strategy_params)
        return self._strategy_by_symbol[symbol]

    def _get_guardrail(self, symbol: str):
        if symbol not in self._guardrail_by_symbol:
            self._guardrail_by_symbol[symbol] = self._build_guardrail(self._strategy_params)
        return self._guardrail_by_symbol[symbol]

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
        strategy = self._get_strategy(symbol)
        signal = strategy.generate_signal(market_state)
        action = signal.get("action", "hold")
        guardrail = self._get_guardrail(symbol)
        if guardrail:
            guard_action = guardrail.generate_signal(market_state).get("action", "hold")
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

    def _update_account_metrics(self) -> None:
        try:
            account = self.broker.get_account()
        except Exception as exc:
            logging.warning("Account metrics update failed: %s", exc)
            return
        total = cash = None
        if isinstance(account, dict):
            if "equity" in account:
                total = account.get("equity")
                cash = account.get("cash")
            elif "NetLiquidation" in account:
                total = account.get("NetLiquidation")
                cash = account.get("TotalCashValue")
        try:
            total_val = float(total) if total is not None else None
            cash_val = float(cash) if cash is not None else None
        except (TypeError, ValueError):
            return
        if total_val is None or cash_val is None:
            return
        ACCOUNT_TOTAL.set(total_val)
        ACCOUNT_CASH.set(cash_val)
        ACCOUNT_INVESTED.set(total_val - cash_val)

    def loop(self, symbol: str | list[str], market_data_provider, interval_seconds: int = 60):
        symbols = symbol if isinstance(symbol, list) else [symbol]
        while True:
            if should_restart(self._started_at):
                logging.info("Restart requested; exiting trading loop.")
                raise SystemExit(0)
            for sym in symbols:
                SYMBOL_ACTIVE.labels(symbol=sym).set(1)
            self._update_account_metrics()
            market_open = is_market_open(self.cfg)
            if market_open != self._last_market_open:
                state = "open" if market_open else "closed"
                logging.info("Market is %s; %s trading loop.", state, "starting" if market_open else "waiting")
                self._last_market_open = market_open
            if not market_open:
                time.sleep(interval_seconds)
                continue
            for sym in symbols:
                market_state = market_data_provider(sym)
                self.run_once(sym, market_state)
            time.sleep(interval_seconds)
