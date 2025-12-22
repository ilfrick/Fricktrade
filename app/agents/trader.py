import logging
import time
from datetime import datetime
from pathlib import Path

from app.execution.executor import ExecutionEngine
from app.monitoring.metrics import (
    TRADES,
    SKIPPED_ORDERS,
    PNL,
    DRAWDOWN,
    ACCOUNT_TOTAL,
    ACCOUNT_CASH,
    ACCOUNT_INVESTED,
    SYMBOL_ACTIVE,
)
from app.risk.manager import RiskManager
from app.strategies.intraday_momentum import IntradayMomentumStrategy
from app.strategies.pattern_trading import PatternTradingStrategy
from app.data.news import fetch_catalyst_symbols
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
        self._news_cache: dict[str, bool] = {}
        self._news_cache_at: datetime | None = None

    def _build_strategy(self, params: dict):
        if self.learning_cfg.get("enabled"):
            model_path = self._select_model_path()
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
        if self.cfg["strategy"].get("name") == "pattern_trading":
            return PatternTradingStrategy(self.cfg.get("pattern_trading", {}))
        return IntradayMomentumStrategy(
            params["lookback_minutes"],
            params["entry_threshold_pct"],
            params["exit_threshold_pct"],
            params["allow_shorts"],
        )

    def _select_model_path(self) -> str:
        model_path = self.learning_cfg.get("model_path", "/app/models/ppo_policy.zip")
        if not self.learning_cfg.get("use_best_model", True):
            return model_path
        best_path = self.learning_cfg.get("best_model_path", "/app/models/ppo_policy_best.zip")
        return best_path if Path(best_path).exists() else model_path

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
        if action == "exit":
            self.broker.close_position(symbol)
            return None

        last_price = market_state.get("last_price")
        if last_price is None:
            prices = market_state.get("prices", [])
            last_price = prices[-1] if prices else None
        if last_price is None:
            SKIPPED_ORDERS.labels(symbol=symbol, side=action, reason="no_price").inc()
            logging.info("Skipping %s for %s: no price available", action, symbol)
            return None

        portfolio = market_state.get("portfolio", {})
        if action == "sell":
            reduce_pct = float(signal.get("reduce_pct", 1.0))
            qty, skip_reason = self._size_order(action, last_price, portfolio, symbol, reduce_pct=reduce_pct)
        else:
            qty, skip_reason = self._size_order(action, last_price, portfolio, symbol)
        if qty <= 0:
            if skip_reason:
                SKIPPED_ORDERS.labels(symbol=symbol, side=action, reason=skip_reason).inc()
                logging.info("Skipping %s for %s: %s", action, symbol, skip_reason)
            return None
        market_state["qty"] = qty

        if not self.risk.can_open_trade(
            exposure_pct=market_state.get("exposure_pct", 0.0),
            short_exposure_pct=market_state.get("short_exposure_pct", 0.0),
            leverage=market_state.get("leverage", 1.0),
        ):
            SKIPPED_ORDERS.labels(symbol=symbol, side=action, reason="risk_block").inc()
            logging.info("Skipping %s for %s: risk limits exceeded", action, symbol)
            return None

        order_id = self.executor.execute(symbol, action, qty=qty)
        if order_id and action in ("buy", "sell"):
            TRADES.labels(symbol=symbol, side=action).inc()
        return order_id

    def _size_order(
        self,
        action: str,
        last_price: float,
        portfolio: dict,
        symbol: str,
        reduce_pct: float = 1.0,
    ) -> tuple[int, str | None]:
        if last_price <= 0:
            return 0, "no_price"
        equity = float(portfolio.get("equity", 0.0) or 0.0)
        cash = float(portfolio.get("cash", 0.0) or 0.0)
        positions = portfolio.get("positions", {})
        current_qty = float(positions.get(symbol, {}).get("qty", 0.0) or 0.0)
        current_value = current_qty * last_price
        max_pos_pct = float(self.cfg["risk"]["max_position_size_pct"])
        max_short_pct = float(self.cfg["risk"]["max_short_exposure_pct"])
        allow_shorts = bool(self._strategy_params.get("allow_shorts", False))

        if action == "buy":
            if equity <= 0 or cash <= 0:
                return 0, "insufficient_cash"
            target_value = equity * (max_pos_pct / 100.0)
            remaining_value = max(0.0, target_value - max(current_value, 0.0))
            if remaining_value <= 0:
                return 0, "position_limit"
            allowed_value = min(remaining_value, cash)
            if allowed_value < last_price:
                return 0, "insufficient_cash"
            return int(allowed_value // last_price), None

        if action == "sell":
            if current_qty > 0:
                qty = int(current_qty * max(min(reduce_pct, 1.0), 0.0))
                return (qty, None) if qty > 0 else (0, "position_limit")
            if not allow_shorts or equity <= 0:
                return 0, "short_limit"
            short_limit = equity * (max_short_pct / 100.0)
            current_short = float(portfolio.get("short_exposure", 0.0) or 0.0)
            remaining_value = max(0.0, short_limit - current_short)
            if remaining_value < last_price:
                return 0, "short_limit"
            return int(remaining_value // last_price), None

        return 0, "unsupported"

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
            portfolio = self._get_portfolio_snapshot()
            for sym in symbols:
                SYMBOL_ACTIVE.labels(symbol=sym).set(1)
            self._update_account_metrics()
            self._refresh_news_cache(symbols)
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
                self._enrich_market_state(market_state, portfolio, sym)
                self.run_once(sym, market_state)
            time.sleep(interval_seconds)

    def _refresh_news_cache(self, symbols: list[str]) -> None:
        news_cfg = self.cfg.get("news", {})
        if not news_cfg.get("enabled", False):
            self._news_cache = {}
            return
        now = datetime.utcnow()
        ttl_minutes = int(news_cfg.get("cache_minutes", 15))
        if self._news_cache_at and (now - self._news_cache_at).total_seconds() < ttl_minutes * 60:
            return
        self._news_cache = fetch_catalyst_symbols(
            symbols=symbols,
            provider=news_cfg.get("provider", "alpaca"),
            base_url=news_cfg.get("base_url", "https://data.alpaca.markets"),
            api_key=news_cfg.get("api_key", ""),
            api_secret=news_cfg.get("api_secret", ""),
            lookback_hours=int(news_cfg.get("lookback_hours", 12)),
            keywords=news_cfg.get("keywords", []),
        )
        self._news_cache_at = now

    def _enrich_market_state(self, market_state: dict, portfolio: dict, symbol: str) -> None:
        equity = float(portfolio.get("equity", 0.0) or 0.0)
        positions = portfolio.get("positions", {})
        last_price = market_state.get("last_price")
        if last_price is None:
            prices = market_state.get("prices", [])
            last_price = prices[-1] if prices else None
        current_qty = float(positions.get(symbol, {}).get("qty", 0.0) or 0.0)
        current_value = current_qty * last_price if last_price else 0.0
        short_exposure = float(portfolio.get("short_exposure", 0.0) or 0.0)
        gross_exposure = float(portfolio.get("gross_exposure", 0.0) or 0.0)
        market_state["exposure_pct"] = (abs(current_value) / equity * 100.0) if equity else 0.0
        market_state["short_exposure_pct"] = (short_exposure / equity * 100.0) if equity else 0.0
        market_state["leverage"] = (gross_exposure / equity) if equity else 1.0
        market_state["portfolio"] = portfolio
        market_state["catalyst"] = self._news_cache.get(symbol, False)

    def _get_portfolio_snapshot(self) -> dict:
        account = self.broker.get_account()
        equity = cash = None
        if isinstance(account, dict):
            if "equity" in account:
                equity = account.get("equity")
                cash = account.get("cash")
            elif "NetLiquidation" in account:
                equity = account.get("NetLiquidation")
                cash = account.get("TotalCashValue")
        try:
            equity_val = float(equity) if equity is not None else 0.0
            cash_val = float(cash) if cash is not None else 0.0
        except (TypeError, ValueError):
            equity_val = 0.0
            cash_val = 0.0

        positions = {}
        gross_exposure = 0.0
        short_exposure = 0.0
        try:
            raw_positions = self.broker.get_positions()
        except Exception as exc:
            logging.warning("Position snapshot failed: %s", exc)
            raw_positions = []

        for pos in raw_positions:
            symbol = pos.get("symbol")
            if not symbol:
                continue
            qty = float(pos.get("qty") or pos.get("position") or 0.0)
            market_value = pos.get("market_value")
            if market_value is None:
                price = pos.get("current_price") or pos.get("market_price") or pos.get("avg_cost") or 0.0
                market_value = qty * float(price)
            else:
                market_value = float(market_value)
            positions[symbol] = {"qty": qty, "value": market_value}
            gross_exposure += abs(market_value)
            if market_value < 0:
                short_exposure += abs(market_value)

        return {
            "equity": equity_val,
            "cash": cash_val,
            "positions": positions,
            "gross_exposure": gross_exposure,
            "short_exposure": short_exposure,
        }
