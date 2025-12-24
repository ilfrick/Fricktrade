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
    STRATEGY_ACTIVE,
    ORCHESTRATOR_STRATEGY_ACTIVE,
    OPEN_ORDERS,
    BROKER_ACTIVE,
)
from app.risk.manager import RiskManager
from app.strategies.intraday_momentum import IntradayMomentumStrategy
from app.strategies.pattern_trading import PatternTradingStrategy
from app.data.news import fetch_catalyst_symbols
from app.data.scanner import ScanFilters, load_universe, scan_symbols
from app.strategies.rl_policy import RLPolicyStrategy
from app.strategies.rl_policy_fees import FeeAwareRLPolicyStrategy
from app.utils.market import is_market_open
from app.utils.restart import should_restart
from app.agents.orchestrator import StrategyOrchestrator, MLStrategyOrchestrator


class TradingAgent:
    def __init__(self, broker, cfg: dict):
        self.cfg = cfg
        self.broker = broker
        self.risk = RiskManager(cfg["risk"])
        self.learning_cfg = cfg.get("learning", {})
        self._account_snapshot: dict[str, object] = {}
        params = cfg["strategy"]["params"]
        self._strategy_params = params
        self._strategy_by_symbol: dict[str, dict[str, object]] = {}
        self._guardrail_by_symbol: dict[str, object] = {}
        self.executor = ExecutionEngine(broker)
        self._last_market_open = None
        self._started_at = datetime.utcnow()
        self._news_cache: dict[str, bool] = {}
        self._news_cache_at: datetime | None = None
        self._strategy_names = self._resolve_strategy_names()
        self._combine_mode = cfg["strategy"].get("combine", "priority")
        orchestrator_cfg = cfg.get("orchestrator", {})
        if orchestrator_cfg.get("ml", {}).get("enabled", False):
            self._orchestrator = MLStrategyOrchestrator(orchestrator_cfg)
        else:
            self._orchestrator = StrategyOrchestrator(orchestrator_cfg)
        self._open_orders_cache: list[dict] = []
        self._open_orders_at: datetime | None = None
        self._broker_name = self._resolve_broker_name()
        self._dynamic_symbols_at: datetime | None = None
        self._symbols: list[str] = []
        self._symbols_by_strategy: dict[str, list[str]] = {}
        self._orchestrator_state: dict[str, dict[str, object]] = {}
        self._last_trade_at: datetime | None = None
        if isinstance(self._orchestrator, MLStrategyOrchestrator):
            self._orchestrator.bootstrap(
                self._strategy_names,
                self._build_strategy,
                self._strategy_params,
                cfg.get("data", {}),
            )

    def _build_strategy(self, name: str, params: dict):
        if name == "rl_policy":
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
                    logging.warning("RL model unavailable, skipping rl_policy: %s", exc)
            return None
        if name == "rl_policy_fees":
            if self.learning_cfg.get("enabled"):
                model_path = self._select_model_path()
                window_size = int(self.learning_cfg.get("window_size", 50))
                device = self.learning_cfg.get("device", "auto")
                feature_config = self.learning_cfg.get("features", {})
                broker_fees = self.cfg.get("brokers", {}).get(self._broker_name, {}).get("fees", {})
                fee_guard = self.cfg.get("strategy", {}).get("fee_aware", {})
                risk_cfg = self.cfg.get("risk", {})
                try:
                    return FeeAwareRLPolicyStrategy(
                        model_path,
                        window_size=window_size,
                        device=device,
                        feature_config=feature_config,
                        broker_fees=broker_fees,
                        fee_guard=fee_guard,
                        risk_cfg=risk_cfg,
                    )
                except FileNotFoundError as exc:
                    logging.warning("RL model unavailable, skipping rl_policy_fees: %s", exc)
            return None
        if name == "pattern_trading":
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

    def _get_strategy(self, symbol: str, name: str):
        if symbol not in self._strategy_by_symbol:
            self._strategy_by_symbol[symbol] = {}
        if name not in self._strategy_by_symbol[symbol]:
            strategy = self._build_strategy(name, self._strategy_params)
            if strategy is None:
                return None
            self._strategy_by_symbol[symbol][name] = strategy
        return self._strategy_by_symbol[symbol][name]

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
        if self._has_pending_order(symbol):
            SKIPPED_ORDERS.labels(symbol=symbol, side="hold", reason="open_order").inc()
            logging.info("Skipping %s: open orders pending", symbol)
            return None
        signals = []
        for name in self._strategy_names:
            strategy_symbols = market_state.get("strategy_symbols", {})
            if isinstance(strategy_symbols, dict):
                allowed = strategy_symbols.get(name)
                if isinstance(allowed, list) and allowed and symbol not in allowed:
                    continue
            strategy = self._get_strategy(symbol, name)
            if not strategy:
                continue
            try:
                signal = strategy.generate_signal(market_state)
            except Exception as exc:
                logging.warning("Strategy %s failed for %s: %s", name, symbol, exc)
                continue
            signal["name"] = name
            signals.append(signal)
        self._update_orchestrator(symbol, market_state)
        if isinstance(self._orchestrator, MLStrategyOrchestrator):
            names, weights = self._orchestrator.select(symbol, self._strategy_names, market_state, signals)
        else:
            names, weights = self._orchestrator.select(self._strategy_names, market_state)
        for name in self._strategy_names:
            ORCHESTRATOR_STRATEGY_ACTIVE.labels(symbol=symbol, strategy=name).set(1 if name in names else 0)
        self._record_orchestrator(symbol, signals, market_state)
        filtered_signals = [signal for signal in signals if signal.get("name") in names]
        action, reduce_pct = self._combine_signals(filtered_signals, weights, order=names)
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
            qty, skip_reason = self._size_order(action, last_price, portfolio, symbol, reduce_pct=reduce_pct)
        else:
            qty, skip_reason = self._size_order(action, last_price, portfolio, symbol)
        if qty <= 0:
            if skip_reason:
                SKIPPED_ORDERS.labels(symbol=symbol, side=action, reason=skip_reason).inc()
                logging.info("Skipping %s for %s: %s", action, symbol, skip_reason)
            return None
        market_state["qty"] = qty

        now = datetime.utcnow()
        if self._is_cooldown_active(market_state, now):
            SKIPPED_ORDERS.labels(symbol=symbol, side=action, reason="cooldown").inc()
            logging.info("Skipping %s for %s: cooldown", action, symbol)
            return None
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
            self._last_trade_at = now
        elif action in ("buy", "sell"):
            SKIPPED_ORDERS.labels(symbol=symbol, side=action, reason="order_failed").inc()
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
        cash = max(0.0, cash - self._reserved_cash(symbol, last_price))

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
            return 0, "no_position"

        return 0, "unsupported"

    def _is_cooldown_active(self, market_state: dict, now: datetime) -> bool:
        cfg = self.cfg.get("risk", {})
        cooldown = int(cfg.get("cooldown_seconds", 0))
        if cooldown <= 0 or not self._last_trade_at:
            return False
        elapsed = (now - self._last_trade_at).total_seconds()
        return elapsed < cooldown

    def _resolve_strategy_names(self) -> list[str]:
        cfg = self.cfg.get("strategy", {})
        names = cfg.get("names")
        if isinstance(names, list) and names:
            return [str(name) for name in names]
        name = cfg.get("name", "intraday_momentum")
        return [str(name)]

    def _resolve_broker_name(self) -> str:
        brokers_cfg = self.cfg.get("brokers", {})
        if brokers_cfg.get("ibkr", {}).get("enabled", False):
            return "ibkr"
        return "alpaca"

    def _combine_signals(
        self,
        signals: list[dict],
        weights: dict[str, float] | None = None,
        order: list[str] | None = None,
    ) -> tuple[str, float]:
        if not signals:
            return "hold", 1.0
        for signal in signals:
            if signal.get("action") == "exit":
                return "exit", 1.0
        mode = self._combine_mode
        if mode == "priority":
            order = order or self._strategy_names
            for name in order:
                for signal in signals:
                    if signal.get("name") == name:
                        action = signal.get("action", "hold")
                        reduce_pct = float(signal.get("reduce_pct", 1.0))
                        return action, reduce_pct
            return "hold", 1.0
        weights = weights or {}
        buy_score = 0.0
        sell_score = 0.0
        sells = []
        for signal in signals:
            action = signal.get("action")
            name = signal.get("name")
            weight = float(weights.get(name, 1.0))
            if action == "buy":
                buy_score += weight
            elif action == "sell":
                sell_score += weight
                sells.append(signal)
        if buy_score == sell_score:
            return "hold", 1.0
        if buy_score > sell_score:
            return "buy", 1.0
        reduce_pct = max(float(s.get("reduce_pct", 1.0)) for s in sells) if sells else 1.0
        return "sell", reduce_pct

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
        self._symbols = symbol if isinstance(symbol, list) else [symbol]
        while True:
            if should_restart(self._started_at):
                logging.info("Restart requested; exiting trading loop.")
                raise SystemExit(0)
            portfolio = self._get_portfolio_snapshot()
            symbols = self._resolve_active_symbols()
            for name in self._strategy_names:
                STRATEGY_ACTIVE.labels(strategy=name).set(1)
            BROKER_ACTIVE.labels(broker=self._broker_name).set(1)
            self._update_account_metrics()
            self._refresh_news_cache(symbols)
            self._refresh_dynamic_symbols(portfolio)
            symbols = self._resolve_active_symbols()
            symbols = self._merge_symbols_with_positions(symbols, portfolio)
            for sym in symbols:
                SYMBOL_ACTIVE.labels(symbol=sym).set(1)
            self._refresh_open_orders_cache(symbols)
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
                market_state["strategy_symbols"] = self._symbols_by_strategy
                self.run_once(sym, market_state)
            time.sleep(interval_seconds)

    def _merge_symbols_with_positions(self, symbols: list[str], portfolio: dict) -> list[str]:
        positions = portfolio.get("positions", {})
        if not positions:
            return symbols
        merged = set(symbols)
        for symbol, position in positions.items():
            qty = float(position.get("qty", 0.0) or 0.0)
            if qty != 0:
                merged.add(symbol)
        return list(merged)

    def _update_orchestrator(self, symbol: str, market_state: dict) -> None:
        if isinstance(self._orchestrator, MLStrategyOrchestrator):
            self._orchestrator.update(symbol, market_state)
            return
        state = self._orchestrator_state.get(symbol)
        if not state:
            return
        last_price = state.get("last_price")
        if last_price is None:
            return
        current_price = market_state.get("last_price")
        if current_price is None:
            prices = market_state.get("prices", [])
            current_price = prices[-1] if prices else None
        if current_price is None:
            return
        decisions = state.get("decisions", {})
        if isinstance(decisions, dict):
            self._orchestrator.update_biases(decisions, float(last_price), float(current_price))
        self._orchestrator_state.pop(symbol, None)

    def _record_orchestrator(self, symbol: str, signals: list[dict], market_state: dict) -> None:
        if isinstance(self._orchestrator, MLStrategyOrchestrator):
            self._orchestrator.record(symbol, signals, market_state)
            return
        decisions = {s.get("name"): s.get("action") for s in signals if s.get("name")}
        if not decisions:
            return
        last_price = market_state.get("last_price")
        if last_price is None:
            prices = market_state.get("prices", [])
            last_price = prices[-1] if prices else None
        if last_price is None:
            return
        self._orchestrator_state[symbol] = {"decisions": decisions, "last_price": last_price}

    def _refresh_news_cache(self, symbols: list[str], now: datetime | None = None) -> None:
        news_cfg = self.cfg.get("news", {})
        if not news_cfg.get("enabled", False):
            self._news_cache = {}
            return
        now = now or datetime.utcnow()
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
            timeout_seconds=int(news_cfg.get("timeout_seconds", 10)),
            retries=int(news_cfg.get("retries", 2)),
        )
        self._news_cache_at = now

    def _refresh_dynamic_symbols(self, portfolio: dict, now: datetime | None = None) -> None:
        dyn_cfg = self.cfg.get("data", {}).get("dynamic_symbols", {})
        if not dyn_cfg.get("enabled", False):
            return
        now = now or datetime.utcnow()
        refresh_minutes = int(dyn_cfg.get("refresh_minutes", 15))
        if self._dynamic_symbols_at and (now - self._dynamic_symbols_at).total_seconds() < refresh_minutes * 60:
            return

        provider = dyn_cfg.get("provider", "alpaca")
        if provider != "alpaca":
            return
        alpaca_cfg = self.cfg.get("brokers", {}).get("alpaca", {})
        api_key = alpaca_cfg.get("api_key", "")
        api_secret = alpaca_cfg.get("api_secret", "")

        universe_cfg = dyn_cfg.get("universe", self._symbols)
        max_universe = int(dyn_cfg.get("max_universe", 500))
        universe = load_universe(api_key, api_secret, universe_cfg, max_universe=max_universe)
        if not universe:
            return

        self._symbols_by_strategy = {}
        filters_cfg_default = dyn_cfg.get("filters", {})
        global_candidates = self._scan_with_filters(
            portfolio,
            filters_cfg_default,
            dyn_cfg,
            api_key,
            api_secret,
            universe,
        )
        if global_candidates:
            self._symbols_by_strategy["__global__"] = global_candidates
        for name in self._strategy_names:
            if name == "pattern_trading":
                filters_cfg = self.cfg.get("pattern_trading", {}).get("selection", {})
            else:
                filters_cfg = filters_cfg_default
            candidates = self._scan_with_filters(
                portfolio,
                filters_cfg,
                dyn_cfg,
                api_key,
                api_secret,
                universe,
            )
            if candidates:
                self._symbols_by_strategy[name] = candidates
        if self._symbols_by_strategy:
            self._symbols = self._symbols_by_strategy.get("__global__", self._symbols)
        self._dynamic_symbols_at = now
        return

    def _apply_cash_cap(self, price_min: float, price_max: float, portfolio: dict, dyn_cfg: dict) -> float:
        if not dyn_cfg.get("cash_aware", True):
            return price_max
        try:
            cash = float(portfolio.get("cash", 0.0) or 0.0)
            equity = float(portfolio.get("equity", 0.0) or 0.0)
        except (TypeError, ValueError):
            return price_max
        if cash <= 0 or equity <= 0:
            return price_max
        cash_mode = str(dyn_cfg.get("cash_cap_mode", "cash")).lower()
        cash_max_pct = float(dyn_cfg.get("cash_max_pct", 100.0))
        cash_cap = cash * max(cash_max_pct, 0.0) / 100.0
        if cash_mode == "risk":
            max_pos_pct = float(self.cfg.get("risk", {}).get("max_position_size_pct", 0.0))
            target_value = equity * (max_pos_pct / 100.0)
            cash_cap = min(cash_cap, target_value)
        buffer_pct = float(dyn_cfg.get("cash_buffer_pct", 95.0))
        cap = cash_cap * max(buffer_pct, 0.0) / 100.0
        if cap <= 0:
            return price_max
        capped = min(price_max, cap)
        if capped < price_min:
            logging.info("Dynamic symbols cash cap %.2f below price_min %.2f; keeping price_max %.2f", cap, price_min, price_max)
            return price_max
        return capped

    def _resolve_active_symbols(self) -> list[str]:
        if not self._symbols_by_strategy:
            return self._symbols
        merged = set()
        for symbols in self._symbols_by_strategy.values():
            merged.update(symbols)
        return list(merged) if merged else self._symbols

    def _scan_with_filters(
        self,
        portfolio: dict,
        filters_cfg: dict,
        dyn_cfg: dict,
        api_key: str,
        api_secret: str,
        universe: list[str],
    ) -> list[str]:
        price_min = float(filters_cfg.get("price_min", 1.0))
        price_max = float(filters_cfg.get("price_max", 20.0))
        price_max = self._apply_cash_cap(price_min, price_max, portfolio, dyn_cfg)
        filters = ScanFilters(
            price_min=price_min,
            price_max=price_max,
            relative_volume_min=float(filters_cfg.get("relative_volume_min", 2.0)),
            premarket_gain_min_pct=float(filters_cfg.get("premarket_gain_min_pct", 5.0)),
            min_shares_traded=float(filters_cfg.get("min_shares_traded", 1_000_000)),
            max_spread_pct=float(filters_cfg.get("max_spread_pct", 1.0)),
            require_catalyst=bool(filters_cfg.get("require_catalyst", False)),
            strict_spread=bool(filters_cfg.get("strict_spread", False)),
        )
        max_symbols = int(dyn_cfg.get("max_symbols", 50))
        feed = dyn_cfg.get("feed", "iex")
        timeout_seconds = int(dyn_cfg.get("timeout_seconds", 10))
        retries = int(dyn_cfg.get("retries", 2))
        catalyst_map = {}
        if filters.require_catalyst:
            catalyst_map = fetch_catalyst_symbols(
                symbols=universe,
                provider=self.cfg.get("news", {}).get("provider", "alpaca"),
                base_url=self.cfg.get("news", {}).get("base_url", "https://data.alpaca.markets"),
                api_key=self.cfg.get("news", {}).get("api_key", ""),
                api_secret=self.cfg.get("news", {}).get("api_secret", ""),
                lookback_hours=int(self.cfg.get("news", {}).get("lookback_hours", 12)),
                keywords=self.cfg.get("news", {}).get("keywords", []),
                timeout_seconds=int(self.cfg.get("news", {}).get("timeout_seconds", 10)),
                retries=int(self.cfg.get("news", {}).get("retries", 2)),
            )
        candidates = scan_symbols(
            universe,
            api_key=api_key,
            api_secret=api_secret,
            feed=feed,
            filters=filters,
            catalyst_map=catalyst_map,
            max_symbols=max_symbols,
            timeout_seconds=timeout_seconds,
            retries=retries,
        )
        if candidates:
            return candidates
        fallback_cfg = dyn_cfg.get("fallback", {})
        if not fallback_cfg.get("enabled", False):
            return []
        logging.info("Dynamic symbols fallback enabled; relaxing filters.")
        fallback_filters = ScanFilters(
            price_min=float(fallback_cfg.get("price_min", price_min)),
            price_max=float(fallback_cfg.get("price_max", price_max)),
            relative_volume_min=float(fallback_cfg.get("relative_volume_min", 0.5)),
            premarket_gain_min_pct=float(fallback_cfg.get("premarket_gain_min_pct", 0.0)),
            min_shares_traded=float(fallback_cfg.get("min_shares_traded", 100_000)),
            max_spread_pct=float(fallback_cfg.get("max_spread_pct", 2.0)),
            require_catalyst=bool(fallback_cfg.get("require_catalyst", False)),
            strict_spread=bool(fallback_cfg.get("strict_spread", False)),
        )
        fallback_catalysts = self._news_cache if fallback_filters.require_catalyst else {}
        candidates = scan_symbols(
            universe,
            api_key=api_key,
            api_secret=api_secret,
            feed=feed,
            filters=fallback_filters,
            catalyst_map=fallback_catalysts,
            max_symbols=max_symbols,
            timeout_seconds=timeout_seconds,
            retries=retries,
        )
        if candidates:
            logging.info("Dynamic symbols fallback found %d candidates.", len(candidates))
        return candidates

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
        market_state["open_orders"] = self._open_orders_cache

    def _get_portfolio_snapshot(self) -> dict:
        account = self.broker.get_account()
        self._account_snapshot = account if isinstance(account, dict) else {}
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

    def _refresh_open_orders_cache(self, symbols: list[str]) -> None:
        exec_cfg = self.cfg.get("execution", {}).get("open_orders", {})
        if not exec_cfg.get("enabled", True):
            self._open_orders_cache = []
            self._open_orders_at = None
            self._reset_open_orders_metrics(symbols)
            return
        now = datetime.utcnow()
        interval_seconds = int(exec_cfg.get("interval_seconds", 30))
        if self._open_orders_at and (now - self._open_orders_at).total_seconds() < interval_seconds:
            return
        try:
            self._open_orders_cache = self.broker.get_open_orders()
        except Exception as exc:
            logging.warning("Open orders snapshot failed: %s", exc)
            self._open_orders_cache = []
        self._open_orders_at = now
        self._update_open_orders_metrics(symbols)

    def _update_open_orders_metrics(self, symbols: list[str]) -> None:
        self._reset_open_orders_metrics(symbols)
        counts: dict[tuple[str, str], int] = {}
        for order in self._open_orders_cache:
            symbol = order.get("symbol")
            side = (order.get("side") or "").lower()
            if not symbol or side not in {"buy", "sell"}:
                continue
            key = (symbol, side)
            counts[key] = counts.get(key, 0) + 1
        for (symbol, side), count in counts.items():
            OPEN_ORDERS.labels(symbol=symbol, side=side).set(count)

    def _reset_open_orders_metrics(self, symbols: list[str]) -> None:
        for symbol in symbols:
            for side in ("buy", "sell"):
                OPEN_ORDERS.labels(symbol=symbol, side=side).set(0)

    def _has_pending_order(self, symbol: str) -> bool:
        exec_cfg = self.cfg.get("execution", {}).get("open_orders", {})
        if not exec_cfg.get("skip_if_pending", True):
            return False
        for order in self._open_orders_cache:
            if order.get("symbol") == symbol:
                return True
        return False

    def _reserved_cash(self, symbol: str, last_price: float) -> float:
        reserved = 0.0
        for order in self._open_orders_cache:
            if order.get("symbol") != symbol:
                continue
            side = (order.get("side") or "").lower()
            if side != "buy":
                continue
            qty = float(order.get("qty") or 0.0)
            limit_price = order.get("limit_price")
            price = float(limit_price) if limit_price else float(last_price)
            reserved += qty * price
        return reserved
