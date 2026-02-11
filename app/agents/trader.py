# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import copy
import json
import hashlib
import logging
import os
import time
import threading

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait

from app.utils.account import extract_equity_cash
from app.execution.executor import ExecutionEngine
from app.execution.order_queue import OrderQueue
from app.execution.algos import pov_slices, twap_slices, vwap_slices
from app.execution.impact import estimate_market_impact
from app.execution import routing as routing_utils
from app.monitoring.metrics import (
    TRADES,
    SKIPPED_ORDERS,
    POSITION_QTY,
    POSITION_VALUE,
    POSITION_QTY_BY_BROKER,
    POSITION_VALUE_BY_BROKER,
    SIGNAL_RETURN_30M,
    SIGNAL_RETURN_60M,
    SIGNAL_EARLY_VOL,
    SIGNAL_RUNUP,
    SIGNAL_DRAWDOWN,
    SIGNAL_ABS_MOVE,
    SIGNAL_RUNUP_ABS,
    SIGNAL_DRAWDOWN_ABS,
    STRATEGY_ACTIVE,
    ORCHESTRATOR_STRATEGY_ACTIVE,
    ORCHESTRATOR_STRATEGY_SELECTED,
    BROKER_ACTIVE,
    BROKER_MARKET_OPEN,
    DECISION_LATENCY,
    ORDER_LATENCY,
    SKIPPED_ORDERS_BY_BROKER,
    TRADES_BY_BROKER,
    PORTFOLIO_SCALE_FALLBACK,
    TAKE_PROFIT_EXITS,
)
from app.monitoring.audit import AuditLogger, ComplianceLogger
from app.risk.manager import RiskManager
from app.risk.haircut import apply_haircuts
from app.strategies.intraday_momentum import IntradayMomentumStrategy
from app.strategies.trend_following import TrendFollowingStrategy
from app.strategies.factor_model import FactorModelStrategy
from app.strategies.stat_arb_pairs import StatArbPairsStrategy
from app.strategies.market_maker import MarketMakerStrategy
from app.strategies.pattern_trading import PatternTradingStrategy
from app.data.news import fetch_catalyst_symbols_for_config
from app.data.market_cache import build_market_cache, build_market_cache_config
from app.learning.drift import DriftMonitor
from app.learning.registry import load_active_model, load_latest_feature_stats
from app.learning.live_rewards import LiveRewardTracker
from app.brokers.config_utils import merge_cfg
from app.utils.checkpoint import load_checkpoint, maybe_save_checkpoint
from app.utils.structured_log import StructuredLogger
from app.utils.volatility import realized_volatility_pct

_slog = StructuredLogger("trading_agent")
from app.utils.ops_state import load_ops_state, ops_state_is_running, ops_state_is_sleeping
from app.utils.gpu_state import is_gpu_disabled, disable_gpu_until_restart # Import GPU state utilities
import tensorflow as tf # Import tensorflow for GPU error handling
try:
    import torch
    _TORCH_OOM = (torch.cuda.OutOfMemoryError,)
except Exception:
    _TORCH_OOM = ()
_OOM_ERRORS = (tf.errors.ResourceExhaustedError,) + _TORCH_OOM
from app.strategies.rl_policy import RLPolicyStrategy
from app.strategies.rl_policy_fees import FeeAwareRLPolicyStrategy
from app.utils.market import is_market_open, is_venue_extended
from app.utils.restart import should_restart
from app.agents.account_metrics import AccountMetricsUpdater
from app.agents.open_orders import OpenOrderManager
from app.agents.strategy_config import StrategyConfig
from app.execution.config import ExecutionConfig
from app.agents.orchestrator import RLStrategyOrchestrator
from app.agents.performance import PerformanceTracker
from app.agents.symbol_manager import SymbolManager
from app.learning.regime_hmm import RegimeHMM, RegimeState
from app.portfolio import PortfolioOptimizer, RiskModel


@dataclass
class BrokerState:
    risk: RiskManager
    equity_start: float | None = None
    equity_peak: float | None = None
    current_drawdown_pct: float = 0.0
    equity_history: list[float] = field(default_factory=list)
    var_cvar: dict[str, float] = field(default_factory=dict)
    day_start_date: date | None = None
    day_start_equity: float | None = None
    last_trade_at: datetime | None = None
    pending_entry_strategy: dict[str, dict[str, object]] = field(default_factory=dict)
    position_state: dict[str, dict[str, object]] = field(default_factory=dict)
    strategy_trades: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    symbol_trades: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    disabled_strategies: set[str] = field(default_factory=set)
    performance_last_report_at: datetime | None = None
    risk_outcomes: dict[str, dict] = field(default_factory=dict)
    last_prices: dict[str, float] = field(default_factory=dict)
    last_bar_ts: dict[str, datetime] = field(default_factory=dict)
    buying_power: float = 0.0


class TradingAgent:
    def __init__(self, broker, cfg: dict, account_cfgs: dict | None = None):
        self.cfg = cfg
        self.broker = broker
        self.learning_cfg = cfg.get("learning", {})
        # Enforce CPU if GPU is globally disabled due to previous errors
        if is_gpu_disabled():
            logging.warning("GPU globally disabled. Enforcing CPU for learning configurations.")
            self.learning_cfg["device"] = "cpu"
            # Update orchestrator device config
            if "orchestrator" in self.cfg and "rl" in self.cfg["orchestrator"]:
                self.cfg["orchestrator"]["rl"]["device"] = "cpu"
            # Update AI filter device config
            if "data" in self.cfg and \
               "dynamic_symbols" in self.cfg["data"] and \
               "ai_filter" in self.cfg["data"]["dynamic_symbols"]:
                self.cfg["data"]["dynamic_symbols"]["ai_filter"]["device"] = "cpu"
            # Update backtest GPU usage
            if "backtest" in self.cfg:
                self.cfg["backtest"]["use_gpu"] = False
        self._account_cfgs = account_cfgs or {}
        self._account_snapshot: dict[str, object] = {}
        self._strategy_cfg = StrategyConfig.from_dict(cfg.get("strategy", {}))
        self._execution_cfg = ExecutionConfig.from_dict(cfg.get("execution", {}))
        params = self._strategy_cfg.params
        self._strategy_params = params
        self._strategy_by_symbol: dict[tuple[str, str], dict[str, object]] = {}
        self._guardrail_by_symbol: dict[tuple[str, str], object] = {}
        self.executor = ExecutionEngine(broker)
        self._routing_cfg = self._execution_cfg.routing
        self._broker_map = self._resolve_broker_map()
        self._broker_names = list(self._broker_map.keys())
        self._broker_name = self._resolve_default_broker_name()
        self._broker_name = routing_utils.normalize_broker_name(self._broker_name, self._broker_names)
        retry_cfg = self._execution_cfg.retry
        open_orders_cfg = self._execution_cfg.open_orders
        completion_grace = int(open_orders_cfg.get("missing_grace_seconds", 0))
        self._order_queues = {
            name: OrderQueue(item, name, retry_cfg, completion_grace_seconds=completion_grace)
            for name, item in self._broker_map.items()
        }
        self._order_queue = self._order_queues.get(self._broker_name)
        self._broker_states: dict[str, BrokerState] = {}
        for name in self._broker_map.keys():
            acct_cfg = self._account_cfgs.get(name, {})
            broker_risk_cfg = merge_cfg(cfg["risk"], acct_cfg.get("risk", {}))
            risk_tz_name = cfg.get("risk", {}).get("timezone", "US/Eastern")
            try:
                from zoneinfo import ZoneInfo
                risk_tz = ZoneInfo(risk_tz_name)
            except Exception:
                risk_tz = timezone.utc
            self._broker_states[name] = BrokerState(risk=RiskManager(broker_risk_cfg, tz=risk_tz))
        self._last_market_open = None
        self._started_at = datetime.now(timezone.utc)
        self._news_cache: dict[str, bool] = {}
        self._news_cache_at: datetime | None = None
        self._news_executor = ThreadPoolExecutor(max_workers=1)
        self._symbol_executor = ThreadPoolExecutor(max_workers=4)
        self._news_future = None
        self._news_future_lock = threading.Lock()
        self._news_inflight_at: datetime | None = None
        self._strategy_names = self._resolve_strategy_names()
        self._combine_mode = self._strategy_cfg.combine
        self._orchestrator = RLStrategyOrchestrator(cfg, account_cfgs=self._account_cfgs)
        # Regime detection
        regime_cfg = cfg.get("learning", {}).get("regime", {})
        self._regime_enabled = regime_cfg.get("enabled", False)
        self._regime_hmm: RegimeHMM | None = None
        self._regime_state: RegimeState | None = None
        self._regime_returns_buffer: list[float] = []
        self._regime_buffer_size = int(regime_cfg.get("lookback", 100))
        if self._regime_enabled:
            self._regime_hmm = RegimeHMM(
                n_regimes=int(regime_cfg.get("n_regimes", 3)),
                config=regime_cfg,
            )
            logging.info("RegimeHMM initialized: n_regimes=%d", self._regime_hmm.n_regimes)
        # Portfolio optimization
        portfolio_cfg = cfg.get("portfolio", {})
        self._portfolio_enabled = portfolio_cfg.get("enabled", False)
        self._portfolio_optimizer: PortfolioOptimizer | None = None
        self._risk_model: RiskModel | None = None
        if self._portfolio_enabled:
            self._portfolio_optimizer = PortfolioOptimizer(portfolio_cfg)
            self._risk_model = RiskModel(portfolio_cfg.get("risk_model", {}))
            logging.info("PortfolioOptimizer initialized")
        self._open_order_mgr = OpenOrderManager(cfg, self._broker_name)
        self._live_reward_tracker: LiveRewardTracker | None = None
        if self.cfg.get("learning", {}).get("live_rewards", {}).get("enabled", False):
            self._live_reward_tracker = LiveRewardTracker(self.cfg)
        self._orchestrator_state: dict[tuple[str, str], dict[str, object]] = {}
        self._position_symbols: set[str] = set()
        self._position_symbols_by_broker: dict[str, set[str]] = {}
        self._market_cache_cfg = build_market_cache_config(cfg.get("market_cache", {}))
        self._market_cache = build_market_cache(cfg.get("market_cache", {}))
        self._symbol_mgr = SymbolManager(cfg, self._broker_map, self._broker_name, self._open_order_mgr, self._market_cache_cfg, self._market_cache)
        self._broker_missing_at: datetime | None = None
        self._broker_missing_last_log = 0.0
        self._broker_missing_reason: str | None = None
        report_cfg = cfg.get("reports", {}).get("daily_top_movers", {}) or {}
        trace_cfg = report_cfg.get("decision_trace", {}) or {}
        self._decision_trace_enabled = bool(trace_cfg.get("enabled", True))
        self._decision_trace_dir = str(trace_cfg.get("output_dir", "/data/reports/decision_trace"))
        self._decision_trace_blocked = False
        monitoring_cfg = cfg.get("monitoring", {}) or {}
        audit_cfg = monitoring_cfg.get("audit", {}) or {}
        compliance_cfg = monitoring_cfg.get("compliance", {}) or {}
        self._audit_logger: AuditLogger | None = None
        self._audit_include_features = bool(audit_cfg.get("include_features", False))
        self._audit_include_market_state = bool(audit_cfg.get("include_market_state", False))
        if audit_cfg.get("enabled", False):
            audit_signing = audit_cfg.get("signing", {}) or {}
            audit_secret = None
            secret_env = audit_signing.get("secret_env")
            if secret_env:
                audit_secret = os.getenv(str(secret_env))
            self._audit_logger = AuditLogger(
                str(audit_cfg.get("output_dir", "/data/reports/audit")),
                retention_days=int(audit_cfg.get("retention_days", 0)),
                enforce_reason_codes=bool(audit_cfg.get("enforce_reason_codes", False)),
                reason_codes_path=audit_cfg.get("reason_codes_path"),
                signing_secret=audit_secret,
                signing_enabled=bool(audit_signing.get("enabled", False)),
            )
        self._compliance_logger: ComplianceLogger | None = None
        self._compliance_include_features = bool(compliance_cfg.get("include_features", False))
        self._compliance_include_market_state = bool(compliance_cfg.get("include_market_state", False))
        if compliance_cfg.get("enabled", False):
            compliance_signing = compliance_cfg.get("signing", {}) or {}
            compliance_secret = None
            secret_env = compliance_signing.get("secret_env")
            if secret_env:
                compliance_secret = os.getenv(str(secret_env))
            self._compliance_logger = ComplianceLogger(
                str(compliance_cfg.get("output_dir", "/data/reports/compliance")),
                formats=compliance_cfg.get("formats", ["jsonl"]),
                retention_days=int(compliance_cfg.get("retention_days", 0)),
                enforce_reason_codes=bool(compliance_cfg.get("enforce_reason_codes", False)),
                reason_codes_path=compliance_cfg.get("reason_codes_path"),
                signing_secret=compliance_secret,
                signing_enabled=bool(compliance_signing.get("enabled", False)),
            )
        self._include_feature_snapshots = self._audit_include_features or self._compliance_include_features
        self._drift_monitor: DriftMonitor | None = None
        self._drift_auto_rollback = False
        self._drift_rollback_done = False
        self._active_model_ref: str | None = None
        self._active_model_checked_at: datetime | None = None
        self._active_model_snapshot: dict | None = None
        self._init_drift_monitor()
        self._account_metrics = AccountMetricsUpdater(cfg)
        self._active_kill_switch_profile: str | None = None
        self._checkpoint_at: datetime | None = None
        self._kill_switch_liquidated = False
        self._kill_switch_warned = False
        self._perf_tracker = PerformanceTracker(cfg, self._strategy_names)
        self._lock = threading.RLock()
        if isinstance(self._orchestrator, RLStrategyOrchestrator):
            self._orchestrator.bootstrap(
                self._strategy_names,
                self._build_strategy,
                self._strategy_params,
                cfg.get("data", {}),
            )
        self._load_checkpoint()

    def _record_skip(self, symbol: str, action: str, reason: str, broker_name: str | None = None) -> None:
        SKIPPED_ORDERS.labels(symbol=symbol, side=action, reason=reason).inc()
        if broker_name:
            SKIPPED_ORDERS_BY_BROKER.labels(broker=broker_name, symbol=symbol, side=action, reason=reason).inc()
        _slog.event("info", "risk_blocked", symbol=symbol, action=action, reason=reason, broker=broker_name)

    def _broker_state(self, broker_name: str | None) -> BrokerState:
        if not broker_name:
            return self._broker_states[self._broker_name]
        if broker_name in self._broker_states:
            return self._broker_states[broker_name]
        normalized = self._normalize_broker_name(str(broker_name))
        return self._broker_states.get(normalized, self._broker_states[self._broker_name])

    def _strategy_disabled_globally(self, name: str) -> bool:
        if not self._broker_states:
            return False
        return all(name in state.disabled_strategies for state in self._broker_states.values())

    def _record_risk_outcome(
        self,
        symbol: str,
        action: str,
        allowed: bool,
        reason: str,
        broker_name: str | None = None,
    ) -> None:
        broker_state = self._broker_state(broker_name)
        broker_state.risk_outcomes[symbol] = {
            "action": action,
            "allowed": allowed,
            "reason": reason,
            "at": datetime.now(timezone.utc).isoformat(),
        }

    def _risk_disabled(self) -> bool:
        return not bool(self.cfg.get("risk", {}).get("enabled", True))

    def _build_rl_strategy(self, strategy_cls, name: str, extra_kwargs: dict | None = None):
        """Build an RL strategy with device-fallback loop and OOM handling."""
        if not self.learning_cfg.get("enabled"):
            logging.debug("RL %s not enabled or failed to build.", name)
            return None
        model_path = self._select_model_path()
        window_size = int(self.learning_cfg.get("window_size", 50))
        device = self.learning_cfg.get("device", "auto")
        feature_config = self.learning_cfg.get("features", {})
        try_devices = [device]
        if device != "cpu":
            try_devices.append("cpu")
        base_kwargs = dict(
            window_size=window_size,
            feature_config=feature_config,
            drift_monitor=self._drift_monitor,
            include_features=self._include_feature_snapshots,
            risk_cfg=self.cfg.get("risk", {}),
        )
        if extra_kwargs:
            base_kwargs.update(extra_kwargs)
        for current_device in try_devices:
            try:
                return strategy_cls(model_path, device=current_device, **base_kwargs)
            except (FileNotFoundError, ValueError) as exc:
                logging.warning("RL model unavailable, skipping %s: %s", name, exc)
                break
            except _OOM_ERRORS as exc:
                if current_device != "cpu":
                    logging.warning("CUDA OOM during %s build: %s. Falling back to CPU.", name, exc)
                    disable_gpu_until_restart()
                    self.learning_cfg["device"] = "cpu"
                    continue
                else:
                    logging.error("RL %s failed on CPU after GPU error: %s", name, exc)
                    break
            except Exception as exc:
                logging.warning("Unknown error during %s build: %s", name, exc)
                break
        logging.debug("RL %s not enabled or failed to build.", name)
        return None

    def _build_strategy(self, name: str, params: dict):
        if name == "rl_policy":
            return self._build_rl_strategy(RLPolicyStrategy, "rl_policy")
        if name == "rl_policy_fees":
            return self._build_rl_strategy(
                FeeAwareRLPolicyStrategy, "rl_policy_fees",
                extra_kwargs={
                    "broker_fees": self.cfg.get("brokers", {}).get(self._broker_name, {}).get("fees", {}),
                    "fee_guard": self._strategy_cfg.fee_aware,
                },
            )
        if name == "pattern_trading":
            return PatternTradingStrategy(self.cfg.get("pattern_trading", {}))
        if name == "trend_following":
            return TrendFollowingStrategy(params)
        if name == "factor_model":
            return FactorModelStrategy(params)
        if name == "stat_arb_pairs":
            return StatArbPairsStrategy(params)
        if name == "market_maker":
            return MarketMakerStrategy(params)
        logging.warning("Unknown strategy '%s' requested, skipping.", name)
        return None

    def _select_model_path(self) -> str:
        registry_cfg = self.learning_cfg.get("registry", {}) or {}
        if registry_cfg.get("use_active", True):
            active_path = registry_cfg.get("active_path", "/app/models/model_active.json")
            active = load_active_model(active_path)
            if isinstance(active, dict):
                active_model = active.get("model_path")
                if active_model and Path(active_model).exists():
                    return str(active_model)
        model_path = self.learning_cfg.get("model_path", "/app/models/ppo_policy.zip")
        if not self.learning_cfg.get("use_best_model", True):
            return model_path
        best_path = self.learning_cfg.get("best_model_path", "/app/models/ppo_policy_best.zip")
        return best_path if Path(best_path).exists() else model_path

    def _maybe_reload_active_model(self) -> None:
        registry_cfg = self.learning_cfg.get("registry", {}) or {}
        if not registry_cfg.get("use_active", True):
            return
        refresh_minutes = int(registry_cfg.get("refresh_minutes", 5))
        now = datetime.now(timezone.utc)
        if self._active_model_checked_at and (now - self._active_model_checked_at).total_seconds() < refresh_minutes * 60:
            return
        active_path = registry_cfg.get("active_path", "/app/models/model_active.json")
        active = load_active_model(active_path)
        ref = _active_model_ref(active)
        self._active_model_checked_at = now
        if active:
            self._active_model_snapshot = active
        if ref and ref != self._active_model_ref:
            self._active_model_ref = ref
            self._reload_rl_strategies()
            logging.info("Active model updated; reloading RL strategies.")

    def _current_model_snapshot(self) -> dict | None:
        registry_cfg = self.learning_cfg.get("registry", {}) or {}
        if not registry_cfg.get("use_active", True):
            return None
        return self._active_model_snapshot

    def _init_drift_monitor(self) -> None:
        drift_cfg = self.learning_cfg.get("drift", {}) or {}
        if not drift_cfg.get("enabled", False):
            return
        registry_cfg = self.learning_cfg.get("registry", {}) or {}
        registry_path = str(registry_cfg.get("path", "/app/models/model_registry.json"))
        baseline = load_latest_feature_stats(registry_path)
        if not baseline:
            logging.warning("Drift monitor enabled but no feature baseline found at %s", registry_path)
        self._drift_monitor = DriftMonitor(
            baseline_stats=baseline,
            window=int(drift_cfg.get("window", 120)),
            feature_zscore_threshold=float(drift_cfg.get("feature_zscore_threshold", 3.0)),
            max_drift_feature_pct=float(drift_cfg.get("max_drift_feature_pct", 0.3)),
            pnl_window=int(drift_cfg.get("pnl_window", 30)),
            max_pnl_drop_pct=float(drift_cfg.get("max_pnl_drop_pct", 2.0)),
        )
        self._drift_auto_rollback = bool(drift_cfg.get("auto_rollback", True))

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

    def _strategy_key(self, broker_name: str, symbol: str) -> tuple[str, str]:
        return (self._normalize_broker_name(broker_name), symbol)

    def _get_strategy(self, broker_name: str, symbol: str, name: str):
        key = self._strategy_key(broker_name, symbol)
        if key not in self._strategy_by_symbol:
            self._strategy_by_symbol[key] = {}
        if name not in self._strategy_by_symbol[key]:
            strategy = self._build_strategy(name, self._strategy_params)
            if strategy is None:
                return None
            self._strategy_by_symbol[key][name] = strategy
        return self._strategy_by_symbol[key][name]

    def _get_guardrail(self, broker_name: str, symbol: str):
        key = self._strategy_key(broker_name, symbol)
        if key not in self._guardrail_by_symbol:
            self._guardrail_by_symbol[key] = self._build_guardrail(self._strategy_params)
        return self._guardrail_by_symbol[key]

    def _apply_guardrail(self, action: str, guard_action: str, mode: str) -> str:
        if action not in ("buy", "sell"):
            return action
        if mode == "confirm":
            return action if guard_action == action else "hold"
        if mode == "veto":
            if guard_action in ("hold", "exit") or guard_action != action:
                return "hold"
        return action

    def _check_position_exit(
        self, symbol: str, last_price: float, broker_state
    ) -> tuple[bool, str]:
        """Check if a held long position should be exited based on risk rules.

        Returns (should_exit, reason) where reason is one of:
            hard_stop, trailing_stop, time_exit, take_profit,
            partial_take_profit, or empty string.
        """
        pos = broker_state.position_state.get(symbol)
        if not pos or float(pos.get("qty", 0)) <= 0:
            return False, ""
        avg_entry = pos.get("avg_entry")
        if not avg_entry or float(avg_entry) <= 0:
            return False, ""
        avg_entry = float(avg_entry)
        risk_cfg = broker_state.risk.cfg

        # Hard stop: price dropped X% from entry
        hard_stop = float(risk_cfg.get("hard_stop_pct", 0) or 0)
        if hard_stop > 0 and last_price <= avg_entry * (1 - hard_stop / 100.0):
            return True, "hard_stop"

        # Trailing stop: price dropped X% from peak since entry
        trailing_stop = float(risk_cfg.get("trailing_stop_pct", 0) or 0)
        peak = float(pos.get("peak_price") or avg_entry)
        if trailing_stop > 0 and peak > avg_entry:
            if last_price <= peak * (1 - trailing_stop / 100.0):
                return True, "trailing_stop"

        # Take-profit: full exit when price >= entry * (1 + take_profit_pct/100)
        take_profit_pct = float(risk_cfg.get("take_profit_pct", 0) or 0)
        if take_profit_pct > 0 and last_price >= avg_entry * (1 + take_profit_pct / 100.0):
            TAKE_PROFIT_EXITS.labels(symbol=symbol, reason="take_profit").inc()
            return True, "take_profit"

        # Partial take-profit: sell a fraction when first target hit, let rest run
        partial_tp_pct = float(risk_cfg.get("partial_take_profit_pct", 0) or 0)
        if partial_tp_pct > 0 and not pos.get("took_partial"):
            if last_price >= avg_entry * (1 + partial_tp_pct / 100.0):
                pos["took_partial"] = True
                TAKE_PROFIT_EXITS.labels(symbol=symbol, reason="partial_take_profit").inc()
                return True, "partial_take_profit"

        # Time-based exit: held longer than position_horizon_minutes
        horizon = float(self._strategy_params.get("position_horizon_minutes", 0) or 0)
        opened_at = pos.get("opened_at")
        if horizon > 0 and opened_at is not None:
            if (datetime.now(timezone.utc) - opened_at).total_seconds() > horizon * 60:
                return True, "time_exit"

        return False, ""

    def _select_order_meta(self, signals: list[dict], order: list[str]) -> dict:
        for name in order:
            for signal in signals:
                if signal.get("name") != name:
                    continue
                return {
                    "order_type": signal.get("order_type"),
                    "limit_price": signal.get("limit_price"),
                    "algo": signal.get("algo"),
                }
        return {}

    def _plan_execution(self, action: str, qty: int, last_price: float, market_state: dict, algo_name: str | None):
        algo_cfg = self._execution_cfg.algos
        if not algo_cfg.get("enabled", False):
            return []
        if action not in ("buy", "sell"):
            return []
        notional = qty * last_price
        min_notional = float(algo_cfg.get("min_notional", 0.0))
        if notional < min_notional:
            return []
        name = algo_name or str(algo_cfg.get("default", "twap"))
        adaptive_cfg = algo_cfg.get("adaptive", {}) or {}
        if algo_name is None and adaptive_cfg.get("enabled", False):
            impact_cfg = self._execution_cfg.impact or {}
            impact = estimate_market_impact(notional, last_price, market_state, impact_cfg)
            thresholds = adaptive_cfg.get("impact_bps_thresholds", {}) or {}
            vwap_threshold = float(thresholds.get("vwap", 4.0))
            pov_threshold = float(thresholds.get("pov", 8.0))
            if impact.impact_bps >= pov_threshold:
                name = "pov"
            elif impact.impact_bps >= vwap_threshold:
                name = "vwap"
            else:
                name = "twap"
        if name == "twap":
            duration = int(algo_cfg.get("twap", {}).get("duration_seconds", 120))
            slices = int(algo_cfg.get("twap", {}).get("slices", 4))
            return twap_slices(qty, duration, slices)
        if name == "vwap":
            duration = int(algo_cfg.get("vwap", {}).get("duration_seconds", 120))
            profile = algo_cfg.get("vwap", {}).get("profile", [1, 1, 1, 1])
            return vwap_slices(qty, profile, duration)
        if name == "pov":
            max_participation = float(algo_cfg.get("pov", {}).get("max_participation", 0.1))
            est_volume = float(market_state.get("session_volume", 0.0) or 0.0)
            if est_volume <= 0:
                est_volume = float(algo_cfg.get("pov", {}).get("estimated_volume", 0.0))
            return pov_slices(qty, max_participation, est_volume)
        return []

    def _init_decision_trace(self, symbol: str, market_state: dict) -> dict:
        if not (self._decision_trace_enabled or self._audit_logger or self._compliance_logger):
            return {}
        trace = {
            "symbol": symbol,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        if market_state:
            trace["signal_inputs"] = self._signal_snapshot(market_state)
            if self._audit_include_market_state or self._compliance_include_market_state:
                trace["market_state"] = self._market_state_snapshot(market_state)
        venue = market_state.get("venue") or market_state.get("market_venue")
        if venue:
            trace["venue"] = venue
        return trace

    def _signal_summary(self, signal: dict, include_features: bool = False) -> dict:
        summary = {"name": signal.get("name"), "action": signal.get("action")}
        for key in ("score", "confidence", "strength", "reason", "weight", "signal_bias", "signal_bias_block", "value_estimate", "action_probs"):
            if key in signal:
                summary[key] = signal.get(key)
        if include_features and "features" in signal:
            summary["features"] = signal.get("features")
        return summary

    def _signal_snapshot(self, market_state: dict) -> dict:
        return {
            "30m_return_pct": market_state.get("signal_30m_return_pct"),
            "60m_return_pct": market_state.get("signal_60m_return_pct"),
            "early_volume_pct": market_state.get("signal_early_volume_pct"),
            "runup_pct": market_state.get("signal_runup_pct"),
            "drawdown_pct": market_state.get("signal_drawdown_pct"),
            "abs_move": market_state.get("signal_abs_move"),
            "runup_abs": market_state.get("signal_runup_abs"),
            "drawdown_abs": market_state.get("signal_drawdown_abs"),
        }

    def _market_state_snapshot(self, market_state: dict) -> dict:
        return {
            "last_price": market_state.get("last_price"),
            "prices": market_state.get("prices"),
            "volumes": market_state.get("volumes"),
            "spread_pct": market_state.get("spread_pct"),
            "session_volume": market_state.get("session_volume"),
            "signal_inputs": self._signal_snapshot(market_state),
        }

    def _signal_bias(self, market_state: dict) -> float | None:
        vals = []
        for key in (
            "signal_30m_return_pct",
            "signal_60m_return_pct",
            "signal_runup_pct",
            "signal_drawdown_pct",
        ):
            raw = market_state.get(key)
            if raw is None:
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if val == 0.0:
                continue
            vals.append(1.0 if val > 0 else -1.0)
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    def _detect_regime(self, market_state: dict) -> RegimeState | None:
        """Detect current market regime using HMM."""
        if self._regime_hmm is None:
            return None
        # Extract return from market state
        prices = market_state.get("prices", [])
        if len(prices) >= 2:
            ret = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] != 0 else 0.0
            self._regime_returns_buffer.append(ret)
            # Keep buffer size limited
            if len(self._regime_returns_buffer) > self._regime_buffer_size:
                self._regime_returns_buffer = self._regime_returns_buffer[-self._regime_buffer_size:]
        # Need minimum samples for regime detection
        if len(self._regime_returns_buffer) < 20:
            return None
        returns = np.array(self._regime_returns_buffer)
        try:
            self._regime_state = self._regime_hmm.get_state(returns)
            return self._regime_state
        except (ValueError, RuntimeError) as exc:
            logging.debug("Regime detection failed: %s", exc)
            return None

    def _update_signal_metrics(self, symbol: str, market_state: dict) -> None:
        self._update_signal_metrics_from_values(symbol, market_state)

    def _update_signal_metrics_from_values(self, symbol: str, values: dict) -> None:
        SIGNAL_RETURN_30M.labels(symbol=symbol).set(float(values.get("signal_30m_return_pct", 0.0) or 0.0))
        SIGNAL_RETURN_60M.labels(symbol=symbol).set(float(values.get("signal_60m_return_pct", 0.0) or 0.0))
        SIGNAL_EARLY_VOL.labels(symbol=symbol).set(float(values.get("signal_early_volume_pct", 0.0) or 0.0))
        SIGNAL_RUNUP.labels(symbol=symbol).set(float(values.get("signal_runup_pct", 0.0) or 0.0))
        SIGNAL_DRAWDOWN.labels(symbol=symbol).set(float(values.get("signal_drawdown_pct", 0.0) or 0.0))
        SIGNAL_ABS_MOVE.labels(symbol=symbol).set(float(values.get("signal_abs_move", 0.0) or 0.0))
        SIGNAL_RUNUP_ABS.labels(symbol=symbol).set(float(values.get("signal_runup_abs", 0.0) or 0.0))
        SIGNAL_DRAWDOWN_ABS.labels(symbol=symbol).set(float(values.get("signal_drawdown_abs", 0.0) or 0.0))

    def _portfolio_snapshot_for_trace(self, portfolio: dict, symbol: str) -> dict:
        positions = portfolio.get("positions", {}) if isinstance(portfolio, dict) else {}
        position = positions.get(symbol, {}) if isinstance(positions, dict) else {}
        return {
            "cash": portfolio.get("cash"),
            "equity": portfolio.get("equity"),
            "buying_power": portfolio.get("buying_power"),
            "position_qty": position.get("qty"),
        }

    @staticmethod
    def _haircut_snapshot(market_state: dict) -> dict:
        fields = (
            "stress_haircut_pct",
            "liquidity_haircut_pct",
            "max_participation",
            "session_volume",
            "spread_pct",
        )
        return {key: market_state.get(key) for key in fields if key in market_state}

    def _write_decision_trace(self, payload: dict) -> None:
        try:
            date_str = datetime.now(timezone.utc).date().isoformat()
            path = Path(self._decision_trace_dir) / f"{date_str}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        except OSError as exc:
            if not self._decision_trace_blocked:
                logging.warning("Decision trace write failed; disabling trace output: %s", exc)
            self._decision_trace_blocked = True
            self._decision_trace_enabled = False
        except Exception as exc:
            logging.warning("Decision trace write failed: %s", exc)

    def _emit_decision_trace(
        self,
        base: dict,
        decision: str,
        reason: str,
        stage: str,
        extra: dict | None = None,
    ) -> None:
        if not (self._decision_trace_enabled or self._audit_logger or self._compliance_logger):
            return
        payload = dict(base)
        payload.update(
            {
                "decision": decision,
                "reason": reason,
                "stage": stage,
            }
        )
        start = payload.pop("_decision_start", None)
        if start is not None:
            try:
                latency = time.perf_counter() - float(start)
            except (TypeError, ValueError):
                latency = None
            if latency is not None:
                payload["decision_latency_seconds"] = latency
        if extra:
            payload.update(extra)
        if self._decision_trace_enabled:
            self._write_decision_trace(payload)
        if self._audit_logger:
            self._audit_logger.write(payload)
        if self._compliance_logger:
            self._compliance_logger.write(payload)

    def run_once(self, symbol: str, market_state: dict):
        with self._lock:
            trace = self._init_decision_trace(symbol, market_state)
            if trace and "_decision_start" in market_state:
                trace["_decision_start"] = market_state.get("_decision_start")
            if trace:
                model_snapshot = self._current_model_snapshot()
                if model_snapshot:
                    trace["model_active"] = model_snapshot
            if self._kill_switch_liquidated:
                return None
            self._apply_kill_switch_profile(market_state)
            broker_hint = self._resolve_broker_for_symbol(symbol, self._strategy_names, None)
            broker_override = market_state.get("broker_override")
            if broker_override:
                broker_override = self._normalize_broker_name(str(broker_override))
            strategy_broker = broker_override or broker_hint
            market_state["risk_outcome"] = self._broker_state(strategy_broker).risk_outcomes.get(symbol, {})
            if trace:
                trace["broker_hint"] = broker_hint
            active_strategies = [
                name for name in self._strategy_names if not self._strategy_disabled_globally(name)
            ]
            if not active_strategies:
                logging.warning("No active strategies available; skipping %s", symbol)
                self._emit_decision_trace(trace, "skip", "no_active_strategies", "strategy")
                return None
            
            # Prepare strategies safely within lock
            strategies_to_run = []
            for name in active_strategies:
                strategy_symbols = market_state.get("strategy_symbols", {})
                if isinstance(strategy_symbols, dict):
                    allowed = strategy_symbols.get(name)
                    if isinstance(allowed, list) and allowed and symbol not in allowed:
                        continue
                strategy = self._get_strategy(strategy_broker, symbol, name)
                if not strategy:
                    continue
                strategies_to_run.append((name, strategy))

        # UNLOCKED: Run strategy inference (CPU heavy)
        signals = []
        for name, strategy in strategies_to_run:
            try:
                signal = strategy.generate_signal(market_state)
            except (ValueError, KeyError, RuntimeError) as exc:
                logging.warning("Strategy %s failed for %s: %s", name, symbol, exc)
                continue
            signal["name"] = name
            signals.append(signal)

        # UNLOCKED: Signal bias calculation (pure logic)
        bias = self._signal_bias(market_state)
        guard_cfg = self._strategy_cfg.signal_bias_guard or {}
        if bias is not None:
            for signal in signals:
                signal["signal_bias"] = bias
            if guard_cfg.get("enabled", True):
                threshold = float(guard_cfg.get("threshold", 0.2))
                for signal in signals:
                    action = str(signal.get("action", "hold")).lower()
                    if action == "buy" and bias < -threshold:
                        signal["action"] = "hold"
                        signal["signal_bias_block"] = "negative_bias"
                    elif action == "sell" and bias > threshold:
                        signal["action"] = "hold"
                        signal["signal_bias_block"] = "positive_bias"
        
        if trace:
            trace["signals"] = [
                self._signal_summary(sig, include_features=self._include_feature_snapshots) for sig in signals
            ]
            trace["orchestrator_mode"] = self.cfg.get("orchestrator", {}).get("mode", "direct")

        with self._lock:
            self._update_orchestrator(symbol, market_state, strategy_broker)

            # Regime detection under lock — mutates _regime_returns_buffer and _regime_state
            if self._regime_enabled and self._regime_hmm is not None:
                regime_state = self._detect_regime(market_state)
                if regime_state is not None:
                    market_state["regime"] = regime_state.regime
                    market_state["regime_name"] = regime_state.regime_name
                    market_state["regime_probability"] = regime_state.probability
                    market_state["regime_probs"] = regime_state.regime_probs
                    if trace:
                        trace["regime"] = regime_state.regime
                        trace["regime_name"] = regime_state.regime_name

        # UNLOCKED: Orchestrator inference (CPU heavy)
        if isinstance(self._orchestrator, RLStrategyOrchestrator):
            names, weights = self._orchestrator.select(
                symbol, active_strategies, market_state, signals, strategy_broker
            )
        else:
            names, weights = self._orchestrator.select(active_strategies, market_state)

        # LOCKED: Execution logic (Critical State Updates)
        with self._lock:
            filtered_signals = [signal for signal in signals if signal.get("name") in names]
            action, reduce_pct, action_strategy = self._combine_signals(filtered_signals, weights, order=names)
            order_meta = self._select_order_meta(filtered_signals, names)
            if broker_override:
                broker_name = broker_override
            else:
                broker_name = self._resolve_broker_for_symbol(symbol, names, filtered_signals, action_strategy)
            broker_state = self._broker_state(broker_name)
            risk_disabled = self._risk_disabled()
            if risk_disabled:
                market_state["risk_disabled"] = True
            if broker_name != strategy_broker:
                market_state["risk_outcome"] = broker_state.risk_outcomes.get(symbol, {})
            allowed_names = [name for name in names if name not in broker_state.disabled_strategies]
            if not allowed_names:
                logging.warning("No active strategies available for %s; skipping %s", broker_name, symbol)
                self._emit_decision_trace(trace, "skip", "no_active_strategies", "strategy")
                return None
            if allowed_names != list(names) or (action_strategy and action_strategy not in allowed_names):
                filtered_signals = [signal for signal in signals if signal.get("name") in allowed_names]
                filtered_weights = weights
                if isinstance(weights, dict):
                    filtered_weights = {name: weights.get(name, 1.0) for name in allowed_names}
                action, reduce_pct, action_strategy = self._combine_signals(
                    filtered_signals, filtered_weights, order=allowed_names
                )
                order_meta = self._select_order_meta(filtered_signals, allowed_names)
                if not broker_override:
                    broker_name = self._resolve_broker_for_symbol(
                        symbol, allowed_names, filtered_signals, action_strategy
                    )
                    broker_state = self._broker_state(broker_name)
                    if action_strategy and action_strategy in broker_state.disabled_strategies:
                        self._record_skip(symbol, "hold", "strategy_disabled", broker_name)
                        logging.info("Skipping %s: strategy disabled for broker %s", symbol, broker_name)
                        self._emit_decision_trace(trace, "skip", "strategy_disabled", "strategy")
                        return None
                names = allowed_names
                weights = filtered_weights
            if not risk_disabled and broker_state.risk.should_circuit_break(broker_state.current_drawdown_pct):
                self._record_skip(symbol, "hold", "circuit_breaker", broker_name)
                logging.warning("Skipping %s: circuit breaker drawdown hit", symbol)
                self._emit_decision_trace(trace, "skip", "circuit_breaker", "risk")
                return None
            if trace:
                trace["orchestrator_selected"] = list(names)
                trace["orchestrator_weights"] = list(weights) if isinstance(weights, (list, tuple)) else weights
                if isinstance(self._orchestrator, RLStrategyOrchestrator):
                    diag = getattr(self._orchestrator, "_last_diagnostics", {})
                    if diag:
                        trace["orchestrator_strategy_probs"] = diag.get("strategy_probs")
                        trace["orchestrator_epsilon_explore"] = diag.get("epsilon_explore", False)
            for name in self._strategy_names:
                ORCHESTRATOR_STRATEGY_ACTIVE.labels(symbol=symbol, strategy=name).set(1 if name in names else 0)
            for name in set(names):
                ORCHESTRATOR_STRATEGY_SELECTED.labels(strategy=name).inc()
            market_state["broker"] = broker_name
            self._record_orchestrator(symbol, signals, market_state, broker_name)
            if trace:
                trace["action"] = action
                trace["action_strategy"] = action_strategy
                trace["broker"] = broker_name

            # Position-aware exit: override hold/buy to sell_to_close for held positions
            if action in ("hold", "buy"):
                _positions = market_state.get("portfolio", {}).get("positions", {})
                _cqty = float(_positions.get(symbol, {}).get("qty", 0) or 0)
                if _cqty > 0:
                    _lp = market_state.get("last_price")
                    if _lp is None:
                        _pp = market_state.get("prices", [])
                        _lp = _pp[-1] if _pp else None
                    if _lp is not None:
                        _should_exit, _exit_reason = self._check_position_exit(
                            symbol, float(_lp), broker_state
                        )
                        if _should_exit:
                            action = "sell_to_close"
                            action_strategy = "position_exit"
                            if _exit_reason == "partial_take_profit":
                                _partial_ratio = float(
                                    broker_state.risk.cfg.get("partial_take_profit_ratio", 0.5) or 0.5
                                )
                                reduce_pct = max(0.1, min(_partial_ratio, 0.9))
                            else:
                                reduce_pct = 1.0
                            logging.info(
                                "Position exit for %s: %s (reduce_pct=%.2f)", symbol, _exit_reason, reduce_pct
                            )
                            _slog.event("info", "position_exit", symbol=symbol, reason=_exit_reason, broker=broker_name, reduce_pct=reduce_pct)
                            if trace:
                                trace.update(
                                    action="sell_to_close",
                                    action_strategy="position_exit",
                                    position_exit=True,
                                    position_exit_reason=_exit_reason,
                                    position_exit_reduce_pct=reduce_pct,
                                )

            # Guardrail check (might need lock if it has state, but typically stateless config)
            # We'll keep it locked for safety as it might access broker_state via _get_guardrail
            guardrail = self._get_guardrail(broker_name, symbol)
            if guardrail:
                guard_action = guardrail.generate_signal(market_state).get("action", "hold")
                mode = self.learning_cfg.get("guardrail", {}).get("mode", "confirm")
                action = self._apply_guardrail(action, guard_action, mode)
                if trace:
                    trace["guardrail_action"] = guard_action
                    trace["guardrail_mode"] = mode
                    trace["action"] = action
            if action == "hold":
                if self._cancel_pending_if_needed(symbol, action, broker=broker_name):
                    self._open_order_mgr.remove_pending(symbol, broker=broker_name)
                self._emit_decision_trace(trace, "hold", "strategies_hold", "signal")
                return None
            if action == "exit":
                if self._cancel_pending_if_needed(symbol, action, broker=broker_name):
                    self._open_order_mgr.remove_pending(symbol, broker=broker_name)
                self.broker.close_position(symbol, broker=broker_name)
                self._emit_decision_trace(trace, "exit", "strategy_exit", "signal")
                return None

            # Normalize sell_to_close to sell for the execution path
            if action == "sell_to_close":
                action = "sell"
                if trace:
                    trace["action"] = "sell"
                    trace["sell_to_close"] = True

            if self._open_order_mgr.has_pending(symbol, broker=broker_name):
                if self._cancel_pending_if_needed(symbol, action, broker=broker_name):
                    self._open_order_mgr.remove_pending(symbol, broker=broker_name)
                else:
                    self._record_skip(symbol, "hold", "open_order", broker_name)
                    logging.info("Skipping %s: open orders pending", symbol)
                    self._emit_decision_trace(trace, "skip", "open_order", "open_orders")
                    return None

            last_price = market_state.get("last_price")
            if last_price is None:
                prices = market_state.get("prices", [])
                last_price = prices[-1] if prices else None
            if last_price is None:
                self._record_skip(symbol, action, "no_price", broker_name)
                logging.info("Skipping %s for %s: no price available", action, symbol)
                self._emit_decision_trace(trace, "skip", "no_price", "pricing")
                return None
            min_price = self.cfg.get("trading_limits", {}).get("min_price")
            if min_price is not None and action == "buy" and last_price < float(min_price):
                self._record_skip(symbol, action, "min_price", broker_name)
                logging.info("Skipping %s for %s: price below min_price", action, symbol)
                self._emit_decision_trace(trace, "skip", "min_price", "limits")
                return None
            try:
                broker_state.last_prices[symbol] = float(last_price)
            except (TypeError, ValueError):
                pass

            portfolio = self._portfolio_for_broker(market_state.get("portfolio", {}), broker_name)
            market_state["portfolio"] = portfolio
            self._recalculate_exposure(market_state, portfolio, symbol)
            if trace:
                trace["portfolio"] = self._portfolio_snapshot_for_trace(portfolio, symbol)
                trace["exposure_pct"] = market_state.get("exposure_pct")
                trace["short_exposure_pct"] = market_state.get("short_exposure_pct")
                trace["leverage"] = market_state.get("leverage")
            # Determine if this is a position close (reduces risk, not new entry)
            _is_closing_position = action == "sell" and float(
                portfolio.get("positions", {}).get(symbol, {}).get("qty", 0) or 0
            ) > 0
            if self._is_account_blocked(broker_name):
                self._record_risk_outcome(symbol, action, False, "account_blocked", broker_name)
                self._record_skip(symbol, action, "account_blocked", broker_name)
                logging.info("Skipping %s for %s: account blocked", action, symbol)
                self._emit_decision_trace(trace, "skip", "account_blocked", "account")
                return None
            if not risk_disabled and not _is_closing_position:
                var_reason = self._account_metrics.var_limit_reason(broker_state)
                if var_reason:
                    self._record_risk_outcome(symbol, action, False, var_reason, broker_name)
                    self._record_skip(symbol, action, var_reason, broker_name)
                    logging.info("Skipping %s for %s: %s", action, symbol, var_reason)
                    self._emit_decision_trace(trace, "skip", var_reason, "risk")
                    return None
            can_short = True
            if action == "sell":
                positions = portfolio.get("positions", {})
                current_qty = float(positions.get(symbol, {}).get("qty", 0.0) or 0.0)
                can_short = self._can_short(symbol, portfolio)
                if not risk_disabled and current_qty <= 0 and not can_short:
                    self._record_risk_outcome(symbol, action, False, "shorting_disabled", broker_name)
                    self._record_skip(symbol, action, "shorting_disabled", broker_name)
                    logging.info("Skipping %s for %s: shorting disabled", action, symbol)
                    self._emit_decision_trace(trace, "skip", "shorting_disabled", "shorting")
                    return None
            if not risk_disabled and self._is_action_blocked(symbol, action):
                self._record_risk_outcome(symbol, action, False, "limit_block", broker_name)
                self._record_skip(symbol, action, "limit_block", broker_name)
                logging.info("Skipping %s for %s: limits block action", action, symbol)
                self._emit_decision_trace(trace, "skip", "limit_block", "limits")
                return None
            if action == "sell":
                qty, skip_reason = self._size_order(action, last_price, portfolio, symbol, market_state, reduce_pct=reduce_pct)
            else:
                qty, skip_reason = self._size_order(action, last_price, portfolio, symbol, market_state)
            if trace:
                trace["haircuts"] = self._haircut_snapshot(market_state)
            if qty <= 0:
                if skip_reason:
                    self._record_risk_outcome(symbol, action, False, str(skip_reason), broker_name)
                    self._record_skip(symbol, action, str(skip_reason), broker_name)
                    logging.info("Skipping %s for %s: %s", action, symbol, skip_reason)
                    self._emit_decision_trace(trace, "skip", str(skip_reason), "sizing")
                return None
            market_state["qty"] = qty
            if trace:
                trace["qty"] = qty

            now = datetime.now(timezone.utc)
            # --- Entry-only risk gates (skipped for position closes) ---
            if not risk_disabled and not _is_closing_position:
                caps_cfg = self.cfg.get("risk", {}).get("exposure_caps", {}) or {}
                violates, _ = RiskManager.check_exposure_caps(
                    symbol, action, qty, last_price, portfolio, caps_cfg,
                    venue_fn=self._symbol_mgr.symbol_venue,
                    sector_fn=self._symbol_mgr.symbol_sector,
                )
                if violates:
                    self._record_risk_outcome(symbol, action, False, "exposure_cap", broker_name)
                    self._record_skip(symbol, action, "exposure_cap", broker_name)
                    logging.info("Skipping %s for %s: exposure caps exceeded", action, symbol)
                    self._emit_decision_trace(trace, "skip", "exposure_cap", "risk")
                    return None
                blocked, reason = broker_state.risk.risk_preflight(
                    qty, last_price,
                    self.cfg.get("trading_limits", {}),
                    broker_state.last_trade_at, now,
                    exposure_pct=market_state.get("exposure_pct", 0.0),
                    short_exposure_pct=market_state.get("short_exposure_pct", 0.0),
                    leverage=market_state.get("leverage", 1.0),
                )
                if blocked:
                    category = {"order_limit": "limits", "cooldown": "cooldown", "risk_block": "risk"}.get(reason, "risk")
                    self._record_risk_outcome(symbol, action, False, reason, broker_name)
                    self._record_skip(symbol, action, reason, broker_name)
                    logging.info("Skipping %s for %s: %s", action, symbol, reason)
                    self._emit_decision_trace(trace, "skip", reason, category)
                    return None
                self._record_risk_outcome(symbol, action, True, "ok", broker_name)
            else:
                self._record_risk_outcome(symbol, action, True,
                    "risk_disabled" if risk_disabled else "position_close", broker_name)

            order_type = str(order_meta.get("order_type") or "market").lower()
            limit_price = order_meta.get("limit_price")
            algo_name = order_meta.get("algo")
            if order_type == "limit" and limit_price is None:
                self._record_skip(symbol, action, "limit_price_missing", broker_name)
                logging.info("Skipping %s for %s: limit price missing", action, symbol)
                self._emit_decision_trace(trace, "skip", "limit_price_missing", "pricing")
                return None
            slices = []
            algo_label = str(algo_name or "").lower()
            if order_type == "market" and algo_label not in {"none", "off"}:
                slices = self._plan_execution(action, qty, last_price, market_state, algo_name)

            order_notional = qty * last_price
            order_queue = self._order_queues.get(broker_name, self._order_queue)
            if order_queue is None:
                self._record_skip(symbol, action, "order_queue_missing", broker_name)
                logging.warning("Skipping %s for %s: no order queue for broker %s", action, symbol, broker_name)
                self._emit_decision_trace(trace, "skip", "order_queue_missing", "execution")
                return None
            order_start = time.perf_counter()
            if slices:
                order_id = None
                for order_slice in slices:
                    order_queue.enqueue(
                        symbol,
                        action,
                        qty=order_slice.qty,
                        order_type=order_type,
                        limit_price=limit_price,
                        extended_hours=bool(market_state.get("market_extended", False)),
                        earliest_at=order_slice.earliest_at,
                        notional=order_slice.qty * last_price,
                    )
            else:
                order_id = order_queue.enqueue(
                    symbol,
                    action,
                    qty=qty,
                    order_type=order_type,
                    limit_price=limit_price,
                    extended_hours=bool(market_state.get("market_extended", False)),
                    notional=order_notional,
                )
            order_latency = time.perf_counter() - order_start
            ORDER_LATENCY.labels(symbol=symbol, side=action).observe(order_latency)
            if trace:
                trace["order_latency_seconds"] = order_latency
            if action == "buy":
                strategy_label = action_strategy or (names[0] if names else None)
                if strategy_label:
                    broker_state.pending_entry_strategy[symbol] = {"strategy": strategy_label, "ts": now}
            if order_id and action in ("buy", "sell"):
                TRADES.labels(symbol=symbol, side=action).inc()
                TRADES_BY_BROKER.labels(broker=broker_name, symbol=symbol, side=action).inc()
                broker_state.last_trade_at = now
                _slog.event(
                    "info", "trade_executed",
                    symbol=symbol, action=action, qty=qty,
                    price=last_price, broker=broker_name,
                    order_type=order_type, strategy=action_strategy,
                    notional=qty * last_price,
                )
                self._emit_decision_trace(
                    trace,
                    "order_enqueued",
                    "ok",
                    "execution",
                    {
                        "order_type": order_type,
                        "limit_price": limit_price,
                        "algo": algo_name,
                    },
                )
            elif action in ("buy", "sell"):
                self._record_skip(symbol, action, "order_failed", broker_name)
                self._emit_decision_trace(trace, "skip", "order_failed", "execution")
            return order_id

    def _cancel_pending_if_needed(self, symbol: str, action: str, broker: str | None = None) -> bool:
        pending = self._open_order_mgr.get_pending(symbol, broker=broker)
        if not pending:
            return False
        if action in ("hold", "exit"):
            return self._cancel_pending_orders(pending, symbol)
        pending_sides = {str(order.get("side", "")).lower() for order in pending}
        action_side = action.lower()
        if action_side in ("buy", "sell") and pending_sides and action_side not in pending_sides:
            return self._cancel_pending_orders(pending, symbol)
        return False

    def _cancel_pending_orders(self, orders: list[dict], symbol: str) -> bool:
        canceled = False
        for order in orders:
            order_id = order.get("order_id")
            try:
                broker_name = order.get("broker") or self._broker_name
                broker = self._broker_map.get(broker_name, self.broker)
                broker.cancel_order(str(order_id))
                queue = self._order_queues.get(broker_name, self._order_queue)
                if queue:
                    queue.mark_cancel_requested(str(order_id))
                canceled = True
            except Exception as exc:
                logging.warning("Cancel order failed for %s (%s): %s", symbol, order_id, exc)
        return canceled

    def _flush_order_responses(self) -> None:
        for queue in self._order_queues.values():
            responses = queue.pop_responses()
            if not responses:
                continue
            for response in responses:
                try:
                    self._orchestrator.on_order_update(response.__dict__)
                except (AttributeError, TypeError, ValueError, KeyError) as exc:
                    logging.warning("Orchestrator order feedback failed: %s", exc)
                if self._live_reward_tracker is not None:
                    try:
                        reward_event = self._live_reward_tracker.on_order_response(response.__dict__)
                        if reward_event and isinstance(self._orchestrator, RLStrategyOrchestrator):
                            try:
                                self._orchestrator.on_trade_reward(reward_event)
                            except (AttributeError, TypeError, ValueError, KeyError) as exc:
                                logging.warning("Orchestrator reward update failed: %s", exc)
                    except (AttributeError, TypeError, ValueError, KeyError) as exc:
                        logging.warning("Live reward update failed: %s", exc)

    def _size_order(
        self,
        action: str,
        last_price: float,
        portfolio: dict,
        symbol: str,
        market_state: dict,
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
        vol_scale = self._vol_target_scale(market_state)
        portfolio_scale = self._portfolio_position_scale(symbol, market_state, portfolio)
        max_pos_pct *= vol_scale * portfolio_scale
        max_short_pct = float(self.cfg["risk"]["max_short_exposure_pct"])
        max_short_pct *= vol_scale * portfolio_scale
        allow_shorts = bool(self._strategy_params.get("allow_shorts", False))
        limits = self.cfg.get("trading_limits", {})
        if limits.get("enabled") and not limits.get("allow_shorts", True):
            allow_shorts = False
        cash = max(0.0, cash - self._reserved_cash(symbol, last_price, portfolio.get("broker")))

        if action == "buy":
            if equity <= 0 or cash <= 0:
                return 0, "insufficient_cash"
            target_value = equity * (max_pos_pct / 100.0)
            remaining_value = max(0.0, target_value - max(current_value, 0.0))
            if remaining_value <= 0:
                return 0, "position_limit"
            allowed_value = min(remaining_value, cash)
            allowed_value, haircut_reasons, haircut_metrics = apply_haircuts(
                self.cfg.get("risk", {}),
                allowed_value,
                last_price,
                market_state,
            )
            if haircut_metrics:
                market_state.update(haircut_metrics)
            if allowed_value < last_price:
                if "illiquid_spread" in haircut_reasons:
                    return 0, "illiquid_spread"
                if "illiquid_volume" in haircut_reasons:
                    return 0, "illiquid_volume"
                if haircut_reasons:
                    return 0, "liquidity_haircut"
                return 0, "insufficient_cash"
            return int(allowed_value // last_price), None

        if action == "sell":
            if current_qty > 0:
                qty = int(current_qty * max(min(reduce_pct, 1.0), 0.0))
                return (qty, None) if qty > 0 else (0, "position_limit")
            if not allow_shorts or not self._can_short(symbol, portfolio):
                return 0, "shorting_disabled"
            if equity <= 0:
                return 0, "insufficient_equity"
            target_value = equity * (max_short_pct / 100.0)
            current_short = float(portfolio.get("short_exposure", 0.0) or 0.0)
            remaining_value = max(0.0, target_value - current_short)
            if remaining_value <= 0:
                return 0, "short_limit"
            buying_power = float(portfolio.get("buying_power", 0.0) or 0.0)
            if buying_power <= 0:
                buying_power = cash
            allowed_value = min(remaining_value, buying_power)
            allowed_value, haircut_reasons, haircut_metrics = apply_haircuts(
                self.cfg.get("risk", {}),
                allowed_value,
                last_price,
                market_state,
            )
            if haircut_metrics:
                market_state.update(haircut_metrics)
            if allowed_value < last_price:
                if "illiquid_spread" in haircut_reasons:
                    return 0, "illiquid_spread"
                if "illiquid_volume" in haircut_reasons:
                    return 0, "illiquid_volume"
                if haircut_reasons:
                    return 0, "liquidity_haircut"
                return 0, "insufficient_buying_power"
            return int(allowed_value // last_price), None

        return 0, "unsupported"

    def _vol_target_scale(self, market_state: dict) -> float:
        cfg = self.cfg.get("risk", {}).get("vol_targeting", {})
        if not cfg.get("enabled", False):
            return 1.0
        prices = market_state.get("prices", []) or []
        realized = realized_volatility_pct(prices)
        if realized <= 0:
            return 1.0
        target = float(cfg.get("target_vol_pct", 2.0))
        scale = target / realized
        min_scale = float(cfg.get("min_scale", 0.5))
        max_scale = float(cfg.get("max_scale", 1.5))
        return max(min(scale, max_scale), min_scale)

    def _portfolio_position_scale(self, symbol: str, market_state: dict, portfolio: dict) -> float:
        """Adjust position size based on portfolio optimization."""
        if not self._portfolio_enabled or self._portfolio_optimizer is None:
            return 1.0
        try:
            positions = portfolio.get("positions", {})
            if not positions:
                return 1.0
            # Get current weights
            equity = float(portfolio.get("equity", 0.0) or 0.0)
            if equity <= 0:
                return 1.0
            current_weights = {}
            for sym, pos in positions.items():
                qty = float(pos.get("qty", 0.0) or 0.0)
                price = float(pos.get("current_price", 0.0) or market_state.get("last_price", 0.0) or 0.0)
                if price > 0 and qty > 0:
                    current_weights[sym] = (qty * price) / equity
            if symbol not in current_weights:
                current_weights[symbol] = 0.0
            # Check if symbol is over-allocated (risk parity adjustment)
            portfolio_cfg = self.cfg.get("portfolio", {})
            max_pos_pct = float(portfolio_cfg.get("constraints", {}).get("max_position_pct", 0.25))
            current_weight = current_weights.get(symbol, 0.0)
            if current_weight >= max_pos_pct:
                return 0.5  # Reduce allocation if overweight
            # Scale based on distance from max
            headroom = max_pos_pct - current_weight
            if headroom < 0.05:  # Less than 5% headroom
                return 0.7
            return 1.0
        except (ValueError, TypeError, KeyError) as exc:
            logging.warning("Portfolio position scale failed for %s: %s", symbol, exc)
            PORTFOLIO_SCALE_FALLBACK.labels(symbol=symbol).inc()
            return 1.0

    def _limits_enabled(self) -> bool:
        return bool(self.cfg.get("trading_limits", {}).get("enabled", False))

    def _is_account_blocked(self, broker: str | None = None) -> bool:
        if not self._limits_enabled():
            return False
        limits = self.cfg.get("trading_limits", {})
        if not limits.get("enforce_account_flags", True):
            return False
        account = self._account_for_broker(broker)
        for key in ("account_blocked", "trading_blocked", "trade_suspended_by_user"):
            if str(account.get(key, "")).lower() in {"true", "1", "yes"} or account.get(key) == True:
                return True
        return False

    def _is_action_blocked(self, symbol: str, action: str) -> bool:
        if not self._limits_enabled():
            return False
        limits = self.cfg.get("trading_limits", {})
        blocked_symbols = {s.upper() for s in limits.get("blocked_symbols", [])}
        if symbol.upper() in blocked_symbols:
            return True
        blocked_actions = {str(a).lower() for a in limits.get("blocked_actions", [])}
        return action.lower() in blocked_actions

    def _can_short(self, symbol: str, portfolio: dict) -> bool:
        limits = self.cfg.get("trading_limits", {})
        positions = portfolio.get("positions", {})
        current_qty = float(positions.get(symbol, {}).get("qty", 0.0) or 0.0)
        if current_qty > 0:
            return True
        account = self._account_for_broker(portfolio.get("broker"))
        shorting_enabled = account.get("shorting_enabled")
        if shorting_enabled != True:
            return False
        if not self._limits_enabled():
            return True
        if not limits.get("allow_shorts", True):
            return False
        return True

    def _account_for_broker(self, broker: str | None) -> dict:
        account = self._account_snapshot or {}
        if broker and isinstance(account, dict):
            brokers = account.get("brokers")
            if isinstance(brokers, dict):
                raw = brokers.get(broker, {}).get("raw")
                if isinstance(raw, dict):
                    return raw
        return account

    # Risk check methods moved to RiskManager (app/risk/manager.py)

    def _resolve_strategy_names(self) -> list[str]:
        names = self._strategy_cfg.names
        if isinstance(names, list) and names:
            return [str(name) for name in names]
        return [str(self._strategy_cfg.name)]

    def _resolve_broker_map(self) -> dict[str, object]:
        if hasattr(self.broker, "brokers"):
            return getattr(self.broker, "brokers")
        return {self._resolve_default_broker_name(): self.broker}

    def _resolve_default_broker_name(self) -> str:
        brokers_cfg = self.cfg.get("brokers", {})
        routing_default = str(self._routing_cfg.get("default", "")).strip()
        if routing_default:
            if routing_default == "alpaca" and brokers_cfg.get("alpaca", {}).get("enabled", True):
                return routing_default
            if routing_default == "ibkr" and brokers_cfg.get("ibkr", {}).get("enabled", False):
                return routing_default
        if brokers_cfg.get("ibkr", {}).get("enabled", False):
            return "ibkr"
        return "alpaca"

    def _resolve_broker_for_symbol(
        self,
        symbol: str,
        selected_strategies: list[str] | None = None,
        signals: list[dict] | None = None,
        action_strategy: str | None = None,
    ) -> str:
        broker_name = routing_utils.resolve_broker_for_symbol(
            symbol=symbol,
            broker_names=list(self._broker_map.keys()),
            routing_cfg=self._routing_cfg,
            selected_strategies=selected_strategies,
            signals=signals,
            action_strategy=action_strategy,
        )
        if not broker_name:
            broker_name = self._broker_name
        return self._maybe_fallback_broker(broker_name)

    def _auto_split_broker(self, symbol: str) -> str:
        broker_names = sorted(self._broker_map.keys())
        broker_name = routing_utils.auto_split_broker(symbol, broker_names)
        return broker_name or self._broker_name

    def _normalize_broker_name(self, broker_name: str) -> str:
        return routing_utils.normalize_broker_name(broker_name, list(self._broker_map.keys()))

    def _build_symbol_batches(self, symbols: list[str]) -> list[tuple[str, str | None, list[str]]]:
        routing_mode = str(self._routing_cfg.get("mode", "default")).lower()
        broker_names = list(self._broker_map.keys())

        if routing_mode == "parallel" and len(broker_names) > 1:
            # Parallel mode: each broker gets symbols based on buying power
            broker_buying_power = self._get_broker_buying_power()
            buckets = routing_utils.parallel_partition_symbols(
                symbols,
                broker_names,
                broker_buying_power,
                self._routing_cfg,
                min_symbols=1,
            )
            # Cache for active symbol labels
            self._symbol_mgr.symbols_by_broker = buckets
            return [
                ("parallel", broker, batch)
                for broker, batch in buckets.items()
                if batch
            ]

        if routing_mode == "auto_split" and len(broker_names) > 1:
            if self._symbol_mgr.symbols_by_broker:
                return [
                    ("auto_split", broker, batch)
                    for broker, batch in self._symbol_mgr.symbols_by_broker.items()
                    if batch
                ]
            buckets = routing_utils.partition_symbols(symbols, broker_names, self._routing_cfg)
            return [("auto_split", broker, batch) for broker, batch in buckets.items()]

        return [("default", None, symbols)]

    def _get_broker_buying_power(self) -> dict[str, float]:
        """Get buying power for all brokers from their state."""
        result: dict[str, float] = {}
        for name, state in self._broker_states.items():
            result[name] = state.buying_power
        return result

    def _update_open_order_queues(self) -> None:
        if len(self._broker_map) > 1:
            orders_by_broker: dict[str, list[dict]] = {}
            for order in self._open_order_mgr.cache:
                broker_name = order.get("broker") or self._broker_name
                orders_by_broker.setdefault(str(broker_name), []).append(order)
            for broker_name, queue in self._order_queues.items():
                queue.update(orders_by_broker.get(broker_name, []))
        else:
            if self._order_queue:
                self._order_queue.update(self._open_order_mgr.cache)

    def _prepare_market_data(self, market_data_provider, symbols: list[str]) -> None:
        data_cfg = self.cfg.get("data", {}) or {}
        if not data_cfg.get("prefetch_enabled", True):
            return
        if not data_cfg.get("prefetch_after_filter", True):
            return
        if hasattr(market_data_provider, "prepare"):
            try:
                market_data_provider.prepare(symbols)
            except Exception as exc:
                logging.warning("Market data prefetch failed: %s", exc)

    def _run_cycle_maintenance(self, symbols: list[str], portfolio: dict) -> list[str]:
        for name in self._strategy_names:
            STRATEGY_ACTIVE.labels(strategy=name).set(0 if self._strategy_disabled_globally(name) else 1)
        for broker_name in self._broker_names:
            BROKER_ACTIVE.labels(broker=broker_name).set(1)
        if self._live_reward_tracker is not None:
            try:
                self._live_reward_tracker.sync_positions(portfolio)
            except Exception as exc:
                logging.warning("Live reward sync failed: %s", exc)
        self._account_metrics.update(self.broker, self._broker_states, self._broker_name)
        self._update_position_metrics(portfolio)
        if self._perf_tracker.enabled:
            brokers = portfolio.get("brokers", {})
            if isinstance(brokers, dict) and brokers:
                for broker_name in brokers.keys():
                    broker_portfolio = self._portfolio_for_broker(portfolio, broker_name)
                    broker_state = self._broker_state(broker_name)
                    self._perf_tracker.update_from_positions(
                        broker_state, broker_portfolio, broker_name,
                        self._strategy_names, broker_state.last_prices,
                    )
            else:
                broker_state = self._broker_state(self._broker_name)
                self._perf_tracker.update_from_positions(
                    broker_state, portfolio, self._broker_name,
                    self._strategy_names, broker_state.last_prices,
                )
        self._perf_tracker.maybe_report(
            self._broker_states, self._strategy_disabled_globally
        )
        self._refresh_news_cache(symbols)
        self._log_news_cache()
        self._symbol_mgr.refresh_dynamic_symbols(portfolio, self._strategy_names, signal_metrics_fn=self._update_signal_metrics_from_values)
        self._symbol_mgr.refresh_symbol_venues()
        self._maybe_checkpoint()
        self._symbol_mgr.log_ai_filter_heartbeat()
        self._maybe_reload_active_model()
        symbols = self._symbol_mgr.resolve_active_symbols()
        symbols = self._symbol_mgr.merge_symbols_with_positions(symbols, portfolio)
        self._symbol_mgr.update_active_symbol_metrics(symbols, self._routing_cfg, self._get_broker_buying_power)
        self._open_order_mgr.refresh(self.broker, self._broker_map, symbols)
        self._maybe_force_liquidation(portfolio)
        return symbols

    def _process_single_symbol(
        self,
        sym: str,
        portfolio: dict,
        market_data_provider,
        broker_override: str | None,
        skip_unchanged: bool,
        news_snapshot: dict | None = None,
        orders_snapshot: list | None = None,
    ) -> None:
        if not self._symbol_mgr.is_symbol_market_open(sym):
            return
        
        # Fetch market data UNLOCKED (IO bound)
        try:
            market_state = market_data_provider(sym)
        except Exception as exc:
            broker_name = broker_override or self._broker_name
            with self._lock:
                self._record_skip(sym, "hold", "symbol_error", broker_name)
            logging.warning("Skipping %s: data fetch error: %s", sym, exc)
            return

        # Validate market data
        last_price = market_state.get("last_price")
        prices = market_state.get("prices", [])
        if last_price is None and not prices:
            logging.warning("Skipping %s: no price data available", sym)
            return
        if last_price is not None and last_price <= 0:
            logging.warning("Skipping %s: invalid last_price=%s", sym, last_price)
            return

        # Process Logic (Locking handled inside run_once and critical sections)
        try:
            broker_name = broker_override or self._broker_name
            
            # Access broker state with lock
            with self._lock:
                broker_state = self._broker_state(broker_name)
                last_bar_ts = market_state.get("last_bar_ts")
                if last_bar_ts is not None:
                    last_seen = broker_state.last_bar_ts.get(sym)
                    if last_seen == last_bar_ts and skip_unchanged:
                        return
                    broker_state.last_bar_ts[sym] = last_bar_ts
                
                # Check risk outcome (read safe-ish, but keeping lock for consistency)
                market_state["risk_outcome"] = broker_state.risk_outcomes.get(sym, {})

            self._enrich_market_state(market_state, portfolio, sym,
                                     news_snapshot=news_snapshot,
                                     orders_snapshot=orders_snapshot)

            if broker_override:
                market_state["broker_override"] = broker_override

            # Access strategy symbols with lock
            with self._lock:
                market_state["strategy_symbols"] = self._symbol_mgr.symbols_by_strategy

            self._update_signal_metrics(sym, market_state)

            decision_start = time.perf_counter()
            market_state["_decision_start"] = decision_start

            # Deep-copy market_state to prevent threads from mutating each other's data
            ms_copy = copy.deepcopy(market_state)
            self.run_once(sym, ms_copy)
            
            DECISION_LATENCY.labels(symbol=sym).observe(time.perf_counter() - decision_start)
        except Exception as exc:
                broker_name = broker_override or self._broker_name
                self._record_skip(sym, "hold", "symbol_error", broker_name)
                logging.warning("Skipping %s: symbol processing error: %s", sym, exc)
                trace = self._init_decision_trace(sym, market_state)
                if trace:
                    self._emit_decision_trace(
                        trace,
                        "skip",
                        "symbol_error",
                        "symbol_loop",
                        {"error": str(exc)},
                    )

    def _run_symbol_batch(
        self,
        symbols: list[str],
        portfolio: dict,
        market_data_provider,
        broker_override: str | None,
    ) -> None:
        skip_unchanged = bool(self.cfg.get("data", {}).get("process_on_new_bar_only", False))

        # Determine effective portfolio for batch
        batch_portfolio = portfolio
        if broker_override:
            batch_portfolio = self._portfolio_for_broker(portfolio, broker_override)

        # Snapshot shared state under lock before dispatching to threads
        with self._lock:
            news_snap = dict(self._news_cache)
            orders_snap = list(self._open_order_mgr.cache)

        # Use persistent ThreadPoolExecutor for parallel processing
        futures = [
            self._symbol_executor.submit(
                self._process_single_symbol,
                sym,
                batch_portfolio,
                market_data_provider,
                broker_override,
                skip_unchanged,
                news_snap,
                orders_snap,
            )
            for sym in symbols
        ]
        wait(futures)

    def _portfolio_for_broker(self, portfolio: dict, broker_name: str) -> dict:
        brokers = portfolio.get("brokers")
        if isinstance(brokers, dict) and broker_name in brokers:
            data = dict(brokers[broker_name])
            data.setdefault("broker", broker_name)
            return data
        return portfolio

    def _maybe_fallback_broker(self, broker_name: str) -> str:
        if not self._routing_cfg.get("fallback_enabled", False):
            return broker_name
        broker = self._broker_map.get(broker_name)
        if broker and broker.is_connected():
            return broker_name
        for name, alt in self._broker_map.items():
            if alt.is_connected():
                return name
        return broker_name

    def _combine_signals(
        self,
        signals: list[dict],
        weights: dict[str, float] | None = None,
        order: list[str] | None = None,
    ) -> tuple[str, float, str | None]:
        if not signals:
            return "hold", 1.0, None
        for signal in signals:
            if signal.get("action") == "exit":
                return "exit", 1.0, signal.get("name")
        mode = self._combine_mode
        if mode == "priority":
            order = order or self._strategy_names
            for name in order:
                for signal in signals:
                    if signal.get("name") == name:
                        action = signal.get("action", "hold")
                        reduce_pct = float(signal.get("reduce_pct", 1.0))
                        return action, reduce_pct, name
            return "hold", 1.0, None
        weights = weights or {}
        min_conviction = self._strategy_cfg.min_conviction
        buy_score = 0.0
        sell_score = 0.0
        sells = []
        top_signal = None
        top_weight = None
        top_buy = None
        top_buy_weight = None
        top_sell = None
        top_sell_weight = None
        for signal in signals:
            action = signal.get("action")
            name = signal.get("name")
            confidence = float(signal.get("confidence", 1.0))
            strategy_weight = float(weights.get(name, 1.0))
            effective_weight = confidence * strategy_weight
            if top_weight is None or effective_weight > top_weight:
                top_weight = effective_weight
                top_signal = signal
            if action == "buy":
                buy_score += effective_weight
                if top_buy_weight is None or effective_weight > top_buy_weight:
                    top_buy_weight = effective_weight
                    top_buy = signal
            elif action == "sell":
                sell_score += effective_weight
                sells.append(signal)
                if top_sell_weight is None or effective_weight > top_sell_weight:
                    top_sell_weight = effective_weight
                    top_sell = signal
        if buy_score == sell_score:
            return "hold", 1.0, top_signal.get("name") if top_signal else None
        winning_score = max(buy_score, sell_score)
        if min_conviction > 0 and winning_score < min_conviction:
            return "hold", 1.0, top_signal.get("name") if top_signal else None
        if buy_score > sell_score:
            chosen = top_buy or top_signal
            return "buy", 1.0, chosen.get("name") if chosen else None
        reduce_pct = max(float(s.get("reduce_pct", 1.0)) for s in sells) if sells else 1.0
        chosen = top_sell or top_signal
        strategy = chosen.get("name") if chosen else None
        return "sell", reduce_pct, strategy

    # Account metrics methods moved to AccountMetricsUpdater (app/agents/account_metrics.py)

    def _update_drift_monitor(self, day_pnl_pct: float) -> None:
        if not self._drift_monitor:
            return
        self._drift_monitor.update_pnl(day_pnl_pct)
        reasons = self._drift_monitor.check_drift()
        if reasons:
            self._handle_drift(reasons)

    def _handle_drift(self, reasons: list[str]) -> None:
        if self._drift_rollback_done or not self._drift_auto_rollback:
            return
        self.learning_cfg["use_best_model"] = True
        self._reload_rl_strategies()
        self._drift_rollback_done = True
        logging.warning("Drift detected (%s). Reloaded RL policies using best model.", ", ".join(reasons))

    def _reload_rl_strategies(self) -> None:
        RLPolicyStrategy.clear_model_cache()
        for symbol, strategies in list(self._strategy_by_symbol.items()):
            for name in list(strategies.keys()):
                if name in {"rl_policy", "rl_policy_fees"}:
                    del strategies[name]
            if not strategies:
                del self._strategy_by_symbol[symbol]

    # _violates_exposure_caps moved to RiskManager.check_exposure_caps

    def _apply_kill_switch_profile(self, market_state: dict) -> None:
        profile_cfg = self.cfg.get("risk", {}).get("kill_switch_profiles", {}) or {}
        if not profile_cfg.get("enabled", False):
            return
        mode = str(profile_cfg.get("mode", "static"))
        profiles = profile_cfg.get("profiles", {}) or {}
        if not profiles:
            return
        name = profile_cfg.get("current")
        if mode == "adaptive":
            name = self._select_profile_name(profile_cfg, market_state)
        if not name or name not in profiles:
            return
        if self._active_kill_switch_profile == name:
            return
        updates = profiles.get(name, {}) or {}
        risk_cfg = self.cfg.get("risk", {})
        for key, value in updates.items():
            risk_cfg[key] = value
        self._active_kill_switch_profile = name
        logging.info("Kill switch profile set to %s", name)
        _slog.event("info", "kill_switch_profile", profile=name)

    def _select_profile_name(self, profile_cfg: dict, market_state: dict) -> str | None:
        adaptive = profile_cfg.get("adaptive", {}) or {}
        low_max = float(adaptive.get("low_vol_max_pct", 1.0))
        high_min = float(adaptive.get("high_vol_min_pct", 3.0))
        vol = _realized_volatility_pct(market_state)
        if vol <= low_max:
            return "low"
        if vol >= high_min:
            return "high"
        return "medium"

    def _update_position_metrics(self, portfolio: dict) -> None:
        positions = portfolio.get("positions", {})
        current = set()
        for symbol, pos in positions.items():
            qty = float(pos.get("qty", 0.0) or 0.0)
            value = float(pos.get("value", 0.0) or 0.0)
            if qty == 0 and value == 0:
                continue
            POSITION_QTY.labels(symbol=symbol).set(qty)
            POSITION_VALUE.labels(symbol=symbol).set(value)
            current.add(symbol)
        removed = self._position_symbols - current
        for symbol in removed:
            POSITION_QTY.labels(symbol=symbol).set(0)
            POSITION_VALUE.labels(symbol=symbol).set(0)
        self._position_symbols = current
        brokers = portfolio.get("brokers", {})
        if isinstance(brokers, dict):
            for broker_name, data in brokers.items():
                broker_positions = data.get("positions", {}) if isinstance(data, dict) else {}
                current_broker = self._position_symbols_by_broker.get(broker_name, set())
                new_broker = set()
                for symbol, pos in broker_positions.items():
                    qty = float(pos.get("qty", 0.0) or 0.0)
                    value = float(pos.get("value", 0.0) or 0.0)
                    if qty == 0 and value == 0:
                        continue
                    POSITION_QTY_BY_BROKER.labels(broker=broker_name, symbol=symbol).set(qty)
                    POSITION_VALUE_BY_BROKER.labels(broker=broker_name, symbol=symbol).set(value)
                    new_broker.add(symbol)
                removed = current_broker - new_broker
                for symbol in removed:
                    POSITION_QTY_BY_BROKER.labels(broker=broker_name, symbol=symbol).set(0)
                    POSITION_VALUE_BY_BROKER.labels(broker=broker_name, symbol=symbol).set(0)
                self._position_symbols_by_broker[broker_name] = new_broker

    # Performance tracking methods moved to PerformanceTracker (app/agents/performance.py)

    def _run_reporting_loop(self):
        logging.info("Starting reporting loop thread.")
        while True:
            try:
                # Update account metrics (includes PnL, Equity, Drift)
                self._account_metrics.update(self.broker, self._broker_states, self._broker_name)
                market_open = is_market_open(self.cfg)
                self._last_market_open = self._account_metrics.update_market_open_metrics(
                    self.cfg, self._broker_names or [self._broker_name], market_open, self._last_market_open
                )
            except Exception as exc:
                logging.warning("Reporting loop error: %s", exc)
            time.sleep(15)

    def loop(self, symbol: str | list[str], market_data_provider, interval_seconds: int = 60):
        self._symbol_mgr.symbols = symbol if isinstance(symbol, list) else [symbol]
        
        # Start decoupled reporting thread
        reporting_thread = threading.Thread(target=self._run_reporting_loop, daemon=True)
        reporting_thread.start()
        
        while True:
            if should_restart(self._started_at):
                logging.info("Restart requested; exiting trading loop.")
                raise SystemExit(0)
            portfolio = self._get_portfolio_snapshot()
            if portfolio is None:
                self._log_broker_missing()
                for broker_name in self._broker_names or [self._broker_name]:
                    BROKER_ACTIVE.labels(broker=broker_name).set(0)
                    BROKER_MARKET_OPEN.labels(broker=broker_name).set(0)
                time.sleep(interval_seconds)
                continue
            symbols = self._symbol_mgr.resolve_active_symbols()
            symbols = self._run_cycle_maintenance(symbols, portfolio)
            self._update_open_order_queues()
            self._flush_order_responses()
            if self._ops_state_blocks_run():
                time.sleep(interval_seconds)
                continue
            # Market open check is also done in reporting loop, but we need it here for logic control
            market_open = is_market_open(self.cfg)
            if not market_open:
                time.sleep(interval_seconds)
                continue
            self._prepare_market_data(market_data_provider, symbols)
            symbol_batches = self._build_symbol_batches(symbols)
            for _, broker_override, batch in symbol_batches:
                self._run_symbol_batch(batch, portfolio, market_data_provider, broker_override)
            time.sleep(interval_seconds)

    def _ops_state_blocks_run(self) -> bool:
        ms_cfg = self.cfg.get("healthwatch", {}).get("market_shutdown", {}) or {}
        if not ms_cfg.get("write_state", False):
            return False
        ops_state = load_ops_state(ms_cfg.get("state_path", "/data/system_state.json"))
        if ops_state_is_sleeping(ops_state):
            return True
        if ops_state and not ops_state_is_running(ops_state):
            return False
        return False

    def _log_broker_missing(self) -> None:
        now = time.monotonic()
        if now - self._broker_missing_last_log < 60:
            return
        self._broker_missing_last_log = now
        if self._broker_missing_at is None:
            self._broker_missing_at = datetime.now(timezone.utc)
        brokers = ", ".join(self._broker_names or [self._broker_name])
        reason = self._broker_missing_reason or "broker not initialized"
        logging.warning("Broker unavailable; trading loop idle. brokers=%s reason=%s", brokers, reason)

    def _maybe_checkpoint(self) -> None:
        broker_states_payload: dict[str, dict[str, object]] = {}
        for name, state in self._broker_states.items():
            broker_states_payload[name] = {
                "last_trade_at": _dt_to_str(state.last_trade_at),
                "equity_start": state.equity_start,
                "equity_peak": state.equity_peak,
                "disabled_strategies": sorted(state.disabled_strategies),
            }
        default_state = self._broker_states.get(self._broker_name)
        payload = {
            "dynamic_symbols": self._symbol_mgr.dynamic_symbols,
            "dynamic_symbols_at": _dt_to_str(self._symbol_mgr.dynamic_symbols_at),
            "symbols": self._symbol_mgr.symbols,
            "symbols_by_strategy": self._symbol_mgr.symbols_by_strategy,
            "symbol_venues": self._symbol_mgr.symbol_venues,
            "symbol_venues_at": _dt_to_str(self._symbol_mgr.symbol_venues_at),
            "news_cache": self._news_cache,
            "news_cache_at": _dt_to_str(self._news_cache_at),
            "last_trade_at": _dt_to_str(default_state.last_trade_at) if default_state else None,
            "equity_start": self._account_metrics.equity_start,
            "equity_peak": self._account_metrics.equity_peak,
            "broker_states": broker_states_payload,
        }
        self._checkpoint_at = maybe_save_checkpoint("trader", payload, self.cfg, self._checkpoint_at)

    def _load_checkpoint(self) -> None:
        data = load_checkpoint("trader", self.cfg)
        if not data:
            return
        payload = data.get("payload", {}) or {}
        self._symbol_mgr.dynamic_symbols = list(payload.get("dynamic_symbols", []))
        self._symbol_mgr.dynamic_symbols_at = _dt_from_str(payload.get("dynamic_symbols_at"))
        self._symbol_mgr.symbols = list(payload.get("symbols", []))
        self._symbol_mgr.symbols_by_strategy = dict(payload.get("symbols_by_strategy", {}) or {})
        if not self._symbol_mgr.symbols and self._symbol_mgr.dynamic_symbols:
            self._symbol_mgr.symbols = list(self._symbol_mgr.dynamic_symbols)
        self._symbol_mgr.symbol_venues = dict(payload.get("symbol_venues", {}) or {})
        self._symbol_mgr.symbol_venues_at = _dt_from_str(payload.get("symbol_venues_at"))
        self._news_cache = dict(payload.get("news_cache", {}) or {})
        self._news_cache_at = _dt_from_str(payload.get("news_cache_at"))
        self._account_metrics.equity_start = payload.get("equity_start")
        self._account_metrics.equity_peak = payload.get("equity_peak")
        broker_payload = payload.get("broker_states", {}) or {}
        if isinstance(broker_payload, dict):
            for name, state in broker_payload.items():
                if name not in self._broker_states:
                    continue
                if not isinstance(state, dict):
                    continue
                broker_state = self._broker_states[name]
                broker_state.last_trade_at = _dt_from_str(state.get("last_trade_at"))
                broker_state.equity_start = state.get("equity_start")
                broker_state.equity_peak = state.get("equity_peak")
                disabled = state.get("disabled_strategies")
                if isinstance(disabled, list):
                    broker_state.disabled_strategies = {str(item) for item in disabled}
        legacy_last_trade_at = _dt_from_str(payload.get("last_trade_at"))
        if legacy_last_trade_at and self._broker_name in self._broker_states:
            broker_state = self._broker_states[self._broker_name]
            if broker_state.last_trade_at is None:
                broker_state.last_trade_at = legacy_last_trade_at

    def _orchestrator_key(self, symbol: str, broker_name: str | None) -> tuple[str, str]:
        broker = broker_name or self._broker_name
        return (broker, symbol)

    def _update_orchestrator(
        self, symbol: str, market_state: dict, broker_name: str | None = None
    ) -> None:
        broker_name = (
            broker_name
            or market_state.get("broker_override")
            or self._broker_name
        )
        if isinstance(self._orchestrator, RLStrategyOrchestrator):
            self._orchestrator.update(symbol, market_state, broker_name)
            return
        state = self._orchestrator_state.get(self._orchestrator_key(symbol, broker_name))
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
        self._orchestrator_state.pop(self._orchestrator_key(symbol, broker_name), None)

    def _record_orchestrator(
        self, symbol: str, signals: list[dict], market_state: dict, broker_name: str | None = None
    ) -> None:
        broker_name = broker_name or market_state.get("broker") or self._broker_name
        if isinstance(self._orchestrator, RLStrategyOrchestrator):
            self._orchestrator.record(symbol, signals, market_state, broker_name)
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
        self._orchestrator_state[self._orchestrator_key(symbol, broker_name)] = {
            "decisions": decisions,
            "last_price": last_price,
        }

    def _refresh_news_cache(self, symbols: list[str], now: datetime | None = None) -> None:
        news_cfg = self.cfg.get("news", {})
        if not news_cfg.get("enabled", False):
            self._news_cache = {}
            self._news_cache_at = None
            return
        now = now or datetime.now(timezone.utc)
        ttl_minutes = int(news_cfg.get("cache_minutes", 15))
        if self._news_cache_at and (now - self._news_cache_at).total_seconds() < ttl_minutes * 60:
            return
        with self._news_future_lock:
            if self._news_future is not None:
                if self._news_future.done():
                    try:
                        result = self._news_future.result()
                    except Exception as exc:
                        logging.warning("News catalyst refresh failed: %s", exc)
                    else:
                        if isinstance(result, dict):
                            self._news_cache = result
                            self._news_cache_at = now
                            logging.info("News catalyst refresh completed; symbols=%d", len(result))
                    self._news_future = None
                    self._news_inflight_at = None
                return
            symbols_snapshot = list(symbols)
            self._news_inflight_at = now
            logging.info("News catalyst refresh started; symbols=%d", len(symbols_snapshot))
            self._news_future = self._news_executor.submit(
                fetch_catalyst_symbols_for_config,
                symbols_snapshot,
                dict(news_cfg),
                dict(self.cfg.get("brokers", {})),
            )

    def _log_news_cache(self) -> None:
        news_cfg = self.cfg.get("news", {})
        if not news_cfg.get("enabled", False):
            return
        if self._news_cache_at is None:
            return
        last = self._news_cache_at
        count = len(self._news_cache)
        logging.info("News catalyst cache updated; symbols=%d", count)

    @staticmethod
    def _calc_exposure_metrics(market_state: dict, portfolio: dict, symbol: str) -> None:
        """Calculate and set exposure_pct, short_exposure_pct, leverage on market_state."""
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

    def _enrich_market_state(self, market_state: dict, portfolio: dict, symbol: str,
                              *, news_snapshot: dict | None = None,
                              orders_snapshot: list | None = None) -> None:
        self._calc_exposure_metrics(market_state, portfolio, symbol)
        market_state["portfolio"] = portfolio
        account = self._account_for_broker(portfolio.get("broker"))
        def _flag_value(key: str) -> bool:
            val = account.get(key)
            return str(val).lower() in {"true", "1", "yes"} or val == True
        market_state["account_flags"] = {
            "account_blocked": _flag_value("account_blocked"),
            "trading_blocked": _flag_value("trading_blocked"),
            "trade_suspended_by_user": _flag_value("trade_suspended_by_user"),
        }
        market_state["catalyst"] = (news_snapshot or self._news_cache).get(symbol, False)
        market_state["open_orders"] = orders_snapshot if orders_snapshot is not None else self._open_order_mgr.cache
        market_state["symbol"] = symbol
        venue = self._symbol_mgr.symbol_venue(symbol)
        if venue:
            market_state["market_venue"] = venue
            market_state["market_extended"] = is_venue_extended(self.cfg, venue)
        else:
            market_state["market_extended"] = False

    def _recalculate_exposure(self, market_state: dict, portfolio: dict, symbol: str) -> None:
        self._calc_exposure_metrics(market_state, portfolio, symbol)

    def _get_portfolio_snapshot(self) -> dict | None:
        if self.broker is None:
            self._broker_missing_reason = "broker not initialized"
            return None
        try:
            account = self.broker.get_account()
        except Exception as exc:
            self._broker_missing_reason = f"account snapshot failed: {exc}"
            return None
        self._broker_missing_at = None
        self._broker_missing_reason = None
        self._broker_missing_last_log = 0.0
        self._account_snapshot = account if isinstance(account, dict) else {}
        equity_val = 0.0
        cash_val = 0.0
        buying_power_val = 0.0
        brokers: dict[str, dict] = {}
        if isinstance(account, dict):
            if "brokers" in account and isinstance(account["brokers"], dict):
                for name, data in account["brokers"].items():
                    b_eq, b_cash, b_bp = extract_equity_cash(data)
                    brokers[name] = {
                        "equity": b_eq,
                        "cash": b_cash,
                        "buying_power": b_bp,
                        "positions": {},
                        "gross_exposure": 0.0,
                        "short_exposure": 0.0,
                    }
                equity_val = float(account.get("equity") or sum(v["equity"] for v in brokers.values()))
                cash_val = float(account.get("cash") or sum(v["cash"] for v in brokers.values()))
                buying_power_val = float(
                    account.get("buying_power") or sum(v["buying_power"] for v in brokers.values())
                )
            else:
                equity_val, cash_val, buying_power_val = extract_equity_cash(account)

        positions: dict[str, dict] = {}
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
            avg_entry = pos.get("avg_entry_price")
            if avg_entry is None:
                avg_entry = pos.get("avg_cost") or pos.get("avg_entry") or pos.get("avg_price")
            if avg_entry is not None:
                try:
                    avg_entry = float(avg_entry)
                except (TypeError, ValueError):
                    avg_entry = None
            market_value = pos.get("market_value")
            if market_value is None:
                price = pos.get("current_price") or pos.get("market_price") or pos.get("avg_cost") or 0.0
                market_value = qty * float(price)
            else:
                market_value = float(market_value)
            if symbol in positions:
                prev_qty = float(positions[symbol].get("qty", 0.0) or 0.0)
                positions[symbol]["qty"] = prev_qty + qty
                positions[symbol]["value"] = float(positions[symbol].get("value", 0.0) or 0.0) + market_value
                if avg_entry is not None:
                    prev_avg = positions[symbol].get("avg_entry")
                    weight_prev = abs(prev_qty)
                    weight_new = abs(qty)
                    total_weight = weight_prev + weight_new
                    if total_weight > 0:
                        if prev_avg is None:
                            positions[symbol]["avg_entry"] = avg_entry
                        else:
                            positions[symbol]["avg_entry"] = (
                                float(prev_avg) * weight_prev + avg_entry * weight_new
                            ) / total_weight
            else:
                positions[symbol] = {"qty": qty, "value": market_value, "avg_entry": avg_entry}
            gross_exposure += abs(market_value)
            if market_value < 0:
                short_exposure += abs(market_value)
            broker_name = pos.get("broker")
            if broker_name and broker_name in brokers:
                brokers[broker_name]["positions"][symbol] = {
                    "qty": qty,
                    "value": market_value,
                    "avg_entry": avg_entry,
                }
                brokers[broker_name]["gross_exposure"] += abs(market_value)
                if market_value < 0:
                    brokers[broker_name]["short_exposure"] += abs(market_value)

        return {
            "equity": equity_val,
            "cash": cash_val,
            "buying_power": buying_power_val,
            "positions": positions,
            "gross_exposure": gross_exposure,
            "short_exposure": short_exposure,
            "brokers": brokers,
        }

    def _kill_switch_armed(self) -> bool:
        ks_cfg = self.cfg.get("kill_switch", {}) or {}
        if not ks_cfg.get("armed", False):
            return False
        confirm = str(ks_cfg.get("confirm_code", "")).strip()
        required = str(ks_cfg.get("required_code", "")).strip()
        confirm_phrase = str(ks_cfg.get("confirm_phrase", "YES")).strip()
        if required:
            return confirm == required
        return confirm == confirm_phrase

    def _maybe_force_liquidation(self, portfolio: dict) -> None:
        ks_cfg = self.cfg.get("kill_switch", {}) or {}
        if not ks_cfg.get("force_liquidate", False):
            return
        if not self._kill_switch_armed():
            if not self._kill_switch_warned:
                logging.warning("Kill switch force_liquidate requested but interlock not armed.")
                self._kill_switch_warned = True
            return
        if self._kill_switch_liquidated:
            return
        open_orders = list(self._open_order_mgr.cache)
        if not open_orders:
            try:
                open_orders = self.broker.get_open_orders()
            except Exception as exc:
                logging.warning("Kill switch open orders fetch failed: %s", exc)
                open_orders = []
        canceled = 0
        for order in open_orders:
            order_id = order.get("order_id")
            if not order_id:
                continue
            broker_name = order.get("broker")
            try:
                if broker_name:
                    self.broker.cancel_order(str(order_id), broker=broker_name)
                else:
                    self.broker.cancel_order(str(order_id))
                canceled += 1
            except TypeError:
                try:
                    self.broker.cancel_order(str(order_id))
                    canceled += 1
                except Exception as exc:
                    logging.warning("Kill switch cancel failed for %s: %s", order_id, exc)
            except Exception as exc:
                logging.warning("Kill switch cancel failed for %s: %s", order_id, exc)

        closed = 0
        try:
            positions = self.broker.get_positions()
        except Exception as exc:
            logging.warning("Kill switch positions fetch failed: %s", exc)
            positions = []
        for pos in positions:
            symbol = pos.get("symbol")
            if not symbol:
                continue
            broker_name = pos.get("broker")
            try:
                if broker_name:
                    self.broker.close_position(symbol, broker=broker_name)
                else:
                    self.broker.close_position(symbol)
                closed += 1
            except TypeError:
                try:
                    self.broker.close_position(symbol)
                    closed += 1
                except Exception as exc:
                    logging.warning("Kill switch close failed for %s: %s", symbol, exc)
            except Exception as exc:
                logging.warning("Kill switch close failed for %s: %s", symbol, exc)

        for broker_state in self._broker_states.values():
            broker_state.disabled_strategies = set(self._strategy_names)
        self._kill_switch_liquidated = True
        logging.critical(
            "Kill switch liquidation executed; positions=%d orders=%d",
            closed,
            canceled,
        )
        _slog.event("warning", "kill_switch_liquidation", positions_closed=closed, orders_canceled=canceled)

    def _reserved_cash(self, symbol: str, last_price: float, broker: str | None = None) -> float:
        reserved = 0.0
        for order in self._open_order_mgr.cache:
            if order.get("symbol") != symbol:
                continue
            if broker and order.get("broker") != broker:
                continue
            side = (order.get("side") or "").lower()
            if side != "buy":
                continue
            qty = float(order.get("qty") or 0.0)
            limit_price = order.get("limit_price")
            price = float(limit_price) if limit_price else float(last_price)
            reserved += qty * price
        return reserved


def _dt_to_str(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _dt_from_str(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None



def _realized_volatility_pct(market_state: dict) -> float:
    prices = market_state.get("prices", []) or []
    return realized_volatility_pct(prices)


def _active_model_ref(active: dict | None) -> str | None:
    if not active:
        return None
    model_path = active.get("model_path")
    model_sha = active.get("model_sha256")
    if not model_path:
        return None
    return f"{model_path}:{model_sha or ''}"
