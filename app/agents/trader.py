# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import json
import hashlib
import logging
import os
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait

from app.execution.executor import ExecutionEngine
from app.execution.order_queue import OrderQueue
from app.execution.algos import pov_slices, twap_slices, vwap_slices
from app.execution.impact import estimate_market_impact
from app.execution import routing as routing_utils
from app.monitoring.metrics import (
    TRADES,
    SKIPPED_ORDERS,
    PNL,
    DRAWDOWN,
    ACCOUNT_TOTAL,
    ACCOUNT_CASH,
    ACCOUNT_BUYING_POWER,
    ACCOUNT_INVESTED,
    ACCOUNT_TOTAL_BY_BROKER,
    ACCOUNT_CASH_BY_BROKER,
    ACCOUNT_BUYING_POWER_BY_BROKER,
    ACCOUNT_INVESTED_BY_BROKER,
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
    SYMBOL_ACTIVE,
    SYMBOL_ACTIVE_BY_BROKER,
    STRATEGY_ACTIVE,
    ORCHESTRATOR_STRATEGY_ACTIVE,
    ORCHESTRATOR_STRATEGY_SELECTED,
    STRATEGY_TRADES_REALIZED,
    STRATEGY_WIN_RATE,
    STRATEGY_AVG_PNL_PCT,
    STRATEGY_DRAWDOWN_PCT,
    STRATEGY_DISABLED,
    OPEN_ORDERS,
    OPEN_ORDERS_BY_BROKER,
    BROKER_ACTIVE,
    BROKER_MARKET_OPEN,
    DECISION_LATENCY,
    ORDER_LATENCY,
    SKIPPED_ORDERS_BY_BROKER,
    TRADES_BY_BROKER,
    PNL_BY_BROKER,
    DRAWDOWN_BY_BROKER,
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
from app.data.scanner import ScanFilters, filter_universe_by_price, load_symbol_venues, load_universe, scan_symbols
from app.data.market_cache import build_market_cache, build_market_cache_config, interval_to_seconds
from app.learning.drift import DriftMonitor
from app.learning.registry import load_active_model, load_latest_feature_stats
from app.learning.live_rewards import LiveRewardTracker
from app.brokers.config_utils import get_alpaca_account_cfg
from app.utils.checkpoint import load_checkpoint, maybe_save_checkpoint
from app.utils.ops_state import load_ops_state, ops_state_is_running, ops_state_is_sleeping
from app.utils.gpu_state import is_gpu_disabled, disable_gpu_until_restart # Import GPU state utilities
import tensorflow as tf # Import tensorflow for GPU error handling
try:
    import torch
    _TORCH_OOM = (torch.cuda.OutOfMemoryError,)
except Exception:
    _TORCH_OOM = ()
_OOM_ERRORS = (tf.errors.ResourceExhaustedError,) + _TORCH_OOM
try:
    from app.data.ai_filter import score_symbols
except Exception:
    score_symbols = None
from app.strategies.rl_policy import RLPolicyStrategy
from app.strategies.rl_policy_fees import FeeAwareRLPolicyStrategy
from app.utils.market import is_market_open, is_venue_extended, is_venue_open
from app.utils.restart import should_restart
from app.agents.orchestrator import RLStrategyOrchestrator
from app.agents.pipeline import DecisionPipeline


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
    def __init__(self, broker, cfg: dict):
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
        self._account_snapshot: dict[str, object] = {}
        params = cfg["strategy"]["params"]
        self._strategy_params = params
        self._strategy_by_symbol: dict[tuple[str, str], dict[str, object]] = {}
        self._guardrail_by_symbol: dict[tuple[str, str], object] = {}
        self.executor = ExecutionEngine(broker)
        self._routing_cfg = cfg.get("execution", {}).get("brokers", {}).get("routing", {})
        self._broker_map = self._resolve_broker_map()
        self._broker_names = list(self._broker_map.keys())
        self._broker_name = self._resolve_default_broker_name()
        self._broker_name = routing_utils.normalize_broker_name(self._broker_name, self._broker_names)
        retry_cfg = cfg.get("execution", {}).get("retry", {})
        open_orders_cfg = cfg.get("execution", {}).get("open_orders", {})
        completion_grace = int(open_orders_cfg.get("missing_grace_seconds", 0))
        self._order_queues = {
            name: OrderQueue(item, name, retry_cfg, completion_grace_seconds=completion_grace)
            for name, item in self._broker_map.items()
        }
        self._order_queue = self._order_queues.get(self._broker_name)
        self._broker_states: dict[str, BrokerState] = {
            name: BrokerState(risk=RiskManager(cfg["risk"])) for name in self._broker_map.keys()
        }
        self._last_market_open = None
        self._started_at = datetime.utcnow()
        self._news_cache: dict[str, bool] = {}
        self._news_cache_at: datetime | None = None
        self._news_executor = ThreadPoolExecutor(max_workers=1)
        self._news_future = None
        self._news_future_lock = threading.Lock()
        self._news_inflight_at: datetime | None = None
        self._strategy_names = self._resolve_strategy_names()
        self._combine_mode = cfg["strategy"].get("combine", "priority")
        self._orchestrator = RLStrategyOrchestrator(cfg)
        self._open_orders_cache: list[dict] = []
        self._open_orders_at: datetime | None = None
        self._open_orders_labels: set[tuple[str, str]] = set()
        self._active_symbol_labels: set[str] = set()
        self._active_symbol_labels_by_broker: dict[str, set[str]] = {}
        self._open_orders_labels_by_broker: set[tuple[str, str, str]] = set()
        self._live_reward_tracker: LiveRewardTracker | None = None
        if self.cfg.get("learning", {}).get("live_rewards", {}).get("enabled", False):
            self._live_reward_tracker = LiveRewardTracker(self.cfg)
        self._dynamic_symbols_at: datetime | None = None
        self._dynamic_symbols: list[str] = []
        self._symbols: list[str] = []
        self._symbols_by_broker: dict[str, list[str]] = {}
        self._symbols_by_strategy: dict[str, list[str]] = {}
        self._symbol_venues: dict[str, str] = {}
        self._symbol_venues_at: datetime | None = None
        self._orchestrator_state: dict[tuple[str, str], dict[str, object]] = {}
        self._position_symbols: set[str] = set()
        self._position_symbols_by_broker: dict[str, set[str]] = {}
        self._ai_filter_last_run_at: datetime | None = None
        self._ai_filter_last_log_at: datetime | None = None
        self._ai_filter_last_count: int = 0
        self._ai_filter_last_signals: dict[str, dict[str, float]] = {}
        self._ai_filter_executor = ThreadPoolExecutor(max_workers=1)
        self._ai_filter_future = None
        self._ai_filter_future_lock = threading.Lock()
        self._ai_filter_inflight_at: datetime | None = None
        self._ai_filter_inflight_log_at: datetime | None = None
        self._market_cache_cfg = build_market_cache_config(cfg.get("market_cache", {}))
        self._market_cache = build_market_cache(cfg.get("market_cache", {}))
        self._pipeline = DecisionPipeline(self)
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
        self._equity_start: float | None = None
        self._equity_peak: float | None = None
        self._day_start_date = None
        self._day_start_equity: float | None = None
        self._current_drawdown_pct = 0.0
        self._equity_history: list[float] = []
        self._var_cvar: dict[str, float] = {}
        self._active_kill_switch_profile: str | None = None
        self._checkpoint_at: datetime | None = None
        self._kill_switch_liquidated = False
        self._kill_switch_warned = False
        self._performance_cfg = cfg.get("strategy", {}).get("performance", {})
        self._performance_enabled = bool(self._performance_cfg.get("enabled", False))
        self._performance_window_days = int(self._performance_cfg.get("window_days", 30))
        self._performance_min_trades = int(self._performance_cfg.get("min_trades", 20))
        self._performance_min_win_rate = float(self._performance_cfg.get("min_win_rate", 0.4))
        self._performance_max_drawdown = float(self._performance_cfg.get("max_drawdown_pct", 12.0))
        self._performance_report_interval = int(self._performance_cfg.get("report_interval_minutes", 10))
        self._performance_report_path = str(
            self._performance_cfg.get("report_path", "/data/reports/strategy_performance.json")
        )
        self._kill_switch_enabled = bool(self._performance_cfg.get("kill_switch", {}).get("enabled", True))
        self._performance_last_report_at: datetime | None = None
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
            "at": datetime.utcnow().isoformat(),
        }

    def _risk_disabled(self) -> bool:
        return not bool(self.cfg.get("risk", {}).get("enabled", True))

    def _build_strategy(self, name: str, params: dict):
        if name == "rl_policy":
            if self.learning_cfg.get("enabled"):
                model_path = self._select_model_path()
                window_size = int(self.learning_cfg.get("window_size", 50))
                device = self.learning_cfg.get("device", "auto")
                feature_config = self.learning_cfg.get("features", {})
                
                # Attempt to build with current device setting
                try_devices = [device]
                if device != "cpu":
                    try_devices.append("cpu")

                for current_device in try_devices:
                    try:
                        return RLPolicyStrategy(
                            model_path,
                            window_size=window_size,
                            device=current_device, # Use the current device in the loop
                            feature_config=feature_config,
                            drift_monitor=self._drift_monitor,
                            include_features=self._include_feature_snapshots,
                            risk_cfg=self.cfg.get("risk", {}),
                        )
                    except (FileNotFoundError, ValueError) as exc:
                        logging.warning("RL model unavailable, skipping rl_policy: %s", exc)
                        break # Model not found/invalid, no point in retrying with CPU
                    except _OOM_ERRORS as exc:
                        if current_device != "cpu":
                            logging.warning(
                                "CUDA out of memory during rl_policy build: %s. Falling back to CPU.", exc
                            )
                            disable_gpu_until_restart()
                            self.learning_cfg["device"] = "cpu" # Update config for current run
                            continue # Retry with CPU
                        else:
                            logging.error("RL policy failed on CPU after GPU error: %s", exc)
                            break
                    except Exception as exc:
                        logging.warning("Unknown error during rl_policy build: %s", exc)
                        break
            logging.debug("RL policy not enabled or failed to build.")
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
                
                # Attempt to build with current device setting
                try_devices = [device]
                if device != "cpu":
                    try_devices.append("cpu")

                for current_device in try_devices:
                    try:
                        return FeeAwareRLPolicyStrategy(
                            model_path,
                            window_size=window_size,
                            device=current_device, # Use the current device in the loop
                            feature_config=feature_config,
                            drift_monitor=self._drift_monitor,
                            include_features=self._include_feature_snapshots,
                            broker_fees=broker_fees,
                            fee_guard=fee_guard,
                            risk_cfg=risk_cfg,
                        )
                    except (FileNotFoundError, ValueError) as exc:
                        logging.warning("RL model unavailable, skipping rl_policy_fees: %s", exc)
                        break # Model not found/invalid, no point in retrying with CPU
                    except _OOM_ERRORS as exc:
                        if current_device != "cpu":
                            logging.warning(
                                "CUDA out of memory during rl_policy_fees build: %s. Falling back to CPU.", exc
                            )
                            disable_gpu_until_restart()
                            self.learning_cfg["device"] = "cpu" # Update config for current run
                            continue # Retry with CPU
                        else:
                            logging.error("RL policy fees failed on CPU after GPU error: %s", exc)
                            break
                    except Exception as exc:
                        logging.warning("Unknown error during rl_policy_fees build: %s", exc)
                        break
            logging.debug("RL policy fees not enabled or failed to build.")
            return None
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
        logging.debug("Unknown strategy '%s' requested, returning None.", name)
        return IntradayMomentumStrategy(
            params["lookback_minutes"],
            params["entry_threshold_pct"],
            params["exit_threshold_pct"],
            params["allow_shorts"],
        )

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
        now = datetime.utcnow()
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
        algo_cfg = self.cfg.get("execution", {}).get("algos", {})
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
            impact_cfg = self.cfg.get("execution", {}).get("impact", {}) or {}
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
            "ts": datetime.utcnow().isoformat(),
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
        for key in ("score", "confidence", "strength", "reason", "weight", "signal_bias", "signal_bias_block"):
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
            date_str = datetime.utcnow().date().isoformat()
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
            exec_cfg = self.cfg.get("execution", {}).get("open_orders", {})
            broker_hint = self._resolve_broker_for_symbol(symbol, self._strategy_names, None)
            broker_override = market_state.get("broker_override")
            if broker_override:
                broker_override = self._normalize_broker_name(str(broker_override))
            strategy_broker = broker_override or broker_hint
            market_state["risk_outcome"] = self._broker_state(strategy_broker).risk_outcomes.get(symbol, {})
            if trace:
                trace["broker_hint"] = broker_hint
            if exec_cfg.get("strategy_guard", False) and self._has_pending_order(symbol, broker=broker_hint):
                self._record_skip(symbol, "hold", "open_order", broker_hint)
                logging.info("Skipping %s: open orders pending (strategy guard)", symbol)
                self._emit_decision_trace(trace, "skip", "open_order", "strategy_guard")
                return None
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
            except Exception as exc:
                logging.warning("Strategy %s failed for %s: %s", name, symbol, exc)
                continue
            signal["name"] = name
            signals.append(signal)

        # UNLOCKED: Signal bias calculation (pure logic)
        bias = self._signal_bias(market_state)
        guard_cfg = self.cfg.get("strategy", {}).get("signal_bias_guard", {}) or {}
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
            if not risk_disabled and broker_state.risk.should_circuit_break(broker_state.current_drawdown_pct):
                self._record_skip(symbol, "hold", "circuit_breaker", broker_name)
                logging.warning("Skipping %s: circuit breaker drawdown hit", symbol)
                self._emit_decision_trace(trace, "skip", "circuit_breaker", "risk")
                return None
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
                    self._remove_pending_orders(symbol, broker=broker_name)
                self._emit_decision_trace(trace, "hold", "strategies_hold", "signal")
                return None
            if action == "exit":
                if self._cancel_pending_if_needed(symbol, action, broker=broker_name):
                    self._remove_pending_orders(symbol, broker=broker_name)
                self.broker.close_position(symbol, broker=broker_name)
                self._emit_decision_trace(trace, "exit", "strategy_exit", "signal")
                return None

            if self._has_pending_order(symbol, broker=broker_name):
                if self._cancel_pending_if_needed(symbol, action, broker=broker_name):
                    self._remove_pending_orders(symbol, broker=broker_name)
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
            if min_price is not None and last_price < float(min_price):
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
            if self._is_account_blocked(broker_name):
                self._record_risk_outcome(symbol, action, False, "account_blocked", broker_name)
                self._record_skip(symbol, action, "account_blocked", broker_name)
                logging.info("Skipping %s for %s: account blocked", action, symbol)
                self._emit_decision_trace(trace, "skip", "account_blocked", "account")
                return None
            var_reason = self._var_limit_reason(broker_name)
            if not risk_disabled and var_reason:
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
            if not risk_disabled and self._violates_exposure_caps(symbol, action, qty, last_price, portfolio):
                self._record_risk_outcome(symbol, action, False, "exposure_cap", broker_name)
                self._record_skip(symbol, action, "exposure_cap", broker_name)
                logging.info("Skipping %s for %s: exposure caps exceeded", action, symbol)
                self._emit_decision_trace(trace, "skip", "exposure_cap", "risk")
                return None
            if not risk_disabled and self._violates_order_limits(symbol, action, qty, last_price):
                self._record_risk_outcome(symbol, action, False, "order_limit", broker_name)
                self._record_skip(symbol, action, "order_limit", broker_name)
                logging.info("Skipping %s for %s: order limits", action, symbol)
                self._emit_decision_trace(trace, "skip", "order_limit", "limits")
                return None
            market_state["qty"] = qty
            if trace:
                trace["qty"] = qty

            now = datetime.utcnow()
            if not risk_disabled and self._is_cooldown_active(market_state, now, broker_name):
                self._record_risk_outcome(symbol, action, False, "cooldown", broker_name)
                self._record_skip(symbol, action, "cooldown", broker_name)
                logging.info("Skipping %s for %s: cooldown", action, symbol)
                self._emit_decision_trace(trace, "skip", "cooldown", "cooldown")
                return None
            if not risk_disabled:
                if not broker_state.risk.can_open_trade(
                    exposure_pct=market_state.get("exposure_pct", 0.0),
                    short_exposure_pct=market_state.get("short_exposure_pct", 0.0),
                    leverage=market_state.get("leverage", 1.0),
                ):
                    self._record_risk_outcome(symbol, action, False, "risk_block", broker_name)
                    self._record_skip(symbol, action, "risk_block", broker_name)
                    logging.info("Skipping %s for %s: risk limits exceeded", action, symbol)
                    self._emit_decision_trace(trace, "skip", "risk_block", "risk")
                    return None
                self._record_risk_outcome(symbol, action, True, "ok", broker_name)
            else:
                self._record_risk_outcome(symbol, action, True, "risk_disabled", broker_name)

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
        pending = self._pending_orders(symbol, broker=broker)
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
                except Exception as exc:
                    logging.warning("Orchestrator order feedback failed: %s", exc)
                if self._live_reward_tracker is not None:
                    try:
                        reward_event = self._live_reward_tracker.on_order_response(response.__dict__)
                        if reward_event and isinstance(self._orchestrator, RLStrategyOrchestrator):
                            try:
                                self._orchestrator.on_trade_reward(reward_event)
                            except Exception as exc:
                                logging.warning("Orchestrator reward update failed: %s", exc)
                    except Exception as exc:
                        logging.warning("Live reward update failed: %s", exc)

    def _pending_orders(self, symbol: str, broker: str | None = None) -> list[dict]:
        pending = [order for order in self._open_orders_cache if order.get("symbol") == symbol]
        if broker:
            pending = [order for order in pending if order.get("broker") == broker]
        return pending

    def _remove_pending_orders(self, symbol: str, broker: str | None = None) -> None:
        if broker:
            self._open_orders_cache = [
                order
                for order in self._open_orders_cache
                if order.get("symbol") != symbol or order.get("broker") != broker
            ]
            return
        self._open_orders_cache = [order for order in self._open_orders_cache if order.get("symbol") != symbol]

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
        max_pos_pct *= self._vol_target_scale(market_state)
        max_short_pct = float(self.cfg["risk"]["max_short_exposure_pct"])
        max_short_pct *= self._vol_target_scale(market_state)
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
        if len(prices) < 3:
            return 1.0
        returns = []
        for idx in range(1, len(prices)):
            prev = prices[idx - 1]
            curr = prices[idx]
            if not prev:
                continue
            returns.append((curr - prev) / prev)
        if not returns:
            return 1.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / max(len(returns) - 1, 1)
        realized = (var ** 0.5) * 100.0
        target = float(cfg.get("target_vol_pct", 2.0))
        if realized <= 0:
            return 1.0
        scale = target / realized
        min_scale = float(cfg.get("min_scale", 0.5))
        max_scale = float(cfg.get("max_scale", 1.5))
        return max(min(scale, max_scale), min_scale)

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

    def _violates_order_limits(self, symbol: str, action: str, qty: int, last_price: float) -> bool:
        if not self._limits_enabled():
            return False
        limits = self.cfg.get("trading_limits", {})
        max_qty = limits.get("max_order_qty")
        if max_qty is not None and qty > float(max_qty):
            return True
        notional = qty * last_price
        max_notional = limits.get("max_order_notional")
        if max_notional is not None and notional > float(max_notional):
            return True
        min_notional = limits.get("min_order_notional")
        if min_notional is not None and notional < float(min_notional):
            return True
        return False

    def _is_cooldown_active(
        self, market_state: dict, now: datetime, broker_name: str | None = None
    ) -> bool:
        cfg = self.cfg.get("risk", {})
        cooldown = int(cfg.get("cooldown_seconds", 0))
        broker_state = self._broker_state(broker_name)
        if cooldown <= 0 or not broker_state.last_trade_at:
            return False
        elapsed = (now - broker_state.last_trade_at).total_seconds()
        return elapsed < cooldown

    def _resolve_strategy_names(self) -> list[str]:
        cfg = self.cfg.get("strategy", {})
        names = cfg.get("names")
        if isinstance(names, list) and names:
            return [str(name) for name in names]
        name = cfg.get("name", "intraday_momentum")
        return [str(name)]

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

    def _update_active_symbol_metrics(self, symbols: list[str]) -> None:
        current_symbols = set(symbols)
        # Set active symbols to 1 first to avoid flipping between 0 and 1
        for sym in symbols:
            SYMBOL_ACTIVE.labels(symbol=sym).set(1)
        # Then clear symbols that are no longer active
        for sym in self._active_symbol_labels - current_symbols:
            SYMBOL_ACTIVE.labels(symbol=sym).set(0)
        self._active_symbol_labels = current_symbols
        symbols_by_broker: dict[str, set[str]] = {}
        if self._symbols_by_broker:
            for broker_name, batch in self._symbols_by_broker.items():
                symbols_by_broker[broker_name] = set(batch)
        else:
            routing_mode = str(self._routing_cfg.get("mode", "default")).lower()
            broker_names = list(self._broker_map.keys())
            if routing_mode == "parallel" and len(broker_names) > 1:
                broker_buying_power = self._get_broker_buying_power()
                buckets = routing_utils.parallel_partition_symbols(
                    symbols, broker_names, broker_buying_power, self._routing_cfg, min_symbols=1
                )
                for broker_name, batch in buckets.items():
                    symbols_by_broker[broker_name] = set(batch)
            elif routing_mode == "auto_split" and len(broker_names) > 1:
                buckets = routing_utils.partition_symbols(symbols, broker_names, self._routing_cfg)
                for broker_name, batch in buckets.items():
                    symbols_by_broker[broker_name] = set(batch)
        if symbols_by_broker:
            for broker_name, active_syms in symbols_by_broker.items():
                previous = self._active_symbol_labels_by_broker.get(broker_name, set())
                # Set active symbols to 1 first to avoid flipping between 0 and 1
                for sym in active_syms:
                    SYMBOL_ACTIVE_BY_BROKER.labels(broker=broker_name, symbol=sym).set(1)
                # Then clear symbols that are no longer active
                for sym in previous - active_syms:
                    SYMBOL_ACTIVE_BY_BROKER.labels(broker=broker_name, symbol=sym).set(0)
                self._active_symbol_labels_by_broker[broker_name] = set(active_syms)

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
            self._symbols_by_broker = buckets
            return [
                ("parallel", broker, batch)
                for broker, batch in buckets.items()
                if batch
            ]

        if routing_mode == "auto_split" and len(broker_names) > 1:
            if self._symbols_by_broker:
                return [
                    ("auto_split", broker, batch)
                    for broker, batch in self._symbols_by_broker.items()
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
            for order in self._open_orders_cache:
                broker_name = order.get("broker") or self._broker_name
                orders_by_broker.setdefault(str(broker_name), []).append(order)
            for broker_name, queue in self._order_queues.items():
                queue.update(orders_by_broker.get(broker_name, []))
        else:
            if self._order_queue:
                self._order_queue.update(self._open_orders_cache)

    def _update_market_open_metrics(self, market_open: bool) -> None:
        with self._lock:
            brokers_cfg = self.cfg.get("brokers", {})
            broker_names = [
                name
                for name, cfg in brokers_cfg.items()
                if not isinstance(cfg, dict) or cfg.get("enabled", True)
            ]
            if not broker_names:
                broker_names = self._broker_names or [self._broker_name]
            for name in broker_names:
                BROKER_MARKET_OPEN.labels(broker=name).set(1 if market_open else 0)
            if market_open != self._last_market_open:
                state = "open" if market_open else "closed"
                logging.info("Market is %s; %s trading loop.", state, "starting" if market_open else "waiting")
                self._last_market_open = market_open

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
        self._update_account_metrics()
        self._update_position_metrics(portfolio)
        if self._performance_enabled:
            brokers = portfolio.get("brokers", {})
            if isinstance(brokers, dict) and brokers:
                for broker_name in brokers.keys():
                    broker_portfolio = self._portfolio_for_broker(portfolio, broker_name)
                    self._update_performance_from_positions(broker_portfolio, broker_name)
            else:
                self._update_performance_from_positions(portfolio, self._broker_name)
        self._maybe_report_performance()
        self._refresh_news_cache(symbols)
        self._log_news_cache()
        self._refresh_dynamic_symbols(portfolio)
        self._refresh_symbol_venues()
        self._maybe_checkpoint()
        self._log_ai_filter_heartbeat()
        self._maybe_reload_active_model()
        symbols = self._resolve_active_symbols()
        symbols = self._merge_symbols_with_positions(symbols, portfolio)
        self._update_active_symbol_metrics(symbols)
        self._refresh_open_orders_cache(symbols)
        self._maybe_force_liquidation(portfolio)
        return symbols

    def _process_single_symbol(
        self,
        sym: str,
        portfolio: dict,
        market_data_provider,
        broker_override: str | None,
        skip_unchanged: bool,
    ) -> None:
        if not self._is_symbol_market_open(sym):
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

            self._enrich_market_state(market_state, portfolio, sym)
            
            if broker_override:
                market_state["broker_override"] = broker_override
            
            # Access strategy symbols with lock
            with self._lock:
                market_state["strategy_symbols"] = self._symbols_by_strategy
            
            self._update_signal_metrics(sym, market_state)
            
            decision_start = time.perf_counter()
            market_state["_decision_start"] = decision_start
            
            self._pipeline.run(sym, market_state)
            
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
            
        # Use ThreadPoolExecutor for parallel processing
        # Limit max_workers to avoid API rate limits (e.g. 8-16)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    self._process_single_symbol,
                    sym,
                    batch_portfolio,
                    market_data_provider,
                    broker_override,
                    skip_unchanged,
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
            weight = float(weights.get(name, 1.0))
            if top_weight is None or weight > top_weight:
                top_weight = weight
                top_signal = signal
            if action == "buy":
                buy_score += weight
                if top_buy_weight is None or weight > top_buy_weight:
                    top_buy_weight = weight
                    top_buy = signal
            elif action == "sell":
                sell_score += weight
                sells.append(signal)
                if top_sell_weight is None or weight > top_sell_weight:
                    top_sell_weight = weight
                    top_sell = signal
        if buy_score == sell_score:
            return "hold", 1.0, top_signal.get("name") if top_signal else None
        if buy_score > sell_score:
            chosen = top_buy or top_signal
            return "buy", 1.0, chosen.get("name") if chosen else None
        reduce_pct = max(float(s.get("reduce_pct", 1.0)) for s in sells) if sells else 1.0
        chosen = top_sell or top_signal
        strategy = chosen.get("name") if chosen else None
        return "sell", reduce_pct, strategy

    def _update_broker_equity_state(self, broker_name: str, equity: float, last_equity: float | None = None) -> None:
        broker_state = self._broker_state(broker_name)
        if broker_state.equity_start is None:
            broker_state.equity_start = equity
        if broker_state.equity_peak is None or equity > broker_state.equity_peak:
            broker_state.equity_peak = equity
        base_equity = last_equity if last_equity else broker_state.equity_start
        if base_equity:
            pnl_pct = (equity - base_equity) / base_equity * 100.0
            PNL_BY_BROKER.labels(broker=broker_name).set(pnl_pct)
        if broker_state.equity_peak:
            drawdown_pct = (broker_state.equity_peak - equity) / broker_state.equity_peak * 100.0
            broker_state.current_drawdown_pct = max(drawdown_pct, 0.0)
            DRAWDOWN_BY_BROKER.labels(broker=broker_name).set(max(drawdown_pct, 0.0))
        today = datetime.utcnow().date()
        if broker_state.day_start_date != today or broker_state.day_start_equity is None:
            broker_state.day_start_date = today
            broker_state.day_start_equity = equity
        if broker_state.day_start_equity:
            day_pnl_pct = (equity - broker_state.day_start_equity) / broker_state.day_start_equity * 100.0
            broker_state.risk.update_daily_loss(day_pnl_pct)

    def _update_account_metrics(self) -> None:
        try:
            account = self.broker.get_account()
        except Exception as exc:
            logging.warning("Account metrics update failed: %s", exc)
            return
        with self._lock:
            total_val = cash_val = buying_power_val = None
            broker_equities: dict[str, float] = {}
            if isinstance(account, dict):
                if "brokers" in account and isinstance(account["brokers"], dict):
                    total_val = float(account.get("equity") or 0.0)
                    cash_val = float(account.get("cash") or 0.0)
                    buying_power_val = float(account.get("buying_power") or 0.0)
                    for name, details in account["brokers"].items():
                        equity = float(details.get("equity") or 0.0)
                        cash = float(details.get("cash") or 0.0)
                        buying_power = float(details.get("buying_power") or 0.0)
                        broker_last_equity = None
                        if details.get("last_equity") is not None:
                            try:
                                broker_last_equity = float(details["last_equity"])
                            except (TypeError, ValueError):
                                broker_last_equity = None
                        ACCOUNT_TOTAL_BY_BROKER.labels(broker=name).set(equity)
                        ACCOUNT_CASH_BY_BROKER.labels(broker=name).set(cash)
                        ACCOUNT_BUYING_POWER_BY_BROKER.labels(broker=name).set(buying_power)
                        ACCOUNT_INVESTED_BY_BROKER.labels(broker=name).set(equity - cash)
                        broker_equities[str(name)] = equity
                        self._update_broker_equity_state(str(name), equity, last_equity=broker_last_equity)
                        # Update buying power in broker state for parallel routing
                        broker_state = self._broker_states.get(str(name))
                        if broker_state:
                            broker_state.buying_power = buying_power
                elif "equity" in account:
                    total_val = float(account.get("equity") or 0.0)
                    cash_val = float(account.get("cash") or 0.0)
                    buying_power_val = float(account.get("buying_power") or 0.0)
                elif "NetLiquidation" in account:
                    total_val = float(account.get("NetLiquidation") or 0.0)
                    cash_val = float(account.get("TotalCashValue") or 0.0)
                    buying_power_val = float(account.get("BuyingPower") or account.get("AvailableFunds") or 0.0)
            if total_val is None or cash_val is None:
                return
            if not broker_equities:
                broker_equities[self._broker_name] = total_val
                self._update_broker_equity_state(self._broker_name, total_val)
            if self._equity_start is None:
                self._equity_start = total_val
            if self._equity_peak is None or total_val > self._equity_peak:
                self._equity_peak = total_val
            last_equity = None
            if isinstance(account, dict):
                last_equity = account.get("last_equity")
            base_equity = self._equity_start
            if last_equity is not None:
                try:
                    last_equity_val = float(last_equity)
                except (TypeError, ValueError):
                    last_equity_val = None
                if last_equity_val:
                    base_equity = last_equity_val
            if base_equity:
                pnl_pct = (total_val - base_equity) / base_equity * 100.0
                PNL.set(pnl_pct)
            if self._equity_peak:
                drawdown_pct = (self._equity_peak - total_val) / self._equity_peak * 100.0
                DRAWDOWN.set(max(drawdown_pct, 0.0))
                self._current_drawdown_pct = max(drawdown_pct, 0.0)
            ACCOUNT_TOTAL.set(total_val)
            ACCOUNT_CASH.set(cash_val)
            if buying_power_val is not None:
                ACCOUNT_BUYING_POWER.set(buying_power_val)
            ACCOUNT_INVESTED.set(total_val - cash_val)
            today = datetime.utcnow().date()
            if self._day_start_date != today or self._day_start_equity is None:
                self._day_start_date = today
                self._day_start_equity = total_val
            if self._day_start_equity:
                day_pnl_pct = (total_val - self._day_start_equity) / self._day_start_equity * 100.0
                self._update_drift_monitor(day_pnl_pct)
            self._update_var_cvar(total_val, broker_equities)

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

    def _update_var_cvar(self, total_val: float, broker_equities: dict[str, float] | None) -> None:
        var_cfg = self.cfg.get("risk", {}).get("var", {}) or {}
        if not var_cfg.get("enabled", False):
            return
        window = int(var_cfg.get("window", 60))
        confidence = float(var_cfg.get("confidence", 0.95))
        self._equity_history.append(total_val)
        if len(self._equity_history) > window:
            self._equity_history = self._equity_history[-window:]
        self._var_cvar = _var_cvar_from_history(self._equity_history, confidence)
        broker_equities = broker_equities or {}
        for name, equity in broker_equities.items():
            broker_state = self._broker_state(name)
            broker_state.equity_history.append(float(equity))
            if len(broker_state.equity_history) > window:
                broker_state.equity_history = broker_state.equity_history[-window:]
            broker_state.var_cvar = _var_cvar_from_history(broker_state.equity_history, confidence)

    def _var_limit_reason(self, broker_name: str) -> str | None:
        var_cfg = self.cfg.get("risk", {}).get("var", {}) or {}
        if not var_cfg.get("enabled", False):
            return None
        max_var = float(var_cfg.get("max_var_pct", 0.0) or 0.0)
        max_cvar = float(var_cfg.get("max_cvar_pct", 0.0) or 0.0)
        if max_var and self._var_cvar.get("var_pct", 0.0) > max_var:
            return "var_limit"
        if max_cvar and self._var_cvar.get("cvar_pct", 0.0) > max_cvar:
            return "cvar_limit"
        broker_stats = self._broker_state(broker_name).var_cvar
        if max_var and broker_stats.get("var_pct", 0.0) > max_var:
            return "var_limit_broker"
        if max_cvar and broker_stats.get("cvar_pct", 0.0) > max_cvar:
            return "cvar_limit_broker"
        return None

    def _violates_exposure_caps(self, symbol: str, action: str, qty: int, last_price: float, portfolio: dict) -> bool:
        caps_cfg = self.cfg.get("risk", {}).get("exposure_caps", {}) or {}
        if not caps_cfg.get("enabled", False):
            return False
        if action not in ("buy", "sell"):
            return False
        delta = qty * last_price
        positions = portfolio.get("positions", {})
        current_qty = float(positions.get(symbol, {}).get("qty", 0.0) or 0.0)
        if action == "sell" and current_qty > 0:
            delta = -min(delta, current_qty * last_price)
        venue_caps = caps_cfg.get("venues", {}) or {}
        sector_caps = caps_cfg.get("sectors", {}) or {}
        equity = float(portfolio.get("equity", 0.0) or 0.0)
        if equity <= 0:
            return False
        if venue_caps:
            exposure = _group_exposure(portfolio, self._symbol_venue)
            venue = self._symbol_venue(symbol)
            if venue:
                exposure[venue] = exposure.get(venue, 0.0) + abs(delta)
            for venue_name, cap in venue_caps.items():
                if exposure.get(venue_name, 0.0) / equity * 100.0 > float(cap):
                    return True
        if sector_caps:
            exposure = _group_exposure(portfolio, self._symbol_sector)
            sector = self._symbol_sector(symbol)
            if sector:
                exposure[sector] = exposure.get(sector, 0.0) + abs(delta)
            for sector_name, cap in sector_caps.items():
                if exposure.get(sector_name, 0.0) / equity * 100.0 > float(cap):
                    return True
        return False

    def _symbol_venue(self, symbol: str) -> str | None:
        venue = self._symbol_venues.get(symbol)
        if venue:
            return venue
        market_cfg = self.cfg.get("market", {})
        return market_cfg.get("default_symbol_venue") or market_cfg.get("venue")

    def _symbol_sector(self, symbol: str) -> str | None:
        market_cfg = self.cfg.get("market", {})
        sector_map = market_cfg.get("symbol_sectors", {}) or {}
        return sector_map.get(symbol)

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

    def _prune_pending_entry_strategies(self, now: datetime, broker_name: str) -> None:
        broker_state = self._broker_state(broker_name)
        if not broker_state.pending_entry_strategy:
            return
        cutoff = now - timedelta(days=1)
        stale = [
            symbol
            for symbol, data in broker_state.pending_entry_strategy.items()
            if isinstance(data, dict) and data.get("ts") and data["ts"] < cutoff
        ]
        for symbol in stale:
            broker_state.pending_entry_strategy.pop(symbol, None)

    def _consume_pending_entry_strategy(self, symbol: str, broker_name: str) -> str | None:
        broker_state = self._broker_state(broker_name)
        data = broker_state.pending_entry_strategy.pop(symbol, None)
        if not data:
            return None
        strategy = data.get("strategy")
        if strategy and strategy in self._strategy_names:
            return str(strategy)
        return None

    def _record_trade(
        self, strategy: str | None, symbol: str, pnl_pct: float, ts: datetime, broker_name: str
    ) -> None:
        broker_state = self._broker_state(broker_name)
        record = {"ts": ts, "pnl_pct": pnl_pct}
        broker_state.symbol_trades.setdefault(symbol, []).append(record)
        if strategy and strategy in self._strategy_names:
            broker_state.strategy_trades.setdefault(strategy, []).append(record)
            STRATEGY_TRADES_REALIZED.labels(strategy=strategy).inc()

    def _prune_trade_records(self, records: list[dict], cutoff: datetime) -> list[dict]:
        if not records:
            return []
        return [record for record in records if record.get("ts") and record["ts"] >= cutoff]

    def _compute_trade_stats(self, records: list[dict]) -> dict:
        if not records:
            return {"trades": 0, "win_rate": 0.0, "avg_pnl_pct": 0.0, "drawdown_pct": 0.0}
        ordered = sorted(records, key=lambda r: r.get("ts") or datetime.min)
        total = len(ordered)
        wins = sum(1 for r in ordered if float(r.get("pnl_pct", 0.0) or 0.0) > 0.0)
        avg_pnl = sum(float(r.get("pnl_pct", 0.0) or 0.0) for r in ordered) / total
        equity = 100.0
        peak = 100.0
        max_dd = 0.0
        for record in ordered:
            pnl_pct = float(record.get("pnl_pct", 0.0) or 0.0)
            equity *= 1.0 + pnl_pct / 100.0
            if equity > peak:
                peak = equity
            if peak > 0:
                drawdown = (peak - equity) / peak * 100.0
                if drawdown > max_dd:
                    max_dd = drawdown
        return {
            "trades": total,
            "win_rate": wins / total,
            "avg_pnl_pct": avg_pnl,
            "drawdown_pct": max_dd,
        }

    def _update_performance_from_positions(self, portfolio: dict, broker_name: str) -> None:
        if not self._performance_enabled:
            return
        broker_state = self._broker_state(broker_name)
        now = datetime.utcnow()
        self._prune_pending_entry_strategies(now, broker_name)
        positions = portfolio.get("positions", {}) or {}
        symbols = set(positions.keys()) | set(broker_state.position_state.keys())
        for symbol in symbols:
            current = positions.get(symbol) or {}
            curr_qty = float(current.get("qty", 0.0) or 0.0)
            curr_avg_entry = current.get("avg_entry")
            if curr_avg_entry is not None:
                try:
                    curr_avg_entry = float(curr_avg_entry)
                except (TypeError, ValueError):
                    curr_avg_entry = None
            prev = broker_state.position_state.get(symbol)
            if prev is None:
                if curr_qty != 0:
                    strategy = self._consume_pending_entry_strategy(symbol, broker_name)
                    broker_state.position_state[symbol] = {
                        "qty": curr_qty,
                        "avg_entry": curr_avg_entry,
                        "strategy": strategy,
                    }
                continue
            prev_qty = float(prev.get("qty", 0.0) or 0.0)
            prev_avg = prev.get("avg_entry")
            strategy = prev.get("strategy")
            if curr_qty > prev_qty:
                if strategy is None:
                    strategy = self._consume_pending_entry_strategy(symbol, broker_name)
                prev["qty"] = curr_qty
                if curr_avg_entry is not None:
                    prev["avg_entry"] = curr_avg_entry
                if strategy is not None:
                    prev["strategy"] = strategy
                continue
            if curr_qty < prev_qty:
                exit_price = broker_state.last_prices.get(symbol)
                entry_price = prev_avg or curr_avg_entry
                if entry_price and exit_price:
                    direction = 1.0 if prev_qty > 0 else -1.0
                    pnl_pct = (exit_price - entry_price) / entry_price * 100.0 * direction
                    self._record_trade(strategy, symbol, pnl_pct, now, broker_name)
                if curr_qty == 0:
                    broker_state.position_state.pop(symbol, None)
                else:
                    prev["qty"] = curr_qty
                    if curr_avg_entry is not None:
                        prev["avg_entry"] = curr_avg_entry
                continue

    def _maybe_report_performance(self) -> None:
        if not self._performance_enabled:
            return
        now = datetime.utcnow()
        interval_seconds = self._performance_report_interval * 60
        if self._performance_last_report_at and (now - self._performance_last_report_at).total_seconds() < interval_seconds:
            return
        cutoff = now - timedelta(days=self._performance_window_days)
        strategy_report: dict[str, dict] = {}
        symbol_report: dict[str, dict] = {}
        brokers_report: dict[str, dict] = {}
        aggregate_strategy_records: dict[str, list[dict]] = {name: [] for name in self._strategy_names}
        aggregate_symbol_records: dict[str, list[dict]] = {}

        for broker_name, broker_state in self._broker_states.items():
            broker_strategy_report: dict[str, dict] = {}
            broker_symbol_report: dict[str, dict] = {}
            for name in self._strategy_names:
                records = self._prune_trade_records(broker_state.strategy_trades.get(name, []), cutoff)
                broker_state.strategy_trades[name] = records
                stats = self._compute_trade_stats(records)
                broker_strategy_report[name] = stats | {"disabled": name in broker_state.disabled_strategies}
                aggregate_strategy_records[name].extend(records)
                if (
                    self._kill_switch_enabled
                    and name not in broker_state.disabled_strategies
                    and stats["trades"] >= self._performance_min_trades
                    and (
                        stats["win_rate"] < self._performance_min_win_rate
                        or stats["drawdown_pct"] > self._performance_max_drawdown
                    )
                ):
                    broker_state.disabled_strategies.add(name)
                    logging.warning(
                        "Strategy %s disabled by kill switch (broker=%s trades=%d win_rate=%.2f drawdown=%.2f)",
                        name,
                        broker_name,
                        stats["trades"],
                        stats["win_rate"],
                        stats["drawdown_pct"],
                    )
            for symbol, records in list(broker_state.symbol_trades.items()):
                trimmed = self._prune_trade_records(records, cutoff)
                if trimmed:
                    broker_state.symbol_trades[symbol] = trimmed
                    broker_symbol_report[symbol] = self._compute_trade_stats(trimmed)
                    aggregate_symbol_records.setdefault(symbol, []).extend(trimmed)
                else:
                    broker_state.symbol_trades.pop(symbol, None)
            brokers_report[broker_name] = {
                "strategies": broker_strategy_report,
                "symbols": broker_symbol_report,
                "disabled_strategies": sorted(broker_state.disabled_strategies),
            }
            broker_state.performance_last_report_at = now

        for name in self._strategy_names:
            records = aggregate_strategy_records.get(name, [])
            stats = self._compute_trade_stats(records)
            STRATEGY_WIN_RATE.labels(strategy=name).set(stats["win_rate"])
            STRATEGY_AVG_PNL_PCT.labels(strategy=name).set(stats["avg_pnl_pct"])
            STRATEGY_DRAWDOWN_PCT.labels(strategy=name).set(stats["drawdown_pct"])
            STRATEGY_DISABLED.labels(strategy=name).set(1 if self._strategy_disabled_globally(name) else 0)
            strategy_report[name] = stats | {"disabled": self._strategy_disabled_globally(name)}

        for symbol, records in aggregate_symbol_records.items():
            symbol_report[symbol] = self._compute_trade_stats(records)

        report = {
            "generated_at": now.isoformat(),
            "window_days": self._performance_window_days,
            "min_trades": self._performance_min_trades,
            "min_win_rate": self._performance_min_win_rate,
            "max_drawdown_pct": self._performance_max_drawdown,
            "strategies": strategy_report,
            "symbols": symbol_report,
            "disabled_strategies": sorted(
                [name for name in self._strategy_names if self._strategy_disabled_globally(name)]
            ),
            "brokers": brokers_report,
        }
        try:
            report_path = Path(self._performance_report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2))
        except Exception as exc:
            logging.warning("Performance report write failed: %s", exc)
        self._performance_last_report_at = now

    def _run_reporting_loop(self):
        logging.info("Starting reporting loop thread.")
        while True:
            try:
                # Update account metrics (includes PnL, Equity, Drift)
                self._update_account_metrics()
                market_open = is_market_open(self.cfg)
                self._update_market_open_metrics(market_open)
            except Exception as exc:
                logging.warning("Reporting loop error: %s", exc)
            time.sleep(15)

    def loop(self, symbol: str | list[str], market_data_provider, interval_seconds: int = 60):
        self._symbols = symbol if isinstance(symbol, list) else [symbol]
        
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
            symbols = self._resolve_active_symbols()
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

    def _is_symbol_market_open(self, symbol: str) -> bool:
        market_cfg = self.cfg.get("market", {})
        venue_map = market_cfg.get("symbol_venues", {}) or {}
        default_venue = str(market_cfg.get("default_symbol_venue", "")).strip()
        venue = str(venue_map.get(symbol) or self._symbol_venues.get(symbol) or default_venue).strip()
        if not venue:
            return is_market_open(self.cfg)
        return is_venue_open(self.cfg, venue)

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
            self._broker_missing_at = datetime.utcnow()
        brokers = ", ".join(self._broker_names or [self._broker_name])
        reason = self._broker_missing_reason or "broker not initialized"
        logging.warning("Broker unavailable; trading loop idle. brokers=%s reason=%s", brokers, reason)

    def _refresh_symbol_venues(self) -> None:
        market_cfg = self.cfg.get("market", {})
        auto_cfg = market_cfg.get("symbol_venues_auto", {})
        if not auto_cfg.get("enabled", False):
            return
        now = datetime.utcnow()
        interval = int(auto_cfg.get("refresh_minutes", 60))
        if self._symbol_venues_at and (now - self._symbol_venues_at).total_seconds() < interval * 60:
            return
        alpaca_cfg = get_alpaca_account_cfg(self.cfg)
        api_key = alpaca_cfg.get("api_key", "")
        api_secret = alpaca_cfg.get("api_secret", "")
        if not api_key or not api_secret:
            return
        exchange_map = auto_cfg.get("exchange_venue_map", {}) or {}
        if not exchange_map:
            return
        max_symbols = int(auto_cfg.get("max_symbols", 50000))
        try:
            venues = load_symbol_venues(api_key, api_secret, max_symbols, exchange_map)
        except Exception as exc:
            logging.warning("Symbol venue refresh failed: %s", exc)
            return
        if venues:
            self._symbol_venues = venues
            self._symbol_venues_at = now

    def _log_ai_filter_heartbeat(self) -> None:
        dyn_cfg = self.cfg.get("data", {}).get("dynamic_symbols", {})
        ai_cfg = dyn_cfg.get("ai_filter", {})
        if not ai_cfg.get("enabled", False) or score_symbols is None:
            return
        now = datetime.utcnow()
        if self._ai_filter_last_run_at is None:
            return
        last_log = self._ai_filter_last_log_at
        if last_log and (now - last_log).total_seconds() < 30:
            return
        refresh_minutes = int(dyn_cfg.get("refresh_minutes", 15))
        if (now - self._ai_filter_last_run_at).total_seconds() > refresh_minutes * 60:
            return
        age_sec = int((now - self._ai_filter_last_run_at).total_seconds())
        logging.info(
            "AI filter heartbeat ok; last_run_sec=%d symbols=%d",
            age_sec,
            self._ai_filter_last_count,
        )
        self._ai_filter_last_log_at = now

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
            "dynamic_symbols": self._dynamic_symbols,
            "dynamic_symbols_at": _dt_to_str(self._dynamic_symbols_at),
            "symbols": self._symbols,
            "symbols_by_strategy": self._symbols_by_strategy,
            "symbol_venues": self._symbol_venues,
            "symbol_venues_at": _dt_to_str(self._symbol_venues_at),
            "news_cache": self._news_cache,
            "news_cache_at": _dt_to_str(self._news_cache_at),
            "last_trade_at": _dt_to_str(default_state.last_trade_at) if default_state else None,
            "equity_start": self._equity_start,
            "equity_peak": self._equity_peak,
            "broker_states": broker_states_payload,
        }
        self._checkpoint_at = maybe_save_checkpoint("trader", payload, self.cfg, self._checkpoint_at)

    def _load_checkpoint(self) -> None:
        data = load_checkpoint("trader", self.cfg)
        if not data:
            return
        payload = data.get("payload", {}) or {}
        self._dynamic_symbols = list(payload.get("dynamic_symbols", []))
        self._dynamic_symbols_at = _dt_from_str(payload.get("dynamic_symbols_at"))
        self._symbols = list(payload.get("symbols", []))
        self._symbols_by_strategy = dict(payload.get("symbols_by_strategy", {}) or {})
        if not self._symbols and self._dynamic_symbols:
            self._symbols = list(self._dynamic_symbols)
        self._symbol_venues = dict(payload.get("symbol_venues", {}) or {})
        self._symbol_venues_at = _dt_from_str(payload.get("symbol_venues_at"))
        self._news_cache = dict(payload.get("news_cache", {}) or {})
        self._news_cache_at = _dt_from_str(payload.get("news_cache_at"))
        self._equity_start = payload.get("equity_start")
        self._equity_peak = payload.get("equity_peak")
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
        now = now or datetime.utcnow()
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
        alpaca_cfg = get_alpaca_account_cfg(self.cfg)
        api_key = alpaca_cfg.get("api_key", "")
        api_secret = alpaca_cfg.get("api_secret", "")

        universe_cfg = dyn_cfg.get("universe", self._symbols)
        max_universe = int(dyn_cfg.get("max_universe", 500))
        universe = self._resolve_universe(universe_cfg, api_key, api_secret, max_universe, portfolio, dyn_cfg)
        if not universe:
            return
        max_symbols = self._resolve_max_symbols(dyn_cfg, universe, portfolio)

        self._symbols_by_strategy = {}
        self._symbols_by_broker = {}
        ai_cfg = dyn_cfg.get("ai_filter", {})
        if (
            ai_cfg.get("enabled", False)
            and self._market_cache_cfg.enabled
            and self._market_cache_cfg.filtered_symbols_enabled
            and ai_cfg.get("use_cached_symbols", False)
            and self._market_cache is not None
        ):
            cache_interval = str(ai_cfg.get("interval", self.cfg.get("data", {}).get("interval", "1m")))
            max_age = interval_to_seconds(cache_interval)
            cached_symbols = self._market_cache.get_filtered_symbols(cache_interval, max_age_seconds=max_age)
            if cached_symbols:
                ordered = self._merge_with_positions(cached_symbols, portfolio, max_symbols)
                self._symbols_by_broker = self._build_symbols_by_broker(ordered, portfolio, dyn_cfg)
                self._symbols_by_strategy["__global__"] = ordered
                for name in self._strategy_names:
                    self._symbols_by_strategy[name] = ordered
                self._symbols = ordered
                self._dynamic_symbols = list(ordered)
                self._dynamic_symbols_at = now
                return
        if ai_cfg.get("enabled", False) and score_symbols is not None:
            ai_cfg_payload = dict(ai_cfg)
            ai_cfg_payload["market_cache"] = self.cfg.get("market_cache", {})
            with self._ai_filter_future_lock:
                if self._ai_filter_future is None:
                    logging.info("AI filter run starting; universe=%d", len(universe))
                    self._ai_filter_future = self._ai_filter_executor.submit(
                        score_symbols,
                        universe,
                        api_key,
                        api_secret,
                        ai_cfg_payload,
                        self.cfg.get("brokers", {}),
                    )
                    self._ai_filter_inflight_at = now
                    return
                if not self._ai_filter_future.done():
                    if self._ai_filter_inflight_at is not None:
                        elapsed = int((now - self._ai_filter_inflight_at).total_seconds())
                        last_log = self._ai_filter_inflight_log_at
                        if last_log is None or (now - last_log).total_seconds() >= 60:
                            logging.info("AI filter still running; elapsed_sec=%d", elapsed)
                            self._ai_filter_inflight_log_at = now
                    return
                try:
                    result = self._ai_filter_future.result()
                    if isinstance(result, tuple) and len(result) == 3:
                        ordered, scores, signal_map = result
                    else:
                        ordered, scores = result
                        signal_map = {}
                except Exception as exc:
                    elapsed = None
                    if self._ai_filter_inflight_at is not None:
                        elapsed = int((now - self._ai_filter_inflight_at).total_seconds())
                    logging.warning("AI filter run failed; elapsed_sec=%s err=%s", elapsed, exc)
                    ordered = []
                    scores = {}
                    signal_map = {}
                inflight_at = self._ai_filter_inflight_at
                self._ai_filter_future = None
                self._ai_filter_inflight_at = None
                self._ai_filter_inflight_log_at = None
            elapsed = None
            if inflight_at is not None:
                elapsed = int((now - inflight_at).total_seconds())
            logging.info(
                "AI filter scored %d symbols (enabled); elapsed_sec=%s top=%s",
                len(ordered),
                elapsed,
                ",".join(ordered[:5]),
            )
            self._ai_filter_last_run_at = now
            self._ai_filter_last_count = len(ordered)
            self._ai_filter_last_signals = dict(signal_map)
            if signal_map:
                for sym, vals in signal_map.items():
                    if vals:
                        self._update_signal_metrics_from_values(sym, vals)
            if not ordered:
                ordered = list(universe)
            if ai_cfg.get("coverage_filter", False) and signal_map:
                covered = {sym for sym, vals in signal_map.items() if vals}
                if covered:
                    before = len(ordered)
                    ordered = [sym for sym in ordered if sym in covered]
                    removed = before - len(ordered)
                    if removed > 0:
                        logging.info("AI filter coverage removed %d symbols without bars.", removed)
            ordered = self._merge_with_positions(ordered, portfolio, max_symbols)
            self._symbols_by_broker = self._build_symbols_by_broker(ordered, portfolio, dyn_cfg)
            self._symbols_by_strategy["__global__"] = ordered
            for name in self._strategy_names:
                self._symbols_by_strategy[name] = ordered
            self._symbols = ordered
            self._dynamic_symbols = list(ordered)
            self._dynamic_symbols_at = now
            if self._market_cache_cfg.enabled and self._market_cache_cfg.filtered_symbols_enabled:
                cache_interval = str(ai_cfg.get("interval", self.cfg.get("data", {}).get("interval", "1m")))
                ttl_seconds = interval_to_seconds(cache_interval)
                if self._market_cache is not None:
                    self._market_cache.set_filtered_symbols(ordered, cache_interval, ttl_seconds=ttl_seconds)
            return
        if ai_cfg.get("enabled", False) and score_symbols is None:
            logging.warning("AI filter enabled but module unavailable; falling back to scanner filters.")
        filters_cfg_default = dyn_cfg.get("filters", {})
        global_candidates = self._scan_with_filters(
            portfolio,
            filters_cfg_default,
            dyn_cfg,
            api_key,
            api_secret,
            universe,
            max_symbols=max_symbols,
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
                max_symbols=max_symbols,
            )
            if candidates:
                self._symbols_by_strategy[name] = candidates
        if self._symbols_by_strategy:
            self._symbols = self._symbols_by_strategy.get("__global__", self._symbols)
            self._dynamic_symbols = list(self._symbols)
            if self._symbols:
                self._symbols_by_broker = self._build_symbols_by_broker(self._symbols, portfolio, dyn_cfg)
        self._dynamic_symbols_at = now
        return

    def _resolve_max_symbols(self, dyn_cfg: dict, universe: list[str], portfolio: dict) -> int:
        try:
            max_symbols = int(dyn_cfg.get("max_symbols", 50))
        except (TypeError, ValueError):
            max_symbols = 50
        if universe:
            max_symbols = min(max_symbols, len(universe)) if max_symbols > 0 else len(universe)
        extras = set()
        for symbol in portfolio.get("positions", {}).keys():
            if symbol:
                extras.add(symbol)
        for order in self._open_orders_cache:
            symbol = order.get("symbol")
            if symbol:
                extras.add(symbol)
        if extras:
            max_symbols = max(max_symbols, len(extras))
        return max_symbols

    def _resolve_max_symbols_for_broker(
        self,
        dyn_cfg: dict,
        universe: list[str],
        portfolio: dict,
        broker_name: str,
    ) -> int:
        try:
            max_symbols = int(dyn_cfg.get("max_symbols", 50))
        except (TypeError, ValueError):
            max_symbols = 50
        if universe:
            max_symbols = min(max_symbols, len(universe)) if max_symbols > 0 else len(universe)
        extras = set()
        for symbol in portfolio.get("positions", {}).keys():
            if symbol:
                extras.add(symbol)
        for symbol in self._open_order_symbols_for_broker(broker_name):
            if symbol:
                extras.add(symbol)
        if extras:
            max_symbols = max(max_symbols, len(extras))
        max_symbols = self._cap_symbols_by_cash(max_symbols, portfolio, dyn_cfg)
        if extras:
            max_symbols = max(max_symbols, len(extras))
        return max_symbols

    def _cap_symbols_by_cash(self, max_symbols: int, portfolio: dict, dyn_cfg: dict) -> int:
        if not dyn_cfg.get("cash_aware", True):
            return max_symbols
        filters_cfg = dyn_cfg.get("filters", {}) or {}
        price_min = filters_cfg.get("price_min", self.cfg.get("trading_limits", {}).get("min_price"))
        try:
            price_min = float(price_min or 0.0)
        except (TypeError, ValueError):
            return max_symbols
        if price_min <= 0:
            return max_symbols
        cap = self._apply_cash_cap(price_min, float("inf"), portfolio, dyn_cfg)
        if cap <= 0:
            return 0
        affordable = int(cap // price_min)
        if affordable <= 0:
            return 0
        return min(max_symbols, affordable)

    def _open_order_symbols_for_broker(self, broker_name: str) -> list[str]:
        symbols: list[str] = []
        single_broker = len(self._broker_map) <= 1
        for order in self._open_orders_cache:
            symbol = order.get("symbol")
            if not symbol:
                continue
            order_broker = order.get("broker")
            if not order_broker and single_broker:
                order_broker = self._broker_name
            if order_broker == broker_name:
                symbols.append(symbol)
        return symbols

    def _merge_with_positions_for_broker(
        self,
        candidates: list[str],
        portfolio: dict,
        broker_name: str,
        max_symbols: int,
    ) -> list[str]:
        held = [s for s in portfolio.get("positions", {}).keys() if s]
        open_order_symbols = self._open_order_symbols_for_broker(broker_name)
        if not held and not open_order_symbols and not candidates:
            return []
        ordered: list[str] = []
        seen = set()
        for symbol in held + open_order_symbols + candidates:
            if symbol in seen:
                continue
            ordered.append(symbol)
            seen.add(symbol)
            if len(ordered) >= max_symbols:
                break
        return ordered

    def _portfolio_snapshot_for_broker_symbols(self, portfolio: dict, broker_name: str) -> dict:
        brokers = portfolio.get("brokers")
        if isinstance(brokers, dict) and broker_name in brokers:
            data = dict(brokers[broker_name])
            data.setdefault("broker", broker_name)
            data.setdefault("positions", {})
            data.setdefault("gross_exposure", 0.0)
            data.setdefault("short_exposure", 0.0)
            return data
        return portfolio

    def _build_symbols_by_broker(self, ordered: list[str], portfolio: dict, dyn_cfg: dict) -> dict[str, list[str]]:
        exec_cfg = self.cfg.get("execution", {}).get("brokers", {}) or {}
        multi_enabled = bool(exec_cfg.get("enabled", False)) and len(self._broker_map) > 1
        if not multi_enabled:
            return {}
        symbols_by_broker: dict[str, list[str]] = {}
        for broker_name in self._broker_map.keys():
            broker_portfolio = self._portfolio_snapshot_for_broker_symbols(portfolio, broker_name)
            max_symbols = self._resolve_max_symbols_for_broker(
                dyn_cfg,
                ordered,
                broker_portfolio,
                broker_name,
            )
            symbols_by_broker[broker_name] = self._merge_with_positions_for_broker(
                ordered,
                broker_portfolio,
                broker_name,
                max_symbols,
            )
        return symbols_by_broker

    def _resolve_universe(
        self,
        universe_cfg: object,
        api_key: str,
        api_secret: str,
        max_universe: int,
        portfolio: dict,
        dyn_cfg: dict | None = None,
    ) -> list[str]:
        if str(universe_cfg) == "brokers_active":
            alpaca_enabled = self.cfg.get("brokers", {}).get("alpaca", {}).get("enabled", True)
            base_cfg = "alpaca_active" if alpaca_enabled else []
            universe = load_universe(api_key, api_secret, base_cfg, max_universe=max_universe)
        else:
            universe = load_universe(api_key, api_secret, universe_cfg, max_universe=max_universe)
        if dyn_cfg and dyn_cfg.get("universe_price_filter", False):
            filters_cfg = dyn_cfg.get("filters", {}) or {}
            price_min = float(filters_cfg.get("price_min", 0.0))
            price_max = self._apply_cash_cap(price_min, float("inf"), portfolio, dyn_cfg)
            filtered = filter_universe_by_price(
                universe,
                api_key=api_key,
                api_secret=api_secret,
                feed=str(dyn_cfg.get("feed", "iex")),
                price_min=price_min,
                price_max=price_max,
                timeout_seconds=int(dyn_cfg.get("timeout_seconds", 10)),
                retries=int(dyn_cfg.get("retries", 2)),
            )
            if filtered:
                universe = filtered
            else:
                if price_max < price_min or price_max <= 0:
                    universe = []
                else:
                    logging.warning("Universe price filter returned no symbols; keeping base universe.")
        exec_cfg = self.cfg.get("execution", {}).get("brokers", {})
        multi_enabled = bool(exec_cfg.get("enabled", False)) and len(self._broker_map) > 1
        if not multi_enabled and str(universe_cfg) != "brokers_active":
            return universe
        extras = set()
        for symbol in portfolio.get("positions", {}).keys():
            extras.add(symbol)
        for order in self._open_orders_cache:
            symbol = order.get("symbol")
            if symbol:
                extras.add(symbol)
        for symbol in self.cfg.get("data", {}).get("symbols", []):
            extras.add(symbol)
        extras_ordered = sorted(extras)
        if len(extras_ordered) >= max_universe:
            return extras_ordered
        ordered: list[str] = []
        seen = set()
        for symbol in extras_ordered + universe:
            if symbol in seen:
                continue
            ordered.append(symbol)
            seen.add(symbol)
            if len(ordered) >= max_universe:
                break
        return ordered

    def _apply_cash_cap(self, price_min: float, price_max: float, portfolio: dict, dyn_cfg: dict) -> float:
        if not dyn_cfg.get("cash_aware", True):
            return price_max
        try:
            cash = float(portfolio.get("cash", 0.0) or 0.0)
            buying_power = float(portfolio.get("buying_power", 0.0) or 0.0)
            equity = float(portfolio.get("equity", 0.0) or 0.0)
        except (TypeError, ValueError):
            return price_max
        funds = buying_power if buying_power > 0 else cash
        if funds <= 0:
            return 0.0
        cash_mode = str(dyn_cfg.get("cash_cap_mode", "cash")).lower()
        cash_max_pct = float(dyn_cfg.get("cash_max_pct", 100.0))
        cash_cap = funds * max(cash_max_pct, 0.0) / 100.0
        if cash_mode == "risk" and equity > 0:
            max_pos_pct = float(self.cfg.get("risk", {}).get("max_position_size_pct", 0.0))
            target_value = equity * (max_pos_pct / 100.0)
            cash_cap = min(cash_cap, target_value)
        buffer_pct = float(dyn_cfg.get("cash_buffer_pct", 95.0))
        cap = cash_cap * max(buffer_pct, 0.0) / 100.0
        if cap <= 0:
            return 0.0
        capped = min(price_max, cap)
        if capped < price_min:
            logging.info("Dynamic symbols funds cap %.2f below price_min %.2f; enforcing cap.", cap, price_min)
            return capped
        return capped

    def _resolve_active_symbols(self) -> list[str]:
        if not self._symbols_by_strategy:
            if self._symbols_by_broker:
                merged = set()
                for symbols in self._symbols_by_broker.values():
                    merged.update(symbols)
                return list(merged) if merged else self._symbols
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
        max_symbols: int | None = None,
    ) -> list[str]:
        price_min = float(filters_cfg.get("price_min", 1.0))
        price_max = self._apply_cash_cap(price_min, float("inf"), portfolio, dyn_cfg)
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
        max_symbols = int(max_symbols) if max_symbols is not None else int(dyn_cfg.get("max_symbols", 50))
        feed = dyn_cfg.get("feed", "iex")
        timeout_seconds = int(dyn_cfg.get("timeout_seconds", 10))
        retries = int(dyn_cfg.get("retries", 2))
        catalyst_map = {}
        if filters.require_catalyst:
            catalyst_map = fetch_catalyst_symbols_for_config(
                universe,
                self.cfg.get("news", {}),
                self.cfg.get("brokers", {}),
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
            return self._merge_with_positions(candidates, portfolio, max_symbols)
        fallback_cfg = dyn_cfg.get("fallback", {})
        if not fallback_cfg.get("enabled", False):
            return self._merge_with_positions([], portfolio, max_symbols)
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
        return self._merge_with_positions(candidates, portfolio, max_symbols)

    def _merge_with_positions(self, candidates: list[str], portfolio: dict, max_symbols: int) -> list[str]:
        held = [s for s in portfolio.get("positions", {}).keys() if s]
        open_order_symbols = [
            order.get("symbol") for order in self._open_orders_cache if order.get("symbol")
        ]
        if not held and not open_order_symbols and not candidates:
            return []
        ordered: list[str] = []
        seen = set()
        for symbol in held + open_order_symbols + candidates:
            if symbol in seen:
                continue
            ordered.append(symbol)
            seen.add(symbol)
            if len(ordered) >= max_symbols:
                break
        return ordered

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
        account = self._account_for_broker(portfolio.get("broker"))
        def _flag_value(key: str) -> bool:
            val = account.get(key)
            return str(val).lower() in {"true", "1", "yes"} or val == True
        market_state["account_flags"] = {
            "account_blocked": _flag_value("account_blocked"),
            "trading_blocked": _flag_value("trading_blocked"),
            "trade_suspended_by_user": _flag_value("trade_suspended_by_user"),
        }
        market_state["catalyst"] = self._news_cache.get(symbol, False)
        market_state["open_orders"] = self._open_orders_cache
        market_state["symbol"] = symbol
        venue = self._symbol_venue(symbol)
        if venue:
            market_state["market_venue"] = venue
            market_state["market_extended"] = is_venue_extended(self.cfg, venue)
        else:
            market_state["market_extended"] = False

    def _recalculate_exposure(self, market_state: dict, portfolio: dict, symbol: str) -> None:
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
                    brokers[name] = {
                        "equity": float(data.get("equity") or 0.0),
                        "cash": float(data.get("cash") or 0.0),
                        "buying_power": float(data.get("buying_power") or 0.0),
                        "positions": {},
                        "gross_exposure": 0.0,
                        "short_exposure": 0.0,
                    }
                equity_val = float(account.get("equity") or sum(v["equity"] for v in brokers.values()))
                cash_val = float(account.get("cash") or sum(v["cash"] for v in brokers.values()))
                buying_power_val = float(
                    account.get("buying_power") or sum(v["buying_power"] for v in brokers.values())
                )
            elif "equity" in account:
                equity_val = float(account.get("equity") or 0.0)
                cash_val = float(account.get("cash") or 0.0)
                buying_power_val = float(account.get("buying_power") or 0.0)
            elif "NetLiquidation" in account:
                equity_val = float(account.get("NetLiquidation") or 0.0)
                cash_val = float(account.get("TotalCashValue") or 0.0)
                buying_power_val = float(account.get("BuyingPower") or account.get("AvailableFunds") or 0.0)

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
        orders: list[dict] = []
        if len(self._broker_map) > 1:
            for name, broker in self._broker_map.items():
                try:
                    broker_orders = broker.get_open_orders()
                except Exception as exc:
                    logging.warning("Open orders snapshot failed for %s: %s", name, exc)
                    continue
                for order in broker_orders:
                    item = dict(order)
                    item["broker"] = name
                    orders.append(item)
        else:
            try:
                orders = self.broker.get_open_orders()
            except Exception as exc:
                logging.warning("Open orders snapshot failed: %s", exc)
                orders = []
            broker_name = self._broker_name
            for order in orders:
                order.setdefault("broker", broker_name)
        self._open_orders_cache = orders
        self._open_orders_at = now
        self._update_open_orders_metrics(symbols)

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
        open_orders = list(self._open_orders_cache)
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

    def _update_open_orders_metrics(self, symbols: list[str]) -> None:
        previous_labels = set(self._open_orders_labels)
        previous_broker_labels = set(self._open_orders_labels_by_broker)
        counts: dict[tuple[str, str], int] = {}
        broker_counts: dict[tuple[str, str, str], int] = {}
        for order in self._open_orders_cache:
            symbol = order.get("symbol")
            side = (order.get("side") or "").lower()
            if not symbol or side not in {"buy", "sell"}:
                continue
            key = (symbol, side)
            counts[key] = counts.get(key, 0) + 1
            broker = order.get("broker") or self._broker_name
            broker_key = (str(broker), symbol, side)
            broker_counts[broker_key] = broker_counts.get(broker_key, 0) + 1
        new_labels = set(counts.keys())
        new_broker_labels = set(broker_counts.keys())
        for symbol, side in previous_labels - new_labels:
            try:
                OPEN_ORDERS.remove(symbol, side)
            except ValueError:
                pass
        for broker, symbol, side in previous_broker_labels - new_broker_labels:
            try:
                OPEN_ORDERS_BY_BROKER.remove(broker, symbol, side)
            except ValueError:
                pass
        for (symbol, side), count in counts.items():
            OPEN_ORDERS.labels(symbol=symbol, side=side).set(count)
        for (broker, symbol, side), count in broker_counts.items():
            OPEN_ORDERS_BY_BROKER.labels(broker=broker, symbol=symbol, side=side).set(count)
        self._open_orders_labels = new_labels
        self._open_orders_labels_by_broker = new_broker_labels

    def _reset_open_orders_metrics(self, symbols: list[str]) -> None:
        for symbol, side in self._open_orders_labels:
            try:
                OPEN_ORDERS.remove(symbol, side)
            except ValueError:
                pass
        self._open_orders_labels.clear()
        for broker, symbol, side in self._open_orders_labels_by_broker:
            try:
                OPEN_ORDERS_BY_BROKER.remove(broker, symbol, side)
            except ValueError:
                pass
        self._open_orders_labels_by_broker.clear()

    def _has_pending_order(self, symbol: str, broker: str | None = None) -> bool:
        exec_cfg = self.cfg.get("execution", {}).get("open_orders", {})
        if not exec_cfg.get("skip_if_pending", True):
            return False
        for order in self._open_orders_cache:
            if order.get("symbol") == symbol:
                if broker and order.get("broker") != broker:
                    continue
                return True
        return False

    def _reserved_cash(self, symbol: str, last_price: float, broker: str | None = None) -> float:
        reserved = 0.0
        for order in self._open_orders_cache:
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
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _var_cvar_from_history(history: list[float], confidence: float) -> dict[str, float]:
    if len(history) < 2:
        return {"var_pct": 0.0, "cvar_pct": 0.0}
    returns = []
    for idx in range(1, len(history)):
        prev = history[idx - 1]
        curr = history[idx]
        if not prev:
            continue
        returns.append((curr - prev) / prev * 100.0)
    if not returns:
        return {"var_pct": 0.0, "cvar_pct": 0.0}
    returns = sorted(returns)
    cutoff = max(int((1.0 - confidence) * len(returns)) - 1, 0)
    var_val = returns[cutoff]
    tail = [r for r in returns if r <= var_val]
    cvar_val = sum(tail) / len(tail) if tail else var_val
    return {"var_pct": abs(var_val), "cvar_pct": abs(cvar_val)}


def _group_exposure(portfolio: dict, mapper) -> dict[str, float]:
    positions = portfolio.get("positions", {})
    exposure: dict[str, float] = {}
    for symbol, pos in positions.items():
        group = mapper(symbol)
        if not group:
            continue
        value = abs(float(pos.get("value", 0.0) or 0.0))
        exposure[group] = exposure.get(group, 0.0) + value
    return exposure


def _realized_volatility_pct(market_state: dict) -> float:
    prices = market_state.get("prices", []) or []
    if len(prices) < 3:
        return 0.0
    returns = []
    for idx in range(1, len(prices)):
        prev = prices[idx - 1]
        curr = prices[idx]
        if not prev:
            continue
        returns.append((curr - prev) / prev)
    if not returns:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / max(len(returns) - 1, 1)
    return (var**0.5) * 100.0


def _active_model_ref(active: dict | None) -> str | None:
    if not active:
        return None
    model_path = active.get("model_path")
    model_sha = active.get("model_sha256")
    if not model_path:
        return None
    return f"{model_path}:{model_sha or ''}"
