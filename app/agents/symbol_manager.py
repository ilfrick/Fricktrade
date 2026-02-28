# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from app.brokers.config_utils import get_alpaca_account_cfg
from app.data.market_cache import interval_to_seconds
from app.data.news import fetch_catalyst_symbols_for_config
from app.data.scanner import ScanFilters, filter_universe_by_price, load_symbol_venues, load_universe, scan_symbols
from app.monitoring.metrics import SYMBOL_ACTIVE, SYMBOL_ACTIVE_BY_BROKER
from app.utils.market import is_market_open, is_venue_open

try:
    from app.data.ai_filter import score_symbols
except Exception:
    score_symbols = None


class SymbolManager:
    """Manages active symbol selection, filtering, and venue resolution."""

    def __init__(
        self,
        cfg: dict,
        broker_map: dict,
        default_broker: str,
        open_order_mgr,
        market_cache_cfg=None,
        market_cache=None,
    ) -> None:
        self._cfg = cfg
        self._broker_map = broker_map
        self._default_broker = default_broker
        self._open_order_mgr = open_order_mgr
        self._market_cache_cfg = market_cache_cfg
        self._market_cache = market_cache

        # Symbol state
        self._dynamic_symbols: list[str] = []
        self._dynamic_symbols_at: datetime | None = None
        self._symbols: list[str] = []
        self._symbols_by_broker: dict[str, list[str]] = {}
        self._symbols_by_strategy: dict[str, list[str]] = {}
        self._symbol_venues: dict[str, str] = {}
        self._symbol_venues_at: datetime | None = None

        # Metrics label tracking
        self._active_symbol_labels: set[str] = set()
        self._active_symbol_labels_by_broker: dict[str, set[str]] = {}

        # AI filter state
        self._ai_filter_last_run_at: datetime | None = None
        self._ai_filter_last_log_at: datetime | None = None
        self._ai_filter_last_count: int = 0
        self._ai_filter_last_signals: dict[str, dict[str, float]] = {}
        self._ai_filter_executor = ThreadPoolExecutor(max_workers=1)
        self._ai_filter_future = None
        self._ai_filter_future_lock = threading.Lock()
        self._ai_filter_inflight_at: datetime | None = None
        self._ai_filter_inflight_log_at: datetime | None = None

    # ------------------------------------------------------------------
    # Properties for external access
    # ------------------------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        return self._symbols

    @symbols.setter
    def symbols(self, value: list[str]) -> None:
        self._symbols = value

    @property
    def symbols_by_broker(self) -> dict[str, list[str]]:
        return self._symbols_by_broker

    @symbols_by_broker.setter
    def symbols_by_broker(self, value: dict[str, list[str]]) -> None:
        self._symbols_by_broker = value

    @property
    def symbols_by_strategy(self) -> dict[str, list[str]]:
        return self._symbols_by_strategy

    @symbols_by_strategy.setter
    def symbols_by_strategy(self, value: dict[str, list[str]]) -> None:
        self._symbols_by_strategy = value

    @property
    def dynamic_symbols(self) -> list[str]:
        return self._dynamic_symbols

    @dynamic_symbols.setter
    def dynamic_symbols(self, value: list[str]) -> None:
        self._dynamic_symbols = value

    @property
    def dynamic_symbols_at(self) -> datetime | None:
        return self._dynamic_symbols_at

    @dynamic_symbols_at.setter
    def dynamic_symbols_at(self, value: datetime | None) -> None:
        self._dynamic_symbols_at = value

    @property
    def symbol_venues(self) -> dict[str, str]:
        return self._symbol_venues

    @symbol_venues.setter
    def symbol_venues(self, value: dict[str, str]) -> None:
        self._symbol_venues = value

    @property
    def symbol_venues_at(self) -> datetime | None:
        return self._symbol_venues_at

    @symbol_venues_at.setter
    def symbol_venues_at(self, value: datetime | None) -> None:
        self._symbol_venues_at = value

    @property
    def ai_filter_last_run_at(self) -> datetime | None:
        return self._ai_filter_last_run_at

    @property
    def ai_filter_last_count(self) -> int:
        return self._ai_filter_last_count

    @property
    def ai_filter_last_signals(self) -> dict[str, dict[str, float]]:
        return self._ai_filter_last_signals

    @property
    def active_symbol_labels(self) -> set[str]:
        return self._active_symbol_labels

    @property
    def active_symbol_labels_by_broker(self) -> dict[str, set[str]]:
        return self._active_symbol_labels_by_broker

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def update_active_symbol_metrics(
        self,
        symbols: list[str],
        routing_cfg: dict | None = None,
        broker_buying_power_fn=None,
    ) -> None:
        from app.execution import routing as routing_utils

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
            routing_cfg = routing_cfg or {}
            routing_mode = str(routing_cfg.get("mode", "default")).lower()
            broker_names = list(self._broker_map.keys())
            if routing_mode == "parallel" and len(broker_names) > 1:
                broker_buying_power = broker_buying_power_fn() if broker_buying_power_fn else {}
                buckets = routing_utils.parallel_partition_symbols(
                    symbols, broker_names, broker_buying_power, routing_cfg, min_symbols=1
                )
                for broker_name, batch in buckets.items():
                    symbols_by_broker[broker_name] = set(batch)
            elif routing_mode == "auto_split" and len(broker_names) > 1:
                buckets = routing_utils.partition_symbols(symbols, broker_names, routing_cfg)
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

    def symbol_venue(self, symbol: str) -> str | None:
        # Crypto symbols are always on the Crypto venue
        if "/" in symbol:
            return "Crypto"
        venue = self._symbol_venues.get(symbol)
        if venue:
            return venue
        market_cfg = self._cfg.get("market", {})
        return market_cfg.get("default_symbol_venue") or market_cfg.get("venue")

    def symbol_sector(self, symbol: str) -> str | None:
        market_cfg = self._cfg.get("market", {})
        sector_map = market_cfg.get("symbol_sectors", {}) or {}
        return sector_map.get(symbol)

    def merge_symbols_with_positions(self, symbols: list[str], portfolio: dict) -> list[str]:
        positions = portfolio.get("positions", {})
        if not positions:
            logging.debug("merge_symbols_with_positions: no positions in portfolio")
            return symbols
        held_symbols = [s for s, p in positions.items() if float(p.get("qty", 0) or 0) != 0]
        if held_symbols:
            logging.debug("merge_symbols_with_positions: adding held positions %s", held_symbols[:10])
        merged = set(symbols)
        for symbol, position in positions.items():
            qty = float(position.get("qty", 0.0) or 0.0)
            if qty != 0:
                merged.add(symbol)
        return list(merged)

    def is_symbol_market_open(self, symbol: str) -> bool:
        market_cfg = self._cfg.get("market", {})
        venue_map = market_cfg.get("symbol_venues", {}) or {}
        default_venue = str(market_cfg.get("default_symbol_venue", "")).strip()
        venue = str(venue_map.get(symbol) or self._symbol_venues.get(symbol) or default_venue).strip()
        if not venue:
            return is_market_open(self._cfg)
        return is_venue_open(self._cfg, venue)

    def refresh_symbol_venues(self) -> None:
        market_cfg = self._cfg.get("market", {})
        auto_cfg = market_cfg.get("symbol_venues_auto", {})
        if not auto_cfg.get("enabled", False):
            return
        now = datetime.now(timezone.utc)
        interval = int(auto_cfg.get("refresh_minutes", 60))
        if self._symbol_venues_at and (now - self._symbol_venues_at).total_seconds() < interval * 60:
            return
        alpaca_cfg = get_alpaca_account_cfg(self._cfg)
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

    def log_ai_filter_heartbeat(self) -> None:
        dyn_cfg = self._cfg.get("data", {}).get("dynamic_symbols", {})
        ai_cfg = dyn_cfg.get("ai_filter", {})
        if not ai_cfg.get("enabled", False) or score_symbols is None:
            return
        now = datetime.now(timezone.utc)
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

    def refresh_dynamic_symbols(
        self,
        portfolio: dict,
        strategy_names: list[str],
        now: datetime | None = None,
        signal_metrics_fn=None,
    ) -> None:
        dyn_cfg = self._cfg.get("data", {}).get("dynamic_symbols", {})
        if not dyn_cfg.get("enabled", False):
            return
        # Skip dynamic symbol refresh (including AI filter) when market is closed
        if not is_market_open(self._cfg):
            return
        now = now or datetime.now(timezone.utc)
        refresh_minutes = int(dyn_cfg.get("refresh_minutes", 15))
        if self._dynamic_symbols_at and (now - self._dynamic_symbols_at).total_seconds() < refresh_minutes * 60:
            return

        provider = dyn_cfg.get("provider", "alpaca")
        if provider != "alpaca":
            return
        alpaca_cfg = get_alpaca_account_cfg(self._cfg)
        api_key = alpaca_cfg.get("api_key", "")
        api_secret = alpaca_cfg.get("api_secret", "")

        universe_cfg = dyn_cfg.get("universe", self._symbols)
        max_universe = int(dyn_cfg.get("max_universe", 500))
        universe = self.resolve_universe(universe_cfg, api_key, api_secret, max_universe, portfolio, dyn_cfg)
        if not universe:
            return
        max_symbols = self.resolve_max_symbols(dyn_cfg, universe, portfolio)

        ai_cfg = dyn_cfg.get("ai_filter", {})
        if (
            ai_cfg.get("enabled", False)
            and self._market_cache_cfg is not None
            and self._market_cache_cfg.enabled
            and self._market_cache_cfg.filtered_symbols_enabled
            and ai_cfg.get("use_cached_symbols", False)
            and self._market_cache is not None
        ):
            cache_interval = str(ai_cfg.get("interval", self._cfg.get("data", {}).get("interval", "1m")))
            max_age = interval_to_seconds(cache_interval)
            cached_symbols = self._market_cache.get_filtered_symbols(cache_interval, max_age_seconds=max_age)
            if cached_symbols:
                ordered = self.merge_with_positions(cached_symbols, portfolio, max_symbols)
                self._symbols_by_strategy = {}
                self._symbols_by_broker = self.build_symbols_by_broker(ordered, portfolio, dyn_cfg)
                self._symbols_by_strategy["__global__"] = ordered
                for name in strategy_names:
                    self._symbols_by_strategy[name] = ordered
                self._symbols = ordered
                self._dynamic_symbols = list(ordered)
                self._dynamic_symbols_at = now
                return
        if ai_cfg.get("enabled", False) and score_symbols is not None:
            ai_cfg_payload = dict(ai_cfg)
            ai_cfg_payload["market_cache"] = self._cfg.get("market_cache", {})
            _ai_filter_fallback = False
            inflight_at = None
            ordered: list[str] = []
            scores: dict = {}
            signal_map: dict = {}
            with self._ai_filter_future_lock:
                if self._ai_filter_future is None:
                    logging.info("AI filter run starting; universe=%d", len(universe))
                    self._ai_filter_future = self._ai_filter_executor.submit(
                        score_symbols,
                        universe,
                        api_key,
                        api_secret,
                        ai_cfg_payload,
                        self._cfg.get("brokers", {}),
                    )
                    self._ai_filter_inflight_at = now
                    return
                if not self._ai_filter_future.done():
                    _inflight_elapsed = 0
                    if self._ai_filter_inflight_at is not None:
                        _inflight_elapsed = int((now - self._ai_filter_inflight_at).total_seconds())
                    _max_inflight = int(ai_cfg.get("max_inflight_seconds", 300))
                    if _inflight_elapsed > _max_inflight:
                        logging.warning(
                            "AI filter abandoned after %d s (max_inflight_seconds=%d); "
                            "falling back to scanner.",
                            _inflight_elapsed,
                            _max_inflight,
                        )
                        self._ai_filter_future = None
                        self._ai_filter_inflight_at = None
                        self._ai_filter_inflight_log_at = None
                        _ai_filter_fallback = True
                    else:
                        last_log = self._ai_filter_inflight_log_at
                        if last_log is None or (now - last_log).total_seconds() >= 60:
                            logging.info("AI filter still running; elapsed_sec=%d", _inflight_elapsed)
                            self._ai_filter_inflight_log_at = now
                        return
                if not _ai_filter_fallback:
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
            if _ai_filter_fallback:
                pass  # fall through to scanner path below
            else:
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
                if signal_map and signal_metrics_fn is not None:
                    for sym, vals in signal_map.items():
                        if vals:
                            signal_metrics_fn(sym, vals)
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
                ordered = self.merge_with_positions(ordered, portfolio, max_symbols)
                self._symbols_by_strategy = {}
                self._symbols_by_broker = self.build_symbols_by_broker(ordered, portfolio, dyn_cfg)
                self._symbols_by_strategy["__global__"] = ordered
                for name in strategy_names:
                    self._symbols_by_strategy[name] = ordered
                self._symbols = ordered
                self._dynamic_symbols = list(ordered)
                self._dynamic_symbols_at = now
                if (
                    self._market_cache_cfg is not None
                    and self._market_cache_cfg.enabled
                    and self._market_cache_cfg.filtered_symbols_enabled
                ):
                    cache_interval = str(ai_cfg.get("interval", self._cfg.get("data", {}).get("interval", "1m")))
                    ttl_seconds = interval_to_seconds(cache_interval)
                    if self._market_cache is not None:
                        self._market_cache.set_filtered_symbols(ordered, cache_interval, ttl_seconds=ttl_seconds)
                return
        if ai_cfg.get("enabled", False) and score_symbols is None:
            logging.warning("AI filter enabled but module unavailable; falling back to scanner filters.")
        self._symbols_by_strategy = {}
        self._symbols_by_broker = {}
        filters_cfg_default = dyn_cfg.get("filters", {})
        global_candidates = self.scan_with_filters(
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
        for name in strategy_names:
            if name == "pattern_trading":
                filters_cfg = self._cfg.get("pattern_trading", {}).get("selection", {})
            else:
                filters_cfg = filters_cfg_default
            candidates = self.scan_with_filters(
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
                self._symbols_by_broker = self.build_symbols_by_broker(self._symbols, portfolio, dyn_cfg)
        self._dynamic_symbols_at = now
        return

    def resolve_max_symbols(self, dyn_cfg: dict, universe: list[str], portfolio: dict) -> int:
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
        for order in self._open_order_mgr.cache:
            symbol = order.get("symbol")
            if symbol:
                extras.add(symbol)
        if extras:
            max_symbols = max(max_symbols, len(extras))
        return max_symbols

    def resolve_max_symbols_for_broker(
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
        for symbol in self._open_order_mgr.symbols_for_broker(broker_name, self._broker_map):
            if symbol:
                extras.add(symbol)
        if extras:
            max_symbols = max(max_symbols, len(extras))
        return max_symbols

    def cap_symbols_by_cash(self, max_symbols: int, portfolio: dict, dyn_cfg: dict) -> int:
        if not dyn_cfg.get("cash_aware", True):
            return max_symbols
        filters_cfg = dyn_cfg.get("filters", {}) or {}
        price_min = filters_cfg.get("price_min", self._cfg.get("trading_limits", {}).get("min_price"))
        try:
            price_min = float(price_min or 0.0)
        except (TypeError, ValueError):
            return max_symbols
        if price_min <= 0:
            return max_symbols
        cap = self.apply_cash_cap(price_min, float("inf"), portfolio, dyn_cfg)
        if cap <= 0:
            return 0
        affordable = int(cap // price_min)
        if affordable <= 0:
            return 0
        return min(max_symbols, affordable)

    def merge_with_positions_for_broker(
        self,
        candidates: list[str],
        portfolio: dict,
        broker_name: str,
        max_symbols: int,
    ) -> list[str]:
        held = [s for s in portfolio.get("positions", {}).keys() if s]
        open_order_symbols = self._open_order_mgr.symbols_for_broker(broker_name, self._broker_map)
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

    def portfolio_snapshot_for_broker_symbols(self, portfolio: dict, broker_name: str) -> dict:
        brokers = portfolio.get("brokers")
        if isinstance(brokers, dict) and broker_name in brokers:
            data = dict(brokers[broker_name])
            data.setdefault("broker", broker_name)
            data.setdefault("positions", {})
            data.setdefault("gross_exposure", 0.0)
            data.setdefault("short_exposure", 0.0)
            return data
        return portfolio

    def build_symbols_by_broker(self, ordered: list[str], portfolio: dict, dyn_cfg: dict) -> dict[str, list[str]]:
        from app.execution import routing as routing_utils

        exec_cfg = self._cfg.get("execution", {}).get("brokers", {}) or {}
        multi_enabled = bool(exec_cfg.get("enabled", False)) and len(self._broker_map) > 1
        if not multi_enabled:
            return {}
        routing_cfg = exec_cfg.get("routing", {})
        broker_names = list(self._broker_map.keys())

        # Collect per-broker buying power and held/open-order symbols
        broker_buying_power: dict[str, float] = {}
        broker_extras: dict[str, list[str]] = {}
        all_extras: set[str] = set()
        for broker_name in broker_names:
            bp = self.portfolio_snapshot_for_broker_symbols(portfolio, broker_name)
            broker_buying_power[broker_name] = float(bp.get("buying_power", 0.0) or 0.0)
            held = [s for s in bp.get("positions", {}).keys() if s]
            oo = self._open_order_mgr.symbols_for_broker(broker_name, self._broker_map)
            extras = list(dict.fromkeys(held + oo))
            broker_extras[broker_name] = extras
            all_extras.update(extras)

        # Partition non-held candidates across brokers proportional to buying power
        candidates = [s for s in ordered if s not in all_extras]
        partitioned = routing_utils.parallel_partition_symbols(
            candidates, broker_names, broker_buying_power, routing_cfg, min_symbols=0,
        )

        # Merge: held/open-order symbols first, then partitioned candidates
        symbols_by_broker: dict[str, list[str]] = {}
        for broker_name in broker_names:
            extras = broker_extras.get(broker_name, [])
            part = partitioned.get(broker_name, [])
            symbols_by_broker[broker_name] = list(dict.fromkeys(extras + part))
        return symbols_by_broker

    def resolve_universe(
        self,
        universe_cfg: object,
        api_key: str,
        api_secret: str,
        max_universe: int,
        portfolio: dict,
        dyn_cfg: dict | None = None,
    ) -> list[str]:
        if str(universe_cfg) == "brokers_active":
            alpaca_enabled = self._cfg.get("brokers", {}).get("alpaca", {}).get("enabled", True)
            base_cfg = "alpaca_active" if alpaca_enabled else []
            universe = load_universe(api_key, api_secret, base_cfg, max_universe=max_universe)
        else:
            universe = load_universe(api_key, api_secret, universe_cfg, max_universe=max_universe)
        if dyn_cfg and dyn_cfg.get("universe_price_filter", False):
            filters_cfg = dyn_cfg.get("filters", {}) or {}
            price_min = float(filters_cfg.get("price_min", 0.0))
            price_max = float(filters_cfg.get("price_max", float("inf")))
            # Cap price_max at buying power: no point scoring symbols you can't buy
            if dyn_cfg.get("cash_aware", True):
                try:
                    buying_power = float(portfolio.get("buying_power", 0.0) or 0.0)
                except (TypeError, ValueError):
                    buying_power = 0.0
                if buying_power > 0:
                    price_max = min(price_max, buying_power)
                    logging.info("Universe price_max capped at buying_power=%.2f", buying_power)
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
        exec_cfg = self._cfg.get("execution", {}).get("brokers", {})
        multi_enabled = bool(exec_cfg.get("enabled", False)) and len(self._broker_map) > 1
        if not multi_enabled and str(universe_cfg) != "brokers_active":
            return universe
        extras = set()
        for symbol in portfolio.get("positions", {}).keys():
            extras.add(symbol)
        for order in self._open_order_mgr.cache:
            symbol = order.get("symbol")
            if symbol:
                extras.add(symbol)
        for symbol in self._cfg.get("data", {}).get("symbols", []):
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

    def apply_cash_cap(self, price_min: float, price_max: float, portfolio: dict, dyn_cfg: dict) -> float:
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
            max_pos_pct = float(self._cfg.get("risk", {}).get("max_position_size_pct", 0.0))
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

    def resolve_active_symbols(self) -> list[str]:
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

    def scan_with_filters(
        self,
        portfolio: dict,
        filters_cfg: dict,
        dyn_cfg: dict,
        api_key: str,
        api_secret: str,
        universe: list[str],
        max_symbols: int | None = None,
        news_cache: dict | None = None,
    ) -> list[str]:
        price_min = float(filters_cfg.get("price_min", 1.0))
        price_max = float(filters_cfg.get("price_max", float("inf")))
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
                self._cfg.get("news", {}),
                self._cfg.get("brokers", {}),
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
            return self.merge_with_positions(candidates, portfolio, max_symbols)
        fallback_cfg = dyn_cfg.get("fallback", {})
        if not fallback_cfg.get("enabled", False):
            return self.merge_with_positions([], portfolio, max_symbols)
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
        fallback_catalysts = (news_cache or {}) if fallback_filters.require_catalyst else {}
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
        return self.merge_with_positions(candidates, portfolio, max_symbols)

    def merge_with_positions(self, candidates: list[str], portfolio: dict, max_symbols: int) -> list[str]:
        held = [s for s in portfolio.get("positions", {}).keys() if s]
        open_order_symbols = [
            order.get("symbol") for order in self._open_order_mgr.cache if order.get("symbol")
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
