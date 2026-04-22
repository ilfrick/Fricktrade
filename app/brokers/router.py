# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeoutError, as_completed
from typing import Any

from app.brokers.base import Broker
from app.utils.account import extract_equity_cash

try:
    from app.monitoring.metrics import BROKER_REQUESTS, BROKER_LAST_SUCCESS
    _HAS_METRICS = True
except Exception:
    _HAS_METRICS = False

# Consecutive timeouts before a broker is put offline
_CIRCUIT_BREAK_THRESHOLD = 3
# How long (seconds) a broker stays offline before being retried
_CIRCUIT_RESET_SECONDS = 60
# How long (seconds) a cached response is considered fresh enough to serve on timeout
_CACHE_MAX_AGE_SECONDS = 300  # 5 minutes


class BrokerRouter(Broker):
    def __init__(self, brokers: dict[str, Broker], routing: dict | None = None):
        self._brokers = brokers
        self._routing = routing or {}
        # Per-broker response cache: {broker_name: {method: (result, timestamp)}}
        self._response_cache: dict[str, dict[str, tuple[Any, float]]] = {}
        # Circuit-breaker state: consecutive timeouts and offline-until timestamps
        self._consecutive_timeouts: dict[str, int] = {}
        self._offline_until: dict[str, float] = {}

    @property
    def brokers(self) -> dict[str, Broker]:
        return dict(self._brokers)

    def is_connected(self) -> bool:
        return all(broker.is_connected() for broker in self._brokers.values())

    def get_account(self) -> dict:
        total_equity = 0.0
        total_cash = 0.0
        total_buying_power = 0.0
        per_broker: dict[str, dict[str, Any]] = {}
        accounts = self._fetch_per_broker(
            lambda broker: broker.get_account(),
            "get_account",
            "Account fetch failed for %s: %s",
            {},
        )
        for name, account in accounts.items():
            equity, cash, buying_power = extract_equity_cash(account)
            total_equity += equity
            total_cash += cash
            total_buying_power += buying_power
            today_deposits = float(account.get("today_deposits", 0) or 0) if isinstance(account, dict) else 0.0
            per_broker[name] = {
                "equity": equity,
                "cash": cash,
                "buying_power": buying_power,
                "today_deposits": today_deposits,
                "raw": account,
            }
        return {
            "equity": total_equity,
            "cash": total_cash,
            "buying_power": total_buying_power,
            "brokers": per_broker,
        }

    def get_positions(self) -> list[dict]:
        positions: list[dict] = []
        positions_map = self._fetch_per_broker(
            lambda broker: broker.get_positions(),
            "get_positions",
            "Position fetch failed for %s: %s",
            [],
        )
        # Detect virtual accounts sharing the same underlying brokerage account
        # (e.g. alpaca:Realistic and alpaca:Higher on the same API key).
        # Positions are identical across shared accounts — assign each symbol
        # to only the FIRST virtual account to prevent double-counting.
        api_id_map: dict[str, str] = {}  # api_account_id → first broker name
        shared_primary: dict[str, str] = {}  # broker name → primary broker for shared API
        for name, broker in self._brokers.items():
            aid = broker.api_account_id()
            if aid in api_id_map:
                shared_primary[name] = api_id_map[aid]
            else:
                api_id_map[aid] = name
        # Track which (dedup_group, symbol) pairs we've already emitted.
        # For any broker in a shared-API group, use the primary name as the
        # dedup key.  First broker to process a symbol wins; duplicates are
        # skipped regardless of whether they are the primary or secondary.
        _dedup_group: dict[str, str] = {}  # broker name → dedup key (primary name or self)
        for name in self._brokers:
            if name in shared_primary:
                _dedup_group[name] = shared_primary[name]
            elif name in api_id_map.values():
                _dedup_group[name] = name
        _seen: set[tuple[str, str]] = set()
        for name, raw_positions in positions_map.items():
            group = _dedup_group.get(name)
            for pos in raw_positions:
                item = dict(pos)
                item["broker"] = name
                symbol = item.get("symbol", "")
                if group is not None:
                    _key = (group, symbol)
                    if _key in _seen:
                        continue
                    _seen.add(_key)
                positions.append(item)
        return positions

    def get_open_orders(self) -> list[dict]:
        orders: list[dict] = []
        orders_map = self._fetch_per_broker(
            lambda broker: broker.get_open_orders(),
            "get_open_orders",
            "Open orders fetch failed for %s: %s",
            [],
        )
        for name, raw_orders in orders_map.items():
            for order in raw_orders:
                item = dict(order)
                item["broker"] = name
                orders.append(item)
        return orders

    def place_order(self, symbol: str, side: str, qty: float, order_type: str, **kwargs) -> str:
        broker_name = kwargs.get("broker") or self.resolve_broker(symbol, kwargs.get("strategy"))
        broker = self._brokers.get(str(broker_name))
        if broker is None:
            # Log warning and fall back to first available broker
            available = list(self._brokers.keys())
            if not available:
                raise ValueError(f"No brokers available for {symbol}")
            fallback = available[0]
            logging.warning(
                "Unknown broker %r for %s, falling back to %s",
                broker_name,
                symbol,
                fallback,
            )
            broker = self._brokers[fallback]
            broker_name = fallback
        return broker.place_order(symbol, side, qty, order_type, **kwargs)

    def close_position(self, symbol: str, **kwargs) -> None:
        broker_name = kwargs.get("broker")
        if broker_name:
            broker = self._brokers.get(str(broker_name))
            if broker:
                broker.close_position(symbol)
            return
        positions = self.get_positions()
        closed = False
        for pos in positions:
            if pos.get("symbol") == symbol and pos.get("broker") in self._brokers:
                self._brokers[pos["broker"]].close_position(symbol)
                closed = True
        if not closed:
            broker = self._brokers.get(self.resolve_broker(symbol, kwargs.get("strategy")))
            if broker:
                broker.close_position(symbol)

    def cancel_order(self, order_id: str, **kwargs) -> None:
        broker_name = kwargs.get("broker")
        if broker_name:
            broker = self._brokers.get(str(broker_name))
            if broker:
                broker.cancel_order(order_id)
            return
        # No broker specified: try each broker and stop on first success to avoid
        # flooding all brokers with cancel requests for a foreign order_id.
        for name, broker in self._brokers.items():
            try:
                broker.cancel_order(order_id)
                return
            except Exception:
                logging.debug("cancel_order: broker %s does not own order %s, trying next", name, order_id)

    def resolve_broker(self, symbol: str, strategy: str | None = None) -> str:
        routing = self._routing or {}
        symbol_map = routing.get("symbols", {}) or {}
        if symbol in symbol_map:
            return str(symbol_map[symbol])
        if strategy:
            strategy_map = routing.get("strategies", {}) or {}
            if strategy in strategy_map:
                return str(strategy_map[strategy])
        default = routing.get("default")
        if default:
            return str(default)
        return next(iter(self._brokers.keys()))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_circuit_open(self, broker_name: str) -> bool:
        """Return True if broker is currently offline (circuit open)."""
        until = self._offline_until.get(broker_name, 0.0)
        if until and time.monotonic() < until:
            return True
        if until:
            # Reset: offline period expired
            self._offline_until.pop(broker_name, None)
            self._consecutive_timeouts.pop(broker_name, None)
            logging.info("BrokerRouter: %s back online after circuit-break timeout", broker_name)
        return False

    def _record_timeout(self, broker_name: str) -> None:
        """Record a timeout; open circuit after threshold."""
        count = self._consecutive_timeouts.get(broker_name, 0) + 1
        self._consecutive_timeouts[broker_name] = count
        if _HAS_METRICS:
            try:
                BROKER_REQUESTS.labels(broker=broker_name, method="any", status="timeout").inc()
            except Exception:
                pass
        if count >= _CIRCUIT_BREAK_THRESHOLD:
            self._offline_until[broker_name] = time.monotonic() + _CIRCUIT_RESET_SECONDS
            logging.warning(
                "BrokerRouter: %s circuit open after %d consecutive timeouts — "
                "skipping for %ds, serving stale cache",
                broker_name, count, _CIRCUIT_RESET_SECONDS,
            )

    def _record_success(self, broker_name: str, method: str, result: Any) -> None:
        """Cache result and reset circuit-breaker on success."""
        self._consecutive_timeouts.pop(broker_name, None)
        self._offline_until.pop(broker_name, None)
        cache = self._response_cache.setdefault(broker_name, {})
        cache[method] = (result, time.monotonic())
        if _HAS_METRICS:
            try:
                BROKER_REQUESTS.labels(broker=broker_name, method=method, status="success").inc()
                BROKER_LAST_SUCCESS.labels(broker=broker_name, method=method).set(time.time())
            except Exception:
                pass

    def _get_cached(self, broker_name: str, method: str, default: Any) -> Any:
        """Return cached result if available and not too stale, else default."""
        entry = self._response_cache.get(broker_name, {}).get(method)
        if entry is None:
            return default
        result, ts = entry
        age = time.monotonic() - ts
        if age > _CACHE_MAX_AGE_SECONDS:
            logging.warning(
                "BrokerRouter: %s %s cache is %.0fs stale — using anyway (no fresh data)",
                broker_name, method, age,
            )
        else:
            logging.info(
                "BrokerRouter: %s %s serving %.0fs-old cache (broker timeout/offline)",
                broker_name, method, age,
            )
        return result

    def _fetch_per_broker(
        self, func, method: str, error_template: str, default: Any
    ) -> dict[str, Any]:
        if not self._brokers:
            return {}

        # Split into brokers that are online vs circuit-open
        online_brokers = {}
        results: dict[str, Any] = {}
        for name, broker in self._brokers.items():
            if self._is_circuit_open(name):
                results[name] = self._get_cached(name, method, default)
            else:
                online_brokers[name] = broker

        if not online_brokers:
            return results

        max_workers = min(8, len(online_brokers))
        # IMPORTANT: do NOT use 'with ThreadPoolExecutor' — its __exit__ calls
        # shutdown(wait=True) which blocks until all broker threads finish even when
        # as_completed times out, stalling the main loop for as long as the slowest broker.
        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = {
                executor.submit(func, broker): name
                for name, broker in online_brokers.items()
            }
            try:
                for future in as_completed(futures, timeout=30):
                    name = futures[future]
                    try:
                        result = future.result()
                        self._record_success(name, method, result)
                        results[name] = result
                    except Exception as exc:
                        logging.warning(error_template, name, exc)
                        results[name] = self._get_cached(name, method, default)
            except _FutTimeoutError:
                for future, name in futures.items():
                    if name not in results:
                        self._record_timeout(name)
                        results[name] = self._get_cached(name, method, default)
        finally:
            executor.shutdown(wait=False)  # abandon slow threads; do not block
        return results
