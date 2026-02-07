# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import logging
from datetime import datetime, timezone

from app.monitoring.metrics import (
    OPEN_ORDERS,
    OPEN_ORDERS_BY_BROKER,
)


class OpenOrderManager:
    """Manages the open-orders cache and related metrics."""

    def __init__(self, cfg: dict, default_broker: str):
        self._cfg = cfg.get("execution", {}).get("open_orders", {})
        self._default_broker = default_broker
        self._cache: list[dict] = []
        self._cache_at: datetime | None = None
        self._labels: set[tuple[str, str]] = set()
        self._labels_by_broker: set[tuple[str, str, str]] = set()

    @property
    def cache(self) -> list[dict]:
        return self._cache

    @cache.setter
    def cache(self, value: list[dict]) -> None:
        self._cache = value

    def refresh(self, broker, broker_map: dict, symbols: list[str]) -> None:
        if not self._cfg.get("enabled", True):
            self._cache = []
            self._cache_at = None
            self._reset_metrics(symbols)
            return
        now = datetime.now(timezone.utc)
        interval_seconds = int(self._cfg.get("interval_seconds", 30))
        if self._cache_at and (now - self._cache_at).total_seconds() < interval_seconds:
            return
        orders: list[dict] = []
        if len(broker_map) > 1:
            for name, b in broker_map.items():
                try:
                    broker_orders = b.get_open_orders()
                except Exception as exc:
                    logging.warning("Open orders snapshot failed for %s: %s", name, exc)
                    continue
                for order in broker_orders:
                    item = dict(order)
                    item["broker"] = name
                    orders.append(item)
        else:
            try:
                orders = broker.get_open_orders()
            except Exception as exc:
                logging.warning("Open orders snapshot failed: %s", exc)
                orders = []
            for order in orders:
                order.setdefault("broker", self._default_broker)
        self._cache = orders
        self._cache_at = now
        self._update_metrics(symbols)

    def has_pending(self, symbol: str, broker: str | None = None) -> bool:
        if not self._cfg.get("skip_if_pending", True):
            return False
        for order in self._cache:
            if order.get("symbol") == symbol:
                if broker and order.get("broker") != broker:
                    continue
                return True
        return False

    def get_pending(self, symbol: str, broker: str | None = None) -> list[dict]:
        pending = [order for order in self._cache if order.get("symbol") == symbol]
        if broker:
            pending = [order for order in pending if order.get("broker") == broker]
        return pending

    def remove_pending(self, symbol: str, broker: str | None = None) -> None:
        if broker:
            self._cache = [
                order
                for order in self._cache
                if order.get("symbol") != symbol or order.get("broker") != broker
            ]
            return
        self._cache = [order for order in self._cache if order.get("symbol") != symbol]

    def symbols_for_broker(self, broker_name: str, broker_map: dict) -> list[str]:
        symbols: list[str] = []
        single_broker = len(broker_map) <= 1
        for order in self._cache:
            symbol = order.get("symbol")
            if not symbol:
                continue
            order_broker = order.get("broker")
            if not order_broker and single_broker:
                order_broker = self._default_broker
            if order_broker == broker_name:
                symbols.append(symbol)
        return symbols

    def _update_metrics(self, symbols: list[str]) -> None:
        previous_labels = set(self._labels)
        previous_broker_labels = set(self._labels_by_broker)
        counts: dict[tuple[str, str], int] = {}
        broker_counts: dict[tuple[str, str, str], int] = {}
        for order in self._cache:
            symbol = order.get("symbol")
            side = (order.get("side") or "").lower()
            if not symbol or side not in {"buy", "sell"}:
                continue
            key = (symbol, side)
            counts[key] = counts.get(key, 0) + 1
            broker = order.get("broker") or self._default_broker
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
        self._labels = new_labels
        self._labels_by_broker = new_broker_labels

    def _reset_metrics(self, symbols: list[str]) -> None:
        for symbol, side in self._labels:
            try:
                OPEN_ORDERS.remove(symbol, side)
            except ValueError:
                pass
        self._labels.clear()
        for broker, symbol, side in self._labels_by_broker:
            try:
                OPEN_ORDERS_BY_BROKER.remove(broker, symbol, side)
            except ValueError:
                pass
        self._labels_by_broker.clear()
