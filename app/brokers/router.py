# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from app.brokers.base import Broker
from app.utils.account import extract_equity_cash


class BrokerRouter(Broker):
    def __init__(self, brokers: dict[str, Broker], routing: dict | None = None):
        self._brokers = brokers
        self._routing = routing or {}

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
            "Account fetch failed for %s: %s",
            {},
        )
        for name, account in accounts.items():
            equity, cash, buying_power = extract_equity_cash(account)
            total_equity += equity
            total_cash += cash
            total_buying_power += buying_power
            per_broker[name] = {
                "equity": equity,
                "cash": cash,
                "buying_power": buying_power,
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
            "Position fetch failed for %s: %s",
            [],
        )
        for name, raw_positions in positions_map.items():
            for pos in raw_positions:
                item = dict(pos)
                item["broker"] = name
                positions.append(item)
        return positions

    def get_open_orders(self) -> list[dict]:
        orders: list[dict] = []
        orders_map = self._fetch_per_broker(
            lambda broker: broker.get_open_orders(),
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
        for broker in self._brokers.values():
            broker.cancel_order(order_id)

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

    def _fetch_per_broker(self, func, error_template: str, default: Any) -> dict[str, Any]:
        if not self._brokers:
            return {}
        max_workers = min(8, len(self._brokers))
        results: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(func, broker): name for name, broker in self._brokers.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as exc:
                    logging.warning(error_template, name, exc)
                    results[name] = default
        return results


