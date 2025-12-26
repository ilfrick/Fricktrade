from __future__ import annotations

import logging
from typing import Any

from app.brokers.base import Broker


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
        per_broker: dict[str, dict[str, Any]] = {}
        for name, broker in self._brokers.items():
            try:
                account = broker.get_account()
            except Exception as exc:
                logging.warning("Account fetch failed for %s: %s", name, exc)
                account = {}
            equity, cash = _extract_equity_cash(account)
            total_equity += equity
            total_cash += cash
            per_broker[name] = {"equity": equity, "cash": cash, "raw": account}
        return {"equity": total_equity, "cash": total_cash, "brokers": per_broker}

    def get_positions(self) -> list[dict]:
        positions: list[dict] = []
        for name, broker in self._brokers.items():
            try:
                raw_positions = broker.get_positions()
            except Exception as exc:
                logging.warning("Position fetch failed for %s: %s", name, exc)
                continue
            for pos in raw_positions:
                item = dict(pos)
                item["broker"] = name
                positions.append(item)
        return positions

    def get_open_orders(self) -> list[dict]:
        orders: list[dict] = []
        for name, broker in self._brokers.items():
            try:
                raw_orders = broker.get_open_orders()
            except Exception as exc:
                logging.warning("Open orders fetch failed for %s: %s", name, exc)
                continue
            for order in raw_orders:
                item = dict(order)
                item["broker"] = name
                orders.append(item)
        return orders

    def place_order(self, symbol: str, side: str, qty: float, order_type: str, **kwargs) -> str:
        broker_name = kwargs.get("broker") or self.resolve_broker(symbol, kwargs.get("strategy"))
        broker = self._brokers.get(str(broker_name))
        if broker is None:
            raise ValueError(f"Unknown broker {broker_name!r} for {symbol}")
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


def _extract_equity_cash(account: dict) -> tuple[float, float]:
    equity = cash = 0.0
    if "equity" in account:
        equity = float(account.get("equity") or 0.0)
        cash = float(account.get("cash") or 0.0)
    elif "NetLiquidation" in account:
        equity = float(account.get("NetLiquidation") or 0.0)
        cash = float(account.get("TotalCashValue") or 0.0)
    return equity, cash
