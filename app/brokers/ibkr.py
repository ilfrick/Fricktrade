# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from ib_insync import IB, Stock, MarketOrder, LimitOrder

from app.brokers.base import Broker
from app.monitoring.broker_metrics import record_broker_call


class IBKRBroker(Broker):
    def __init__(self, host: str, port: int, client_id: int, name: str = "ibkr", account_id: str | None = None):
        self.ib = IB()
        self.ib.connect(host, port, clientId=client_id)
        self._name = name
        self._account_id = str(account_id) if account_id else ""

    def is_connected(self) -> bool:
        return self.ib.isConnected()

    def get_account(self) -> dict:
        if self._account_id:
            summary = record_broker_call(
                self._name, "get_account", self.ib.accountSummary, account=self._account_id
            )
        else:
            summary = record_broker_call(self._name, "get_account", self.ib.accountSummary)
        data = {s.tag: s.value for s in summary}
        shorting_enabled = _parse_bool(_find_account_flag(data, {"shortingenabled", "shorting_enabled"}))
        if shorting_enabled is not None:
            data["shorting_enabled"] = shorting_enabled
        return data

    def get_positions(self) -> list[dict]:
        positions = []
        for pos in record_broker_call(self._name, "get_positions", self.ib.positions):
            if self._account_id and getattr(pos, "account", None) != self._account_id:
                continue
            positions.append(
                {
                    "symbol": pos.contract.symbol,
                    "qty": float(pos.position),
                    "avg_cost": float(pos.avgCost),
                }
            )
        return positions

    def get_open_orders(self) -> list[dict]:
        orders = []
        for trade in record_broker_call(self._name, "get_open_orders", self.ib.trades):
            order = trade.order
            status = trade.orderStatus.status
            if status not in {"Submitted", "PreSubmitted"}:
                continue
            if self._account_id:
                account = (
                    getattr(trade, "account", None)
                    or getattr(order, "account", None)
                    or getattr(trade.orderStatus, "account", None)
                )
                if account and account != self._account_id:
                    continue
            orders.append(
                {
                    "order_id": str(order.orderId),
                    "symbol": trade.contract.symbol,
                    "side": "buy" if order.action.upper() == "BUY" else "sell",
                    "qty": float(order.totalQuantity),
                    "limit_price": getattr(order, "lmtPrice", None),
                    "status": status,
                    "filled_qty": float(getattr(trade.orderStatus, "filled", 0.0) or 0.0),
                    "filled_avg_price": getattr(trade.orderStatus, "avgFillPrice", None),
                }
            )
        return orders

    def place_order(self, symbol: str, side: str, qty: float, order_type: str, **kwargs) -> str:
        contract = Stock(symbol, "SMART", "EUR")
        action = "BUY" if side.lower() == "buy" else "SELL"
        extended_hours = bool(kwargs.get("extended_hours", False))
        if str(order_type).lower() == "limit":
            limit_price = kwargs.get("limit_price")
            if limit_price is None:
                raise ValueError("limit_price required for limit orders")
            order = LimitOrder(action, qty, float(limit_price))
        else:
            order = MarketOrder(action, qty)
        if extended_hours:
            order.outsideRth = True
        if self._account_id:
            order.account = self._account_id
        trade = record_broker_call(self._name, "place_order", self.ib.placeOrder, contract, order)
        self.ib.sleep(0.5)
        return str(trade.order.permId)

    def close_position(self, symbol: str) -> None:
        positions = record_broker_call(self._name, "close_position", self.ib.positions)
        for pos in positions:
            if self._account_id and getattr(pos, "account", None) != self._account_id:
                continue
            if pos.contract.symbol == symbol:
                side = "SELL" if pos.position > 0 else "BUY"
                order = MarketOrder(side, abs(pos.position))
                if self._account_id:
                    order.account = self._account_id
                record_broker_call(self._name, "close_position", self.ib.placeOrder, pos.contract, order)

    def cancel_order(self, order_id: str) -> None:
        try:
            oid = int(order_id)
        except (TypeError, ValueError):
            return
        for trade in record_broker_call(self._name, "cancel_order", self.ib.trades):
            order = trade.order
            if self._account_id:
                account = (
                    getattr(trade, "account", None)
                    or getattr(order, "account", None)
                    or getattr(trade.orderStatus, "account", None)
                )
                if account and account != self._account_id:
                    continue
            if order.orderId == oid or getattr(order, "permId", None) == oid:
                record_broker_call(self._name, "cancel_order", self.ib.cancelOrder, order)
                return


def _find_account_flag(data: dict, names: set[str]) -> str | None:
    for key, value in data.items():
        if str(key).lower() in names:
            return value
    return None


def _parse_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None
