# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from ib_insync import IB, Stock, MarketOrder

from app.brokers.base import Broker


class IBKRBroker(Broker):
    def __init__(self, host: str, port: int, client_id: int):
        self.ib = IB()
        self.ib.connect(host, port, clientId=client_id)

    def is_connected(self) -> bool:
        return self.ib.isConnected()

    def get_account(self) -> dict:
        summary = self.ib.accountSummary()
        return {s.tag: s.value for s in summary}

    def get_positions(self) -> list[dict]:
        positions = []
        for pos in self.ib.positions():
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
        for trade in self.ib.trades():
            order = trade.order
            status = trade.orderStatus.status
            if status not in {"Submitted", "PreSubmitted"}:
                continue
            orders.append(
                {
                    "order_id": str(order.orderId),
                    "symbol": trade.contract.symbol,
                    "side": "buy" if order.action.upper() == "BUY" else "sell",
                    "qty": float(order.totalQuantity),
                    "limit_price": getattr(order, "lmtPrice", None),
                }
            )
        return orders

    def place_order(self, symbol: str, side: str, qty: float, order_type: str, **kwargs) -> str:
        contract = Stock(symbol, "SMART", "EUR")
        order = MarketOrder("BUY" if side.lower() == "buy" else "SELL", qty)
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(0.5)
        return str(trade.order.permId)

    def close_position(self, symbol: str) -> None:
        positions = self.ib.positions()
        for pos in positions:
            if pos.contract.symbol == symbol:
                side = "SELL" if pos.position > 0 else "BUY"
                order = MarketOrder(side, abs(pos.position))
                self.ib.placeOrder(pos.contract, order)

    def cancel_order(self, order_id: str) -> None:
        try:
            oid = int(order_id)
        except (TypeError, ValueError):
            return
        for trade in self.ib.trades():
            order = trade.order
            if order.orderId == oid or getattr(order, "permId", None) == oid:
                self.ib.cancelOrder(order)
                return
