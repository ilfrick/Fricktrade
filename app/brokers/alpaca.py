from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

from app.brokers.base import Broker
from app.monitoring.broker_metrics import record_broker_call


class AlpacaBroker(Broker):
    def __init__(self, api_key: str, api_secret: str, base_url: str, paper: bool = True, name: str = "alpaca"):
        self.client = TradingClient(api_key, api_secret, paper=paper, url_override=base_url)
        self._name = name

    def is_connected(self) -> bool:
        try:
            record_broker_call(self._name, "get_account", self.client.get_account)
            return True
        except Exception:
            return False

    def get_account(self) -> dict:
        account = record_broker_call(self._name, "get_account", self.client.get_account)
        return account.dict()

    def get_positions(self) -> list[dict]:
        def _fetch_positions():
            try:
                return self.client.get_all_positions()
            except AttributeError:
                return self.client.list_positions()

        positions = record_broker_call(self._name, "get_positions", _fetch_positions)
        return [pos.dict() if hasattr(pos, "dict") else dict(pos) for pos in positions]

    def get_open_orders(self) -> list[dict]:
        def _fetch_orders():
            try:
                return self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
            except AttributeError:
                return self.client.list_orders(status="open")

        orders = record_broker_call(self._name, "get_open_orders", _fetch_orders)
        results = []
        for order in orders:
            data = order.dict() if hasattr(order, "dict") else dict(order)
            results.append(
                {
                    "order_id": data.get("id") or data.get("order_id"),
                    "symbol": data.get("symbol"),
                    "side": data.get("side"),
                    "qty": float(data.get("qty") or 0.0),
                    "limit_price": data.get("limit_price"),
                    "status": data.get("status"),
                    "filled_qty": float(data.get("filled_qty") or 0.0),
                    "filled_avg_price": data.get("filled_avg_price"),
                }
            )
        return results

    def place_order(self, symbol: str, side: str, qty: float, order_type: str, **kwargs) -> str:
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        if str(order_type).lower() == "limit":
            limit_price = kwargs.get("limit_price")
            if limit_price is None:
                raise ValueError("limit_price required for limit orders")
            order_req = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                limit_price=float(limit_price),
            )
        else:
            order_req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
        order = record_broker_call(
            self._name,
            "place_order",
            self.client.submit_order,
            order_req,
        )
        return order.id

    def close_position(self, symbol: str) -> None:
        try:
            record_broker_call(self._name, "close_position", self.client.close_position, symbol)
        except Exception:
            # Ignore if position does not exist.
            return

    def cancel_order(self, order_id: str) -> None:
        if not order_id:
            return
        record_broker_call(self._name, "cancel_order", self.client.cancel_order_by_id, order_id)
