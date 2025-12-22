from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from app.brokers.base import Broker


class AlpacaBroker(Broker):
    def __init__(self, api_key: str, api_secret: str, base_url: str, paper: bool = True):
        self.client = TradingClient(api_key, api_secret, paper=paper, url_override=base_url)

    def is_connected(self) -> bool:
        try:
            self.client.get_account()
            return True
        except Exception:
            return False

    def get_account(self) -> dict:
        account = self.client.get_account()
        return account.dict()

    def get_positions(self) -> list[dict]:
        try:
            positions = self.client.get_all_positions()
        except AttributeError:
            positions = self.client.list_positions()
        return [pos.dict() if hasattr(pos, "dict") else dict(pos) for pos in positions]

    def place_order(self, symbol: str, side: str, qty: float, order_type: str, **kwargs) -> str:
        order_req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = self.client.submit_order(order_req)
        return order.id

    def close_position(self, symbol: str) -> None:
        self.client.close_position(symbol)
