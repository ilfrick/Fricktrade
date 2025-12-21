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
