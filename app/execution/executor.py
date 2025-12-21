from app.brokers.base import Broker


class ExecutionEngine:
    def __init__(self, broker: Broker):
        self.broker = broker

    def execute(self, symbol: str, action: str, qty: float) -> str | None:
        if action == "buy":
            return self.broker.place_order(symbol, "buy", qty, "market")
        if action == "sell":
            return self.broker.place_order(symbol, "sell", qty, "market")
        if action == "exit":
            self.broker.close_position(symbol)
            return None
        return None
