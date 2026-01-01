from __future__ import annotations

from app.execution.order_queue import OrderQueue


class _StubBroker:
    def __init__(self) -> None:
        self.orders = []

    def place_order(self, symbol: str, side: str, qty: float, order_type: str, **kwargs):
        order_id = f"order-{len(self.orders) + 1}"
        self.orders.append({"order_id": order_id, "symbol": symbol, "side": side, "qty": qty})
        return order_id


def test_order_queue_fifo() -> None:
    broker = _StubBroker()
    queue = OrderQueue(broker, "alpaca")
    first_id = queue.enqueue("AAA", "buy", 1)
    second_id = queue.enqueue("BBB", "sell", 2)
    assert first_id is not None
    assert second_id is None

    open_orders = [{"order_id": first_id, "symbol": "AAA", "side": "buy", "qty": 1}]
    queue.update(open_orders)
    responses = queue.pop_responses()
    assert responses

    queue.update([])
    responses = queue.pop_responses()
    assert any(r.status == "completed" for r in responses)
    submitted = any(r.status == "submitted" for r in responses)

    queue.update([{"order_id": "order-2", "symbol": "BBB", "side": "sell", "qty": 2}])
    responses = queue.pop_responses()
    if not submitted:
        assert any(r.status == "submitted" for r in responses)


def test_order_queue_retry() -> None:
    class _FlakyBroker:
        def __init__(self) -> None:
            self.calls = 0
            self.orders = []

        def place_order(self, symbol: str, side: str, qty: float, order_type: str, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise Exception("temporary failure")
            order_id = f"order-{len(self.orders) + 1}"
            self.orders.append({"order_id": order_id, "symbol": symbol, "side": side, "qty": qty})
            return order_id

    broker = _FlakyBroker()
    retry_cfg = {
        "enabled": True,
        "max_attempts": 1,
        "backoff_seconds": 0,
        "max_notional": 0,
        "reasons": ["unknown"],
    }
    queue = OrderQueue(broker, "alpaca", retry_cfg)
    queue.enqueue("AAA", "buy", 1, notional=100)
    responses = queue.pop_responses()
    assert any(r.status == "retrying" for r in responses)
    queue.update([])
    responses = queue.pop_responses()
    assert any(r.status == "submitted" for r in responses)
