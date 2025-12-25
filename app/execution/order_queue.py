import logging
from dataclasses import dataclass, field
from datetime import datetime

from app.brokers.base import Broker


@dataclass
class OrderRequest:
    symbol: str
    side: str
    qty: float
    order_type: str = "market"
    created_at: datetime = field(default_factory=datetime.utcnow)
    order_id: str | None = None


@dataclass
class OrderResponse:
    symbol: str
    broker: str
    status: str
    order_id: str | None = None
    side: str | None = None
    qty: float | None = None
    filled_qty: float | None = None
    filled_avg_price: float | None = None
    received_at: datetime = field(default_factory=datetime.utcnow)


class OrderQueue:
    def __init__(self, broker: Broker, broker_name: str):
        self._broker = broker
        self._broker_name = broker_name
        self._queue: list[OrderRequest] = []
        self._active: OrderRequest | None = None
        self._active_snapshot: dict | None = None
        self._cancel_requested: set[str] = set()
        self._responses: list[OrderResponse] = []

    def enqueue(self, symbol: str, side: str, qty: float) -> str | None:
        request = OrderRequest(symbol=symbol, side=side, qty=qty)
        self._queue.append(request)
        if self._active is None:
            self._start_next()
            return self._active.order_id if self._active else None
        logging.info("Order queued: %s %s qty=%s", side, symbol, qty)
        return None

    def mark_cancel_requested(self, order_id: str) -> None:
        if order_id:
            self._cancel_requested.add(order_id)

    def update(self, open_orders: list[dict]) -> None:
        open_by_id = {str(o.get("order_id")): o for o in open_orders if o.get("order_id")}
        open_ids = set(open_by_id.keys())
        if self._active and self._active.order_id:
            active_id = self._active.order_id
            if active_id in open_by_id:
                snapshot = open_by_id[active_id]
                response = self._response_from_snapshot(snapshot, "open")
                if response and response != self._active_snapshot:
                    self._active_snapshot = response
                    self._responses.append(OrderResponse(**response))
            elif active_id not in open_ids:
                status = "canceled" if self._active.order_id in self._cancel_requested else "completed"
                snapshot = self._active_snapshot or {}
                self._responses.append(
                    OrderResponse(
                        symbol=self._active.symbol,
                        broker=self._broker_name,
                        status=status,
                        order_id=self._active.order_id,
                        side=self._active.side,
                        qty=self._active.qty,
                        filled_qty=snapshot.get("filled_qty"),
                        filled_avg_price=snapshot.get("filled_avg_price"),
                    )
                )
                self._cancel_requested.discard(self._active.order_id)
                self._active = None
                self._active_snapshot = None
        if self._active is None and self._queue:
            self._start_next()

    def pop_responses(self) -> list[OrderResponse]:
        responses = self._responses
        self._responses = []
        return responses

    def _start_next(self) -> None:
        if not self._queue:
            return
        request = self._queue.pop(0)
        try:
            order_id = self._broker.place_order(request.symbol, request.side, request.qty, request.order_type)
        except Exception as exc:
            logging.warning("Queued order failed (%s %s qty=%s): %s", request.side, request.symbol, request.qty, exc)
            self._responses.append(
                OrderResponse(
                    symbol=request.symbol,
                    broker=self._broker_name,
                    status="rejected",
                    order_id=None,
                    side=request.side,
                    qty=request.qty,
                )
            )
            return
        request.order_id = str(order_id)
        self._active = request
        self._responses.append(
            OrderResponse(
                symbol=request.symbol,
                broker=self._broker_name,
                status="submitted",
                order_id=request.order_id,
                side=request.side,
                qty=request.qty,
            )
        )

    def _response_from_snapshot(self, snapshot: dict, fallback_status: str) -> dict | None:
        if not snapshot:
            return None
        order_id = snapshot.get("order_id")
        symbol = snapshot.get("symbol")
        if not order_id or not symbol:
            return None
        status = snapshot.get("status") or fallback_status
        return {
            "symbol": symbol,
            "broker": self._broker_name,
            "status": str(status).lower(),
            "order_id": str(order_id),
            "side": snapshot.get("side"),
            "qty": snapshot.get("qty"),
            "filled_qty": snapshot.get("filled_qty"),
            "filled_avg_price": snapshot.get("filled_avg_price"),
        }
