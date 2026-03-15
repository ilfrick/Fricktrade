# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import heapq
import itertools
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from functools import total_ordering

from app.brokers.base import Broker
from app.monitoring.metrics import ORDER_REJECTS, PDT_BLOCKS


@total_ordering
@dataclass
class OrderRequest:
    """
    Represents an order request in the queue.

    Uses @total_ordering for heapq compatibility (ordered by earliest_at).
    """

    symbol: str
    side: str
    qty: float
    order_type: str = "market"
    limit_price: float | None = None
    extended_hours: bool = False
    earliest_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    order_id: str | None = None
    attempts: int = 0
    notional: float | None = None
    is_position_close: bool = False
    _seq: int = 0

    def __lt__(self, other: "OrderRequest") -> bool:
        return (self.earliest_at, self._seq) < (other.earliest_at, other._seq)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OrderRequest):
            return NotImplemented
        return (self.earliest_at, self._seq) == (other.earliest_at, other._seq)


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
    reason: str = ""
    reserved_notional: float = 0.0  # original notional reserved at enqueue time; fallback for release when fill price unavailable
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class OrderQueue:
    """
    FIFO order queue with priority scheduling and retry logic.

    Uses heapq for O(log n) insertion instead of O(n log n) sorting.
    Thread-safe: All operations are protected by a lock.
    """

    def __init__(
        self,
        broker: Broker,
        broker_name: str,
        retry_cfg: dict | None = None,
        completion_grace_seconds: int = 0,
    ):
        self._broker = broker
        self._broker_name = broker_name
        self._retry_cfg = retry_cfg or {}
        self._queue: list[OrderRequest] = []  # heapq
        self._active: OrderRequest | None = None
        self._active_snapshot: dict | None = None
        self._cancel_requested: set[str] = set()
        self._responses: list[OrderResponse] = []
        self._retry_notional_used = 0.0
        self._retry_reset_date: date | None = None
        self._seq_counter = itertools.count()
        self._completion_grace_seconds = max(int(completion_grace_seconds), 0)
        self._missing_since: datetime | None = None
        self._lock = threading.Lock()

    def enqueue(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: float | None = None,
        extended_hours: bool = False,
        earliest_at: datetime | None = None,
        notional: float | None = None,
        is_position_close: bool = False,
    ) -> str | None:
        request = OrderRequest(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            limit_price=limit_price,
            extended_hours=extended_hours,
            earliest_at=earliest_at or datetime.now(timezone.utc),
            notional=notional,
            is_position_close=is_position_close,
        )
        result_order_id = None
        started_immediately = False

        with self._lock:
            request._seq = next(self._seq_counter)
            heapq.heappush(self._queue, request)
            if self._active is None:
                self._start_next()
                if self._active:
                    result_order_id = self._active.order_id
                    started_immediately = True

        if not started_immediately:
            logging.info("Order queued: %s %s qty=%s type=%s", side, symbol, qty, order_type)
            return "queued"  # truthy sentinel: order accepted to heap, not yet submitted to broker

        return result_order_id

    def mark_cancel_requested(self, order_id: str) -> None:
        if order_id:
            with self._lock:
                self._cancel_requested.add(order_id)

    def update(self, open_orders: list[dict]) -> None:
        now = datetime.now(timezone.utc)
        open_by_id = {str(o.get("order_id")): o for o in open_orders if o.get("order_id")}
        open_ids = set(open_by_id.keys())
        with self._lock:
            if self._active and self._active.order_id:
                active_id = self._active.order_id
                if active_id in open_by_id:
                    # Stuck-order timeout: if the order has been open longer than
                    # max_order_age_seconds, cancel it and unblock the queue.
                    # Default 0 = disabled.
                    max_age = int(self._retry_cfg.get("max_order_age_seconds", 0))
                    age = (now - self._active.created_at).total_seconds()
                    timed_out = max_age > 0 and age > max_age
                    if timed_out:
                        try:
                            self._broker.cancel_order(active_id)
                        except Exception as _exc:
                            logging.warning(
                                "Stuck order cancel failed (%s %s id=%s age=%.0fs): %s",
                                self._active.side, self._active.symbol, active_id, age, _exc,
                            )
                        logging.warning(
                            "Stuck order timed out after %.0fs — cancelled and unblocked queue "
                            "(%s %s qty=%s id=%s)",
                            age, self._active.side, self._active.symbol,
                            self._active.qty, active_id,
                        )
                        self._responses.append(OrderResponse(
                            symbol=self._active.symbol,
                            broker=self._broker_name,
                            status="timed_out",
                            order_id=active_id,
                            side=self._active.side,
                            qty=self._active.qty,
                            reserved_notional=float(self._active.notional or 0.0),
                        ))
                        self._cancel_requested.discard(active_id)
                        self._active = None
                        self._active_snapshot = None
                    else:
                        self._missing_since = None
                        snapshot = open_by_id[active_id]
                        response = self._response_from_snapshot(snapshot, "open")
                        if response and response != self._active_snapshot:
                            self._active_snapshot = response
                            self._responses.append(OrderResponse(**response))
                elif active_id not in open_ids:
                    if self._completion_grace_seconds > 0:
                        if self._missing_since is None:
                            self._missing_since = now
                            return
                        if (now - self._missing_since).total_seconds() < self._completion_grace_seconds:
                            return
                    self._missing_since = None
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
                            reserved_notional=float(self._active.notional or 0.0),
                        )
                    )
                    self._cancel_requested.discard(self._active.order_id)
                    self._active = None
                    self._active_snapshot = None
            if self._active is None and self._queue:
                self._start_next()

    def pop_responses(self) -> list[OrderResponse]:
        with self._lock:
            responses = self._responses
            self._responses = []
            return responses

    def _start_next(self) -> None:
        """Start processing the next order in queue. Must be called with lock held."""
        if not self._queue:
            return
        now = datetime.now(timezone.utc)
        # Peek at the top of the heap
        request = self._queue[0]
        if request.earliest_at > now:
            return
        # Pop from heap
        request = heapq.heappop(self._queue)
        try:
            order_id = self._broker.place_order(
                request.symbol,
                request.side,
                request.qty,
                request.order_type,
                limit_price=request.limit_price,
                extended_hours=request.extended_hours,
            )
        except Exception as exc:
            code = "unknown"
            try:
                payload = json.loads(str(exc))
                if isinstance(payload, dict) and payload.get("code") is not None:
                    code = str(payload.get("code"))
            except Exception:
                pass
            reason = _reject_reason(code, exc)
            if self._should_retry(request, reason):
                self._enqueue_retry(request)
                self._responses.append(
                    OrderResponse(
                        symbol=request.symbol,
                        broker=self._broker_name,
                        status="retrying",
                        order_id=None,
                        side=request.side,
                        qty=request.qty,
                    )
                )
                return
            ORDER_REJECTS.labels(
                broker=self._broker_name,
                side=request.side,
                reason=reason,
            ).inc()
            if reason == "pdt_protection":
                PDT_BLOCKS.labels(
                    broker=self._broker_name,
                    symbol=request.symbol,
                    side=request.side,
                ).inc()
            logging.warning(
                "Queued order failed (%s %s qty=%s code=%s reason=%s): %s",
                request.side,
                request.symbol,
                request.qty,
                code,
                reason,
                exc,
            )
            self._responses.append(
                OrderResponse(
                    symbol=request.symbol,
                    broker=self._broker_name,
                    status="rejected",
                    order_id=None,
                    side=request.side,
                    qty=request.qty,
                    reason=reason,
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
                reserved_notional=float(request.notional or 0.0),
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

    def _maybe_reset_retry_budget(self) -> None:
        """Reset retry notional budget at the start of each new day."""
        today = datetime.now(timezone.utc).date()
        if self._retry_reset_date != today:
            self._retry_notional_used = 0.0
            self._retry_reset_date = today

    def _should_retry(self, request: OrderRequest, reason: str) -> bool:
        self._maybe_reset_retry_budget()
        if not self._retry_cfg.get("enabled", False):
            return False
        max_attempts = int(self._retry_cfg.get("max_attempts", 0))
        if max_attempts <= 0:
            return False
        if request.attempts >= max_attempts:
            return False
        allowed = self._retry_cfg.get("reasons", [])
        if isinstance(allowed, list) and allowed:
            if reason not in {str(item) for item in allowed}:
                return False
        # Position-close orders bypass the retry notional budget
        if not request.is_position_close:
            max_notional = float(self._retry_cfg.get("max_notional", 0.0) or 0.0)
            notional = float(request.notional or 0.0)
            if max_notional > 0 and (self._retry_notional_used + notional) > max_notional:
                return False
        return True

    def _enqueue_retry(self, request: OrderRequest) -> None:
        """Re-enqueue a request for retry with exponential backoff."""
        request.attempts += 1
        backoff = int(self._retry_cfg.get("backoff_seconds", 5))
        request.earliest_at = datetime.now(timezone.utc) + timedelta(seconds=backoff * request.attempts)
        request._seq = next(self._seq_counter)
        # Position-close orders skip notional accounting
        if not request.is_position_close:
            notional = float(request.notional or 0.0)
            if notional > 0:
                self._retry_notional_used += notional
        heapq.heappush(self._queue, request)


def _reject_reason(code: str, exc: Exception) -> str:
    """Extract reject reason from error code or exception text."""
    text = str(exc).lower()
    # Text checks take priority — Alpaca reuses code 40310000 for both
    # "cost basis < $10" and "insufficient balance for USDC/USDT".
    if "floors to 0" in text or "floors to zero" in text:
        return "floors_to_zero"  # sub-step qty — will never succeed, do not retry
    if "filter failure: notional" in text or "notional" in text and "filter" in text:
        return "floors_to_zero"  # Binance -1013: sub-minimum notional, same treatment
    if "pattern day trading" in text or "pdt" in text:
        return "pdt_protection"
    if "insufficient balance for" in text:
        return "insufficient_stablecoin"
    if "insufficient" in text or "insufficient buying power" in text:
        return "insufficient_funds"
    if "cost basis" in text or "minimal amount" in text:
        return "min_order_notional"
    mapping = {
        "40310100": "pdt_protection",
        "40310000": "min_order_notional",  # Alpaca: cost basis < $10 minimum
        "-1013": "floors_to_zero",         # Binance: Filter failure NOTIONAL — sub-minimum value
        "-1111": "floors_to_zero",         # Binance: LOT_SIZE precision error
        "-1121": "invalid_symbol",         # Binance: symbol not listed on this account/env
    }
    if code in mapping:
        return mapping[code]
    return "unknown"
