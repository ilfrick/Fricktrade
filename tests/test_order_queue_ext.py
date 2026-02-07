# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import heapq
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from app.execution.order_queue import OrderQueue, OrderRequest


def _mock_broker():
    broker = MagicMock()
    broker.submit_order.return_value = {"id": "order-1", "status": "accepted"}
    return broker


class TestFIFOStability:
    def test_same_timestamp_dequeue_in_insertion_order(self):
        """Two OrderRequests with the same earliest_at sort by _seq (insertion order)."""
        now = datetime.now(timezone.utc)
        r1 = OrderRequest(symbol="AAPL", side="buy", qty=10, earliest_at=now, _seq=0)
        r2 = OrderRequest(symbol="GOOG", side="buy", qty=5, earliest_at=now, _seq=1)

        heap = []
        heapq.heappush(heap, r2)
        heapq.heappush(heap, r1)

        first = heapq.heappop(heap)
        second = heapq.heappop(heap)
        assert first.symbol == "AAPL"
        assert second.symbol == "GOOG"

    def test_seq_counter_increments(self):
        queue = OrderQueue(_mock_broker(), "test")
        now = datetime.now(timezone.utc)
        # Enqueue two items — first may be popped by _start_next
        queue.enqueue("AAPL", "buy", 10, earliest_at=now)
        queue.enqueue("GOOG", "buy", 5, earliest_at=now)
        # Collect seqs from both active and queue
        all_items = list(queue._queue)
        if queue._active is not None:
            all_items.append(queue._active)
        seqs = sorted(r._seq for r in all_items)
        assert len(seqs) == 2
        assert seqs[0] < seqs[1]


class TestRetryBudgetReset:
    def test_retry_budget_resets_daily(self):
        queue = OrderQueue(
            _mock_broker(), "test",
            retry_cfg={"enabled": True, "max_attempts": 3, "max_notional": 1000},
        )
        queue._retry_notional_used = 900.0
        queue._retry_reset_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()

        req = OrderRequest(symbol="AAPL", side="buy", qty=10, notional=500.0)
        result = queue._should_retry(req, "timeout")
        # Budget should have been reset, so 500 < 1000 => True
        assert result is True
        assert queue._retry_notional_used == 0.0

    def test_retry_budget_not_reset_same_day(self):
        queue = OrderQueue(
            _mock_broker(), "test",
            retry_cfg={"enabled": True, "max_attempts": 3, "max_notional": 1000},
        )
        queue._retry_notional_used = 900.0
        queue._retry_reset_date = datetime.now(timezone.utc).date()

        req = OrderRequest(symbol="AAPL", side="buy", qty=10, notional=500.0)
        result = queue._should_retry(req, "timeout")
        # 900 + 500 > 1000 => False
        assert result is False
        assert queue._retry_notional_used == 900.0
