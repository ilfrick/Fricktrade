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
        # Collect seqs from both active and entry queue
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


class TestPositionCloseBypass:
    def test_position_close_bypasses_notional_budget(self):
        queue = OrderQueue(
            _mock_broker(), "test",
            retry_cfg={"enabled": True, "max_attempts": 3, "max_notional": 1000},
        )
        queue._retry_notional_used = 999.0
        queue._retry_reset_date = datetime.now(timezone.utc).date()

        req = OrderRequest(symbol="AAPL", side="sell", qty=10, notional=500.0,
                          is_position_close=True)
        result = queue._should_retry(req, "unknown")
        # Position close bypasses budget check
        assert result is True

    def test_regular_order_blocked_by_budget(self):
        queue = OrderQueue(
            _mock_broker(), "test",
            retry_cfg={"enabled": True, "max_attempts": 3, "max_notional": 1000},
        )
        queue._retry_notional_used = 999.0
        queue._retry_reset_date = datetime.now(timezone.utc).date()

        req = OrderRequest(symbol="AAPL", side="sell", qty=10, notional=500.0,
                          is_position_close=False)
        result = queue._should_retry(req, "unknown")
        # Regular order blocked by budget
        assert result is False

    def test_position_close_retry_skips_notional_accounting(self):
        queue = OrderQueue(
            _mock_broker(), "test",
            retry_cfg={"enabled": True, "max_attempts": 3, "backoff_seconds": 5,
                       "max_notional": 1000},
        )
        queue._retry_notional_used = 0.0
        queue._retry_reset_date = datetime.now(timezone.utc).date()

        req = OrderRequest(symbol="AAPL", side="sell", qty=10, notional=500.0,
                          is_position_close=True)
        queue._enqueue_retry(req)
        # Position close should not add to notional used
        assert queue._retry_notional_used == 0.0

    def test_regular_retry_adds_notional(self):
        queue = OrderQueue(
            _mock_broker(), "test",
            retry_cfg={"enabled": True, "max_attempts": 3, "backoff_seconds": 5,
                       "max_notional": 5000},
        )
        queue._retry_notional_used = 0.0
        queue._retry_reset_date = datetime.now(timezone.utc).date()

        req = OrderRequest(symbol="AAPL", side="buy", qty=10, notional=500.0,
                          is_position_close=False)
        queue._enqueue_retry(req)
        assert queue._retry_notional_used == 500.0

    def test_position_close_retry_routes_to_exit_queue(self):
        queue = OrderQueue(
            _mock_broker(), "test",
            retry_cfg={"enabled": True, "max_attempts": 3, "backoff_seconds": 0,
                       "max_notional": 1000},
        )
        req = OrderRequest(symbol="AAPL", side="sell", qty=10, notional=500.0,
                          is_position_close=True)
        queue._enqueue_retry(req)
        assert len(queue._exit_queue) == 1
        assert len(queue._queue) == 0

    def test_regular_retry_routes_to_entry_queue(self):
        queue = OrderQueue(
            _mock_broker(), "test",
            retry_cfg={"enabled": True, "max_attempts": 3, "backoff_seconds": 0,
                       "max_notional": 5000},
        )
        req = OrderRequest(symbol="AAPL", side="buy", qty=10, notional=500.0,
                          is_position_close=False)
        queue._enqueue_retry(req)
        assert len(queue._queue) == 1
        assert len(queue._exit_queue) == 0


class TestDualLane:
    """Exit orders run in a separate lane, never blocked by entry orders."""

    def test_exit_bypasses_active_entry(self):
        """An exit order executes immediately even when an entry is active."""
        broker = MagicMock()
        broker.place_order.side_effect = lambda sym, *a, **kw: f"id-{sym}"
        queue = OrderQueue(broker, "test")

        # Enqueue an entry — becomes active
        entry_id = queue.enqueue("BTC/USD", "buy", 0.1)
        assert entry_id == "id-BTC/USD"
        assert queue._active is not None

        # Enqueue an exit — should start immediately in exit lane
        exit_id = queue.enqueue("ETH/USD", "sell", 1.0, is_position_close=True)
        assert exit_id == "id-ETH/USD"
        assert queue._active_exit is not None

        # Both lanes active simultaneously
        assert queue._active.symbol == "BTC/USD"
        assert queue._active_exit.symbol == "ETH/USD"
        assert broker.place_order.call_count == 2

    def test_exit_completion_frees_exit_lane_only(self):
        """Completing an exit doesn't affect the entry lane."""
        broker = MagicMock()
        broker.place_order.side_effect = lambda sym, *a, **kw: f"id-{sym}"
        queue = OrderQueue(broker, "test", completion_grace_seconds=0)

        queue.enqueue("BTC/USD", "buy", 0.1)
        queue.enqueue("ETH/USD", "sell", 1.0, is_position_close=True)

        # Exit fills (disappears from open_orders), entry still open
        queue.update([{"order_id": "id-BTC/USD", "symbol": "BTC/USD", "side": "buy", "qty": 0.1}])
        responses = queue.pop_responses()
        assert any(r.status == "completed" and r.symbol == "ETH/USD" for r in responses)
        assert queue._active_exit is None  # exit lane free
        assert queue._active is not None   # entry lane still busy

    def test_entry_queues_while_exit_runs(self):
        """A second entry queues behind the first; exits are unaffected."""
        broker = MagicMock()
        broker.place_order.side_effect = lambda sym, *a, **kw: f"id-{sym}"
        queue = OrderQueue(broker, "test")

        queue.enqueue("BTC/USD", "buy", 0.1)
        second = queue.enqueue("SOL/USD", "buy", 5.0)
        assert second == "queued"  # blocked behind BTC entry

        # Exit still goes through immediately
        exit_id = queue.enqueue("ETH/USD", "sell", 1.0, is_position_close=True)
        assert exit_id == "id-ETH/USD"
