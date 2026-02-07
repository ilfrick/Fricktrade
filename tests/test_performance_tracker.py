# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from datetime import datetime, timedelta, timezone

from app.agents.performance import PerformanceTracker


class FakeBrokerState:
    def __init__(self):
        self.pending_entry_strategy = {}
        self.position_state = {}
        self.strategy_trades = {}
        self.symbol_trades = {}
        self.disabled_strategies = set()
        self.performance_last_report_at = None
        self.last_prices = {}


class TestComputeTradeStats:
    def test_empty_records(self):
        stats = PerformanceTracker.compute_trade_stats([])
        assert stats["trades"] == 0
        assert stats["win_rate"] == 0.0

    def test_all_winning(self):
        now = datetime.now(timezone.utc)
        records = [
            {"ts": now - timedelta(hours=2), "pnl_pct": 1.0},
            {"ts": now - timedelta(hours=1), "pnl_pct": 2.0},
            {"ts": now, "pnl_pct": 0.5},
        ]
        stats = PerformanceTracker.compute_trade_stats(records)
        assert stats["trades"] == 3
        assert stats["win_rate"] == 1.0
        assert stats["avg_pnl_pct"] > 0

    def test_mixed_trades(self):
        now = datetime.now(timezone.utc)
        records = [
            {"ts": now - timedelta(hours=2), "pnl_pct": 5.0},
            {"ts": now - timedelta(hours=1), "pnl_pct": -3.0},
        ]
        stats = PerformanceTracker.compute_trade_stats(records)
        assert stats["trades"] == 2
        assert stats["win_rate"] == 0.5
        assert stats["drawdown_pct"] > 0


class TestPruneTradeRecords:
    def test_prune_removes_old(self):
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=7)
        records = [
            {"ts": now - timedelta(days=10), "pnl_pct": 1.0},
            {"ts": now - timedelta(days=1), "pnl_pct": 2.0},
        ]
        result = PerformanceTracker.prune_trade_records(records, cutoff)
        assert len(result) == 1
        assert result[0]["pnl_pct"] == 2.0

    def test_prune_empty(self):
        now = datetime.now(timezone.utc)
        assert PerformanceTracker.prune_trade_records([], now) == []


class TestRecordTrade:
    def test_record_trade_appends(self):
        tracker = PerformanceTracker(
            {"strategy": {"performance": {"enabled": True}}},
            strategy_names=["strat_a"],
        )
        bs = FakeBrokerState()
        now = datetime.now(timezone.utc)
        tracker.record_trade(bs, "strat_a", "AAPL", 1.5, now)
        assert len(bs.symbol_trades["AAPL"]) == 1
        assert len(bs.strategy_trades["strat_a"]) == 1

    def test_record_trade_unknown_strategy(self):
        tracker = PerformanceTracker(
            {"strategy": {"performance": {"enabled": True}}},
            strategy_names=["strat_a"],
        )
        bs = FakeBrokerState()
        now = datetime.now(timezone.utc)
        tracker.record_trade(bs, "unknown_strat", "AAPL", 1.5, now)
        assert len(bs.symbol_trades["AAPL"]) == 1
        assert "unknown_strat" not in bs.strategy_trades


class TestPendingEntryStrategy:
    def test_consume(self):
        tracker = PerformanceTracker(
            {"strategy": {"performance": {"enabled": True}}},
            strategy_names=["strat_a"],
        )
        bs = FakeBrokerState()
        bs.pending_entry_strategy["AAPL"] = {"strategy": "strat_a", "ts": datetime.now(timezone.utc)}
        result = tracker.consume_pending_entry_strategy("AAPL", bs, ["strat_a"])
        assert result == "strat_a"
        assert "AAPL" not in bs.pending_entry_strategy

    def test_consume_empty(self):
        tracker = PerformanceTracker(
            {"strategy": {"performance": {"enabled": True}}},
            strategy_names=["strat_a"],
        )
        bs = FakeBrokerState()
        result = tracker.consume_pending_entry_strategy("AAPL", bs, ["strat_a"])
        assert result is None

    def test_prune_stale(self):
        tracker = PerformanceTracker(
            {"strategy": {"performance": {"enabled": True}}},
            strategy_names=["strat_a"],
        )
        bs = FakeBrokerState()
        old_ts = datetime.now(timezone.utc) - timedelta(days=2)
        bs.pending_entry_strategy["AAPL"] = {"strategy": "strat_a", "ts": old_ts}
        bs.pending_entry_strategy["GOOG"] = {"strategy": "strat_a", "ts": datetime.now(timezone.utc)}
        tracker.prune_pending_entry_strategies(datetime.now(timezone.utc), bs)
        assert "AAPL" not in bs.pending_entry_strategy
        assert "GOOG" in bs.pending_entry_strategy
