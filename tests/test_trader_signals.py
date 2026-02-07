# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""Tests for TradingAgent._combine_signals logic."""

import pytest

pytest.importorskip("tensorflow")
from unittest.mock import MagicMock

from app.agents.trader import TradingAgent


def _make_agent(combine="priority", strategy_names=None):
    """Create a minimal TradingAgent with mocked broker for signal testing."""
    broker = MagicMock()
    broker.get_account.return_value = {"equity": 100000, "cash": 50000, "buying_power": 200000}
    broker.get_positions.return_value = []
    broker.get_open_orders.return_value = []
    broker.is_connected.return_value = True
    cfg = {
        "strategy": {
            "combine": combine,
            "name": "intraday_momentum",
            "names": strategy_names or ["intraday_momentum", "trend_following"],
            "params": {"allow_shorts": False},
        },
        "risk": {
            "max_position_size_pct": 5.0,
            "max_short_exposure_pct": 5.0,
            "max_daily_loss_pct": 2.0,
            "max_portfolio_leverage": 2.0,
            "circuit_breaker_drawdown_pct": 10.0,
        },
        "data": {},
        "execution": {},
        "market": {},
    }
    agent = TradingAgent(broker, cfg)
    return agent


class TestCombineSignalsPriority:
    def test_empty_signals(self):
        agent = _make_agent("priority")
        action, reduce_pct, strategy = agent._combine_signals([])
        assert action == "hold"
        assert reduce_pct == 1.0
        assert strategy is None

    def test_exit_always_wins(self):
        agent = _make_agent("priority")
        signals = [
            {"action": "buy", "name": "intraday_momentum"},
            {"action": "exit", "name": "trend_following"},
        ]
        action, reduce_pct, strategy = agent._combine_signals(signals)
        assert action == "exit"
        assert strategy == "trend_following"

    def test_priority_order(self):
        agent = _make_agent("priority", ["trend_following", "intraday_momentum"])
        signals = [
            {"action": "buy", "name": "intraday_momentum"},
            {"action": "sell", "name": "trend_following"},
        ]
        action, reduce_pct, strategy = agent._combine_signals(signals)
        assert action == "sell"
        assert strategy == "trend_following"

    def test_first_matching_strategy(self):
        agent = _make_agent("priority", ["intraday_momentum", "trend_following"])
        signals = [
            {"action": "buy", "name": "intraday_momentum"},
            {"action": "sell", "name": "trend_following"},
        ]
        action, reduce_pct, strategy = agent._combine_signals(signals)
        assert action == "buy"
        assert strategy == "intraday_momentum"

    def test_unknown_strategy_hold(self):
        agent = _make_agent("priority", ["unknown"])
        signals = [{"action": "buy", "name": "intraday_momentum"}]
        action, _, strategy = agent._combine_signals(signals)
        assert action == "hold"

    def test_reduce_pct_passed_through(self):
        agent = _make_agent("priority")
        signals = [{"action": "sell", "name": "intraday_momentum", "reduce_pct": 0.5}]
        action, reduce_pct, strategy = agent._combine_signals(signals)
        assert action == "sell"
        assert reduce_pct == 0.5


class TestCombineSignalsWeighted:
    def test_buy_wins_by_weight(self):
        agent = _make_agent("weighted")
        signals = [
            {"action": "buy", "name": "a"},
            {"action": "sell", "name": "b"},
        ]
        action, _, strategy = agent._combine_signals(signals, weights={"a": 2.0, "b": 1.0})
        assert action == "buy"
        assert strategy == "a"

    def test_sell_wins_by_weight(self):
        agent = _make_agent("weighted")
        signals = [
            {"action": "buy", "name": "a"},
            {"action": "sell", "name": "b"},
        ]
        action, _, strategy = agent._combine_signals(signals, weights={"a": 1.0, "b": 2.0})
        assert action == "sell"
        assert strategy == "b"

    def test_tied_scores_hold(self):
        agent = _make_agent("weighted")
        signals = [
            {"action": "buy", "name": "a"},
            {"action": "sell", "name": "b"},
        ]
        action, _, _ = agent._combine_signals(signals, weights={"a": 1.0, "b": 1.0})
        assert action == "hold"

    def test_default_weight_1(self):
        agent = _make_agent("weighted")
        signals = [
            {"action": "buy", "name": "a"},
            {"action": "buy", "name": "b"},
            {"action": "sell", "name": "c"},
        ]
        action, _, _ = agent._combine_signals(signals)
        # buy_score=2.0, sell_score=1.0 with default weights
        assert action == "buy"

    def test_sell_reduce_pct_max(self):
        agent = _make_agent("weighted")
        signals = [
            {"action": "sell", "name": "a", "reduce_pct": 0.3},
            {"action": "sell", "name": "b", "reduce_pct": 0.7},
        ]
        action, reduce_pct, _ = agent._combine_signals(signals)
        assert action == "sell"
        assert reduce_pct == 0.7
