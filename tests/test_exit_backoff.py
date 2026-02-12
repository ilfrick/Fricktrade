# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

tf = __import__("pytest").importorskip("tensorflow")

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from app.execution.order_queue import OrderResponse


def _make_agent():
    from app.agents.trader import TradingAgent
    broker = MagicMock()
    broker.get_account.return_value = {"equity": 100000, "cash": 50000}
    broker.get_positions.return_value = []
    cfg = {
        "strategies": {"active": ["trend_following"], "combine": "first", "params": {}},
        "risk": {"enabled": False},
        "data": {},
        "learning": {},
        "orchestrator": {"mode": "select", "rl": {"enabled": False}},
        "brokers": {},
    }
    with patch("app.agents.trader.build_market_cache_config"), \
         patch("app.agents.trader.build_market_cache"):
        agent = TradingAgent(broker, cfg)
    return agent


def test_no_backoff_initially():
    agent = _make_agent()
    assert agent._should_skip_exit("broker1", "AAPL") is False


def test_backoff_activates_after_failure():
    agent = _make_agent()
    agent._record_exit_failure("broker1", "AAPL")
    assert agent._should_skip_exit("broker1", "AAPL") is True


def test_backoff_exponential():
    agent = _make_agent()
    # First failure: 1 min
    agent._record_exit_failure("broker1", "AAPL")
    until1 = agent._exit_backoff_until[("broker1", "AAPL")]

    # Second failure: 2 min
    agent._record_exit_failure("broker1", "AAPL")
    until2 = agent._exit_backoff_until[("broker1", "AAPL")]

    # Third failure: 4 min
    agent._record_exit_failure("broker1", "AAPL")
    until3 = agent._exit_backoff_until[("broker1", "AAPL")]

    assert agent._exit_fail_counts[("broker1", "AAPL")] == 3
    # Each backoff should be longer than the previous
    assert until2 > until1
    assert until3 > until2


def test_backoff_caps_at_15_minutes():
    agent = _make_agent()
    for _ in range(10):
        agent._record_exit_failure("broker1", "AAPL")
    until = agent._exit_backoff_until[("broker1", "AAPL")]
    now = datetime.now(timezone.utc)
    # Should cap at 15 minutes, not grow beyond
    assert (until - now).total_seconds() <= 15 * 60 + 1


def test_clear_backoff():
    agent = _make_agent()
    agent._record_exit_failure("broker1", "AAPL")
    assert agent._should_skip_exit("broker1", "AAPL") is True
    agent._clear_exit_backoff("broker1", "AAPL")
    assert agent._should_skip_exit("broker1", "AAPL") is False
    assert ("broker1", "AAPL") not in agent._exit_fail_counts
    assert ("broker1", "AAPL") not in agent._exit_backoff_until


def test_backoff_expires():
    agent = _make_agent()
    agent._exit_fail_counts[("broker1", "AAPL")] = 1
    # Set backoff to the past
    agent._exit_backoff_until[("broker1", "AAPL")] = datetime.now(timezone.utc) - timedelta(seconds=1)
    assert agent._should_skip_exit("broker1", "AAPL") is False
    # State should be cleaned up
    assert ("broker1", "AAPL") not in agent._exit_fail_counts
    assert ("broker1", "AAPL") not in agent._exit_backoff_until


def test_different_symbols_independent():
    agent = _make_agent()
    agent._record_exit_failure("broker1", "AAPL")
    assert agent._should_skip_exit("broker1", "AAPL") is True
    assert agent._should_skip_exit("broker1", "TSLA") is False


def test_flush_clears_backoff_on_sell_completed():
    agent = _make_agent()
    agent._record_exit_failure("broker1", "AAPL")
    assert agent._should_skip_exit("broker1", "AAPL") is True

    resp = OrderResponse(
        symbol="AAPL", broker="broker1", status="completed",
        order_id="123", side="sell", qty=10,
    )
    queue = MagicMock()
    queue.pop_responses.return_value = [resp]
    agent._order_queues = {"broker1": queue}
    agent._orchestrator = MagicMock()
    agent._live_reward_tracker = None
    agent._flush_order_responses()
    assert agent._should_skip_exit("broker1", "AAPL") is False


def test_flush_records_failure_on_sell_rejected():
    agent = _make_agent()
    resp = OrderResponse(
        symbol="TSLA", broker="broker1", status="rejected",
        order_id=None, side="sell", qty=5,
    )
    queue = MagicMock()
    queue.pop_responses.return_value = [resp]
    agent._order_queues = {"broker1": queue}
    agent._orchestrator = MagicMock()
    agent._live_reward_tracker = None
    agent._flush_order_responses()
    assert agent._should_skip_exit("broker1", "TSLA") is True
