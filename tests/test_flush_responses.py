# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

tf = __import__("pytest").importorskip("tensorflow")

from unittest.mock import MagicMock, patch
from app.execution.order_queue import OrderResponse


def _make_agent():
    """Create a minimal TradingAgent for testing _flush_order_responses."""
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


def test_flush_responses_dispatches_to_orchestrator():
    agent = _make_agent()
    resp = OrderResponse(
        symbol="AAPL", broker="test", status="filled",
        order_id="123", side="buy", qty=10, price=150.0,
    )
    queue = MagicMock()
    queue.pop_responses.return_value = [resp]
    agent._order_queues = {"test": queue}
    agent._orchestrator = MagicMock()
    agent._live_reward_tracker = None
    agent._flush_order_responses()
    agent._orchestrator.on_order_update.assert_called_once()


def test_flush_responses_handles_orchestrator_error():
    agent = _make_agent()
    resp = OrderResponse(
        symbol="AAPL", broker="test", status="filled",
        order_id="123", side="buy", qty=10, price=150.0,
    )
    queue = MagicMock()
    queue.pop_responses.return_value = [resp]
    agent._order_queues = {"test": queue}
    agent._orchestrator = MagicMock()
    agent._orchestrator.on_order_update.side_effect = ValueError("test error")
    agent._live_reward_tracker = None
    # Should not raise — error is caught
    agent._flush_order_responses()


def test_flush_responses_empty_queue():
    agent = _make_agent()
    queue = MagicMock()
    queue.pop_responses.return_value = []
    agent._order_queues = {"test": queue}
    agent._flush_order_responses()
