# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

tf = __import__("pytest").importorskip("tensorflow")

import threading
from unittest.mock import MagicMock, patch
from app.execution.order_queue import OrderResponse


def _make_agent():
    from app.agents.trader import TradingAgent
    broker = MagicMock()
    broker.get_account.return_value = {"equity": 100000, "cash": 50000}
    broker.get_positions.return_value = []
    cfg = {
        "strategies": {"active": ["trend_following"], "combine": "first", "params": {}},
        "risk": {"enabled": False, "max_portfolio_leverage": 1.5},
        "data": {},
        "learning": {},
        "orchestrator": {"mode": "select", "rl": {"enabled": False}},
        "brokers": {},
    }
    with patch("app.agents.trader.build_market_cache_config"), \
         patch("app.agents.trader.build_market_cache"):
        agent = TradingAgent(broker, cfg)
    return agent


def test_reserve_notional_under_limit():
    agent = _make_agent()
    portfolio = {"equity": 100000, "gross_exposure": 100000}
    # 100k gross + 10k new = 110k / 100k = 1.1x < 1.5x -> allowed
    assert agent._check_and_reserve_notional("test", 10000, portfolio) is True
    assert agent._pending_notional["test"] == 10000


def test_reserve_notional_over_limit():
    agent = _make_agent()
    portfolio = {"equity": 100000, "gross_exposure": 140000}
    # 140k gross + 20k new = 160k / 100k = 1.6x > 1.5x -> rejected
    assert agent._check_and_reserve_notional("test", 20000, portfolio) is False
    assert agent._pending_notional.get("test", 0) == 0


def test_reserve_accounts_for_pending():
    agent = _make_agent()
    portfolio = {"equity": 100000, "gross_exposure": 100000}
    # First: 100k + 40k = 140k / 100k = 1.4x -> ok
    assert agent._check_and_reserve_notional("test", 40000, portfolio) is True
    # Second: 100k + 40k(pending) + 20k(new) = 160k / 100k = 1.6x -> rejected
    assert agent._check_and_reserve_notional("test", 20000, portfolio) is False


def test_release_notional():
    agent = _make_agent()
    agent._pending_notional["test"] = 50000
    agent._release_pending_notional("test", 20000)
    assert agent._pending_notional["test"] == 30000


def test_release_notional_floor_zero():
    agent = _make_agent()
    agent._pending_notional["test"] = 5000
    agent._release_pending_notional("test", 99999)
    assert agent._pending_notional["test"] == 0.0


def test_reserve_thread_safety():
    agent = _make_agent()
    portfolio = {"equity": 100000, "gross_exposure": 0}
    results = []

    def reserve():
        ok = agent._check_and_reserve_notional("test", 30000, portfolio)
        results.append(ok)

    threads = [threading.Thread(target=reserve) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # At most 5 should succeed (5 * 30k = 150k = 1.5x)
    accepted = sum(1 for r in results if r)
    assert accepted == 5
    assert agent._pending_notional["test"] == 150000


def test_flush_releases_pending_notional_on_buy_completed():
    agent = _make_agent()
    agent._pending_notional["broker1"] = 15000
    resp = OrderResponse(
        symbol="AAPL", broker="broker1", status="completed",
        order_id="123", side="buy", qty=100, filled_avg_price=150.0,
    )
    queue = MagicMock()
    queue.pop_responses.return_value = [resp]
    agent._order_queues = {"broker1": queue}
    agent._orchestrator = MagicMock()
    agent._live_reward_tracker = None
    agent._flush_order_responses()
    assert agent._pending_notional["broker1"] == 0.0


def test_flush_releases_pending_notional_on_buy_rejected():
    agent = _make_agent()
    agent._pending_notional["broker1"] = 10000
    resp = OrderResponse(
        symbol="AAPL", broker="broker1", status="rejected",
        order_id=None, side="buy", qty=100, filled_avg_price=100.0,
    )
    queue = MagicMock()
    queue.pop_responses.return_value = [resp]
    agent._order_queues = {"broker1": queue}
    agent._orchestrator = MagicMock()
    agent._live_reward_tracker = None
    agent._flush_order_responses()
    assert agent._pending_notional["broker1"] == 0.0


def test_zero_equity_rejects():
    agent = _make_agent()
    portfolio = {"equity": 0, "gross_exposure": 0}
    assert agent._check_and_reserve_notional("test", 1000, portfolio) is False
