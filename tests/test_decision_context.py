# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from app.agents.decision_context import DecisionContext


def test_decision_context_defaults_are_isolated():
    first = DecisionContext(symbol="AAA")
    second = DecisionContext(symbol="BBB")
    first.market_state["foo"] = "bar"
    first.signals.append({"name": "s1"})

    assert "foo" not in second.market_state
    assert second.signals == []


def test_decision_context_snapshot():
    ctx = DecisionContext(symbol="AAA", broker="alpaca:Realistic", action="buy", reason="ok")
    snapshot = ctx.snapshot()

    assert snapshot["symbol"] == "AAA"
    assert snapshot["broker"] == "alpaca:Realistic"
    assert snapshot["action"] == "buy"
    assert snapshot["reason"] == "ok"
