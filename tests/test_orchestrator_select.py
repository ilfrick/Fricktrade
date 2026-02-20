# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

tf = __import__("pytest").importorskip("tensorflow")

from unittest.mock import patch, MagicMock

from app.agents.orchestrator import RLStrategyOrchestrator


def _make_orchestrator(mode="select", enabled=True, top_k=2, min_score=0.0):
    cfg = {
        "orchestrator": {
            "mode": mode,
            "top_k": top_k,
            "min_score": min_score,
            "rl": {"enabled": enabled, "model_type": "lstm", "device": "cpu"},
        }
    }
    return RLStrategyOrchestrator(cfg)


def test_select_disabled_returns_all():
    orch = _make_orchestrator(enabled=False)
    names, weights = orch.select("AAPL", ["s1", "s2"], {})
    assert names == ["s1", "s2"]
    assert weights == {"s1": 1.0, "s2": 1.0}


def test_select_empty_strategies():
    orch = _make_orchestrator()
    names, weights = orch.select("AAPL", [], {})
    assert names == []
    assert weights == {}


def test_select_single_strategy():
    orch = _make_orchestrator()
    with patch.object(orch, "_predict", return_value={"s1": 0.9}):
        with patch.object(orch, "_ensure_model"):
            names, weights = orch.select("AAPL", ["s1"], {})
    assert names == ["s1"]
    assert weights == {"s1": 1.0}


def test_select_direct_mode():
    orch = _make_orchestrator(mode="direct", min_score=0.3)
    probs = {"s1": 0.8, "s2": 0.5, "s3": 0.1}
    with patch.object(orch, "_predict", return_value=probs):
        with patch.object(orch, "_ensure_model"):
            # Avoid epsilon exploration so direct mode remains deterministic.
            with patch("random.random", return_value=1.0):
                names, weights = orch.select("AAPL", ["s1", "s2", "s3"], {})
    assert len(names) == 1
    assert names[0] == "s1"


def test_select_weight_mode():
    orch = _make_orchestrator(mode="weight", top_k=2)
    probs = {"s1": 0.8, "s2": 0.5, "s3": 0.1}
    with patch.object(orch, "_predict", return_value=probs):
        with patch.object(orch, "_ensure_model"):
            # Force no epsilon exploration
            with patch("random.random", return_value=1.0):
                names, weights = orch.select("AAPL", ["s1", "s2", "s3"], {})
    assert "s1" in names
    assert weights.get("s1", 0) >= weights.get("s2", 0)
