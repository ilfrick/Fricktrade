# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from app.llm.ollama_sentiment import AggregateSentimentAnalyzer, OllamaSentimentAnalyzer


class _FakeClient:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_aggregate_sentiment_uses_selected_shared_provider():
    client = _FakeClient({
        "score": 1.7, "bias": "INVALID", "key_theme": "risk on", "confidence": -0.2,
    })
    analyzer = AggregateSentimentAnalyzer(
        client, backend="gemini", model="gemini-2.5-flash", max_headlines=1,
    )
    result = analyzer.analyze(["headline one", "headline two"])

    assert client.calls[0]["backend"] == "gemini"
    assert client.calls[0]["model"] == "gemini-2.5-flash"
    assert client.calls[0]["retries"] == 1
    assert result.score == 1.0
    assert result.bias == "neutral"
    assert result.confidence == 0.0
    assert "headline one" in client.calls[0]["user_prompt"]
    assert "headline two" not in client.calls[0]["user_prompt"]


def test_failure_returns_stale_value_and_legacy_name_remains():
    client = _FakeClient({"score": -0.4, "bias": "bearish", "confidence": 0.8})
    analyzer = OllamaSentimentAnalyzer(client, backend="ollama")
    first = analyzer.analyze(["headline"])
    client.result = RuntimeError("provider unavailable")
    stale = analyzer.analyze(["new headline"])
    assert stale is first
    assert stale.score == -0.4


def test_empty_headlines_do_not_make_a_provider_call():
    client = _FakeClient({})
    analyzer = AggregateSentimentAnalyzer(client)
    analyzer.analyze([])
    assert client.calls == []
