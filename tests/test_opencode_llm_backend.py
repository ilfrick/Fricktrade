# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import pytest

from app.llm.client import LLMClient, LLMResponse, LLMUsageTracker, OpenCodeBackend


def test_opencode_backend_uses_stateless_model_reference_and_estimates_usage(monkeypatch):
    backend = OpenCodeBackend(allow_chatgpt_plan=True)
    calls = []

    def request(path, payload=None):
        calls.append((path, payload))
        return {"data": {"text": '{"score": 1}'}} if payload else {"data": []}

    monkeypatch.setattr(backend, "_request", request)
    monkeypatch.setattr(backend, "_prices", lambda: {})
    result = backend.complete("JSON only", "analyze this", model="openai/gpt-test")

    assert calls[0][0] == "/api/experimental/generate/text"
    assert calls[0][1]["model"] == {"providerID": "openai", "id": "gpt-test"}
    assert "JSON only" in calls[0][1]["prompt"]
    assert "analyze this" in calls[0][1]["prompt"]
    assert result.content == '{"score": 1}'
    assert result.input_tokens > 0 and result.output_tokens > 0


def test_opencode_openai_route_requires_explicit_background_consent(monkeypatch):
    backend = OpenCodeBackend(allow_chatgpt_plan=False)
    monkeypatch.setattr(backend, "_request", lambda *_args, **_kwargs: pytest.fail("request must be blocked"))

    with pytest.raises(RuntimeError, match="explicit"):
        backend.complete("system", "user", model="openai/gpt-test")


def test_llm_client_routes_by_usage_and_preserves_fallback(monkeypatch):
    class FakeBackend:
        def __init__(self):
            self.model_seen = None

        def complete(self, *_args, model=None):
            self.model_seen = model
            return LLMResponse("ok", model or "default", 1, 1, 1.0, 0.0)

    fake = FakeBackend()
    monkeypatch.setitem(LLMClient.BACKENDS, "fake", lambda: fake)
    client = LLMClient(routes={
        "sentiment": {"backend": "fake", "model": "catalog/model"},
    })
    response = client.complete(
        backend="ollama", system_prompt="s", user_prompt="u",
        usage="sentiment", retries=1,
    )
    assert fake.model_seen == "catalog/model"
    assert response.model == "catalog/model"

    client2 = LLMClient()
    monkeypatch.setattr(client2, "_get_backend", lambda _name: fake)
    fake.model_seen = None
    client2._trackers["ollama"] = LLMUsageTracker()
    client2.complete(
        backend="ollama", system_prompt="s", user_prompt="u", usage="unconfigured",
        retries=1,
    )
    assert fake.model_seen is None
