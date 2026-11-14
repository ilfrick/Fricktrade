# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import yaml
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from app.api import server


def _write_config(path):
    routes = {
        use: {"backend": "", "model": ""}
        for use in server._LLM_USES
    }
    path.write_text(yaml.safe_dump({
        "api": {"auth": {"enabled": True, "token_env": "TEST_TOKEN"}},
        "strategy": {"alpha_gate": {"enabled": True}},
        "llm": {
            "routes": routes,
            "chatgpt_plan_background_consent": False,
        },
    }), encoding="utf-8")


def _client(monkeypatch, tmp_path):
    path = tmp_path / "config.yaml"
    _write_config(path)
    monkeypatch.setattr(server, "CONFIG_PATH", path)
    monkeypatch.setenv("TEST_TOKEN", "test-token")
    return TestClient(server.app), path


def test_llm_settings_save_validates_route_and_keeps_alpha_gate(monkeypatch, tmp_path):
    client, path = _client(monkeypatch, tmp_path)
    routes = {use: {"backend": "", "model": ""} for use in server._LLM_USES}
    routes["sentiment"] = {"backend": "opencode", "model": "openrouter/openai/gpt-test"}
    response = client.put(
        "/llm/settings",
        headers={"X-API-Key": "test-token"},
        json={"routes": routes, "chatgpt_plan_background_consent": True},
    )
    assert response.status_code == 200
    saved = yaml.safe_load(path.read_text())
    assert saved["llm"]["routes"]["sentiment"]["model"] == "openrouter/openai/gpt-test"
    assert saved["strategy"]["alpha_gate"]["enabled"] is True


def test_llm_settings_rejects_missing_model_for_opencode(monkeypatch, tmp_path):
    client, _ = _client(monkeypatch, tmp_path)
    routes = {use: {"backend": "", "model": ""} for use in server._LLM_USES}
    routes["sentiment"] = {"backend": "opencode", "model": "not-a-model"}
    response = client.put(
        "/llm/settings",
        headers={"X-API-Key": "test-token"},
        json={"routes": routes, "chatgpt_plan_background_consent": False},
    )
    assert response.status_code == 400


def test_catalog_does_not_return_auth_form_defaults(monkeypatch, tmp_path):
    client, _ = _client(monkeypatch, tmp_path)
    responses = iter([
        (200, {"data": [{
            "id": "provider", "name": "Provider",
            "connections": [{"type": "credential", "method": "oauth", "label": "Main",
                             "status": {"status": "connected", "message": "ok"}}],
            "methods": [{"id": "oauth", "type": "oauth", "label": "OAuth",
                         "form": [{"type": "string", "name": "client_secret",
                                   "label": "Client secret", "default": "hidden-value"}]}],
        }]}),
        (200, {"data": []}),
        (200, {"data": [{"id": "m", "providerID": "provider", "name": "Model",
                         "type": "chat", "enabled": True}]}),
    ])
    monkeypatch.setattr(server, "_opencode_request", lambda *_a, **_kw: next(responses))
    response = client.get("/llm/catalog", headers={"X-API-Key": "test-token"})
    assert response.status_code == 200
    assert "hidden-value" not in response.text
    assert "default" not in response.text


def test_api_key_connection_is_not_echoed_or_written_to_config(monkeypatch, tmp_path):
    client, path = _client(monkeypatch, tmp_path)
    seen = {}

    def request(method, route, payload=None):
        seen.update(method=method, route=route, payload=payload)
        return 204, {}

    monkeypatch.setattr(server, "_opencode_request", request)
    key = "super-secret-provider-key"
    response = client.post(
        "/llm/integrations/openai/key",
        headers={"X-API-Key": "test-token"},
        json={"key": key},
    )
    assert response.status_code == 200
    assert key not in response.text
    assert key not in path.read_text()
    assert seen["payload"] == {"key": key}
