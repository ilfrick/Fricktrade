# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import yaml
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from app.api import server


def _write_cfg(path, *, auth_enabled: bool) -> None:
    cfg = {
        "api": {"auth": {"enabled": auth_enabled, "token_env": "TEST_TOKEN"}},
        "brokers": {"alpaca": {"api_key": "k", "api_secret": "s", "accounts": []}},
        "data": {"sources": []},
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def test_api_auth_required(tmp_path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_cfg(cfg_path, auth_enabled=True)
    monkeypatch.setenv("TEST_TOKEN", "secret")
    monkeypatch.setattr(server, "CONFIG_PATH", cfg_path)
    client = TestClient(server.app)

    assert client.get("/config/raw").status_code == 401
    ok = client.get("/config/raw", headers={"X-API-Key": "secret"})
    assert ok.status_code == 200


def test_api_auth_disabled(tmp_path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_cfg(cfg_path, auth_enabled=False)
    monkeypatch.delenv("TEST_TOKEN", raising=False)
    monkeypatch.setattr(server, "CONFIG_PATH", cfg_path)
    client = TestClient(server.app)

    assert client.get("/config/raw").status_code == 200


def test_merge_secrets_by_name() -> None:
    target = {
        "brokers": {
            "alpaca": {
                "accounts": [
                    {"name": "secondary", "api_key": "***", "api_secret": "***"},
                    {"name": "primary", "api_key": "***", "api_secret": "***"},
                ]
            }
        }
    }
    source = {
        "brokers": {
            "alpaca": {
                "accounts": [
                    {"name": "primary", "api_key": "k1", "api_secret": "s1"},
                    {"name": "secondary", "api_key": "k2", "api_secret": "s2"},
                ]
            }
        }
    }

    server._merge_secrets(target, source)

    accounts = target["brokers"]["alpaca"]["accounts"]
    assert accounts[0]["api_key"] == "k2"
    assert accounts[0]["api_secret"] == "s2"
    assert accounts[1]["api_key"] == "k1"
    assert accounts[1]["api_secret"] == "s1"
