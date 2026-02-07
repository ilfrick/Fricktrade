# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from app.execution.config import ExecutionConfig


def test_defaults():
    cfg = ExecutionConfig()
    assert cfg.routing == {}
    assert cfg.retry == {}
    assert cfg.open_orders == {}
    assert cfg.algos == {}
    assert cfg.impact == {}


def test_from_dict_with_nested_routing():
    d = {
        "brokers": {"routing": {"default": "alpaca"}},
        "retry": {"max_retries": 3},
        "open_orders": {"missing_grace_seconds": 60},
        "algos": {"enabled": True, "default": "twap"},
        "impact": {"model": "linear"},
    }
    cfg = ExecutionConfig.from_dict(d)
    assert cfg.routing == {"default": "alpaca"}
    assert cfg.retry == {"max_retries": 3}
    assert cfg.open_orders == {"missing_grace_seconds": 60}
    assert cfg.algos == {"enabled": True, "default": "twap"}
    assert cfg.impact == {"model": "linear"}


def test_roundtrip():
    cfg = ExecutionConfig(
        routing={"default": "ib"},
        retry={"max_retries": 2},
    )
    d = cfg.to_dict()
    assert d["routing"] == {"default": "ib"}
    assert d["retry"] == {"max_retries": 2}
    assert d["open_orders"] == {}


def test_empty_dict():
    cfg = ExecutionConfig.from_dict({})
    assert cfg == ExecutionConfig()


def test_unknown_keys_ignored():
    d = {"unknown_section": {"a": 1}, "retry": {"x": 1}}
    cfg = ExecutionConfig.from_dict(d)
    assert cfg.retry == {"x": 1}
    assert cfg.routing == {}
