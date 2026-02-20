# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from app.agents.strategy_config import StrategyConfig


def test_defaults():
    cfg = StrategyConfig()
    assert cfg.name == "intraday_momentum"
    assert cfg.names == ["intraday_momentum"]
    assert cfg.combine == "priority"
    assert cfg.params == {}
    assert cfg.fee_aware == {}
    assert cfg.signal_bias_guard == {}
    assert cfg.performance == {}
    assert cfg.min_conviction == 0.0
    assert cfg.single_sided_conviction_multiplier == 1.0


def test_from_dict_roundtrip():
    d = {
        "name": "trend_following",
        "names": ["trend_following", "factor_model"],
        "combine": "vote",
        "params": {"lookback_minutes": 30},
        "fee_aware": {"enabled": True},
        "signal_bias_guard": {"threshold": 0.3},
        "performance": {"track": True},
        "min_conviction": 0.3,
        "single_sided_conviction_multiplier": 0.5,
    }
    cfg = StrategyConfig.from_dict(d)
    assert cfg.name == "trend_following"
    assert cfg.names == ["trend_following", "factor_model"]
    assert cfg.combine == "vote"
    assert cfg.params == {"lookback_minutes": 30}
    assert cfg.min_conviction == 0.3
    assert cfg.single_sided_conviction_multiplier == 0.5
    assert cfg.to_dict() == d


def test_unknown_keys_ignored():
    d = {"name": "test", "unknown_key": 42, "another": "value"}
    cfg = StrategyConfig.from_dict(d)
    assert cfg.name == "test"
    assert cfg.combine == "priority"


def test_empty_dict():
    cfg = StrategyConfig.from_dict({})
    assert cfg == StrategyConfig()
