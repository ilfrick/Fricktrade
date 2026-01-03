# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from app.execution import routing


def test_normalize_broker_name_prefers_exact_match():
    broker_names = ["alpaca:Realistic", "alpaca:Higher"]
    assert routing.normalize_broker_name("alpaca:Higher", broker_names) == "alpaca:Higher"


def test_normalize_broker_name_falls_back_to_base():
    broker_names = ["alpaca:Realistic", "alpaca:Higher"]
    assert routing.normalize_broker_name("alpaca", broker_names) == "alpaca:Realistic"


def test_auto_split_is_deterministic():
    broker_names = ["alpaca:Realistic", "alpaca:Higher"]
    first = routing.auto_split_broker("AAPL", broker_names)
    second = routing.auto_split_broker("AAPL", broker_names)
    assert first == second
    assert first in broker_names


def test_resolve_uses_symbol_map_over_default():
    broker_names = ["alpaca:Realistic", "alpaca:Higher"]
    cfg = {"symbols": {"AAPL": "alpaca:Higher"}, "default": "alpaca:Realistic"}
    assert routing.resolve_broker_for_symbol("AAPL", broker_names, cfg) == "alpaca:Higher"


def test_resolve_uses_strategy_map():
    broker_names = ["alpaca:Realistic", "alpaca:Higher"]
    cfg = {"strategies": {"rl_policy": "alpaca:Higher"}}
    assert routing.resolve_broker_for_symbol(
        "MSFT",
        broker_names,
        cfg,
        selected_strategies=["rl_policy"],
    ) == "alpaca:Higher"


def test_resolve_uses_auto_split():
    broker_names = ["alpaca:Realistic", "alpaca:Higher"]
    cfg = {"mode": "auto_split"}
    result = routing.resolve_broker_for_symbol("NVDA", broker_names, cfg)
    assert result in broker_names


def test_partition_symbols_auto_split():
    broker_names = ["alpaca:Realistic", "alpaca:Higher"]
    symbols = ["AAPL", "MSFT", "NVDA"]
    buckets = routing.partition_symbols(symbols, broker_names, {"mode": "auto_split"})
    assert set(sum(buckets.values(), [])) == set(symbols)
