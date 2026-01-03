# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import hashlib
from typing import Any


def normalize_broker_name(broker_name: str, broker_names: list[str]) -> str:
    if broker_name in broker_names:
        return broker_name
    base = broker_name.split(":", 1)[0]
    matches = [name for name in broker_names if name == base or name.startswith(f"{base}:")]
    if matches:
        return matches[0]
    return broker_names[0] if broker_names else broker_name


def auto_split_broker(symbol: str, broker_names: list[str]) -> str:
    if not broker_names:
        return ""
    digest = hashlib.md5(symbol.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(broker_names)
    return broker_names[idx]


def resolve_broker_for_symbol(
    symbol: str,
    broker_names: list[str],
    routing_cfg: dict[str, Any] | None,
    selected_strategies: list[str] | None = None,
    signals: list[dict[str, Any]] | None = None,
    action_strategy: str | None = None,
) -> str:
    if len(broker_names) <= 1:
        return broker_names[0] if broker_names else ""
    routing = routing_cfg or {}
    symbol_map = routing.get("symbols", {}) or {}
    if symbol in symbol_map:
        return normalize_broker_name(str(symbol_map[symbol]), broker_names)
    strategy_map = routing.get("strategies", {}) or {}
    strategy = action_strategy
    if not strategy and selected_strategies:
        strategy = selected_strategies[0]
    if not strategy and signals:
        strategy = signals[0].get("name")
    if strategy and strategy in strategy_map:
        return normalize_broker_name(str(strategy_map[strategy]), broker_names)
    if str(routing.get("mode", "default")).lower() == "auto_split":
        return auto_split_broker(symbol, broker_names)
    default = routing.get("default")
    if default:
        return normalize_broker_name(str(default), broker_names)
    return broker_names[0] if broker_names else ""


def partition_symbols(
    symbols: list[str],
    broker_names: list[str],
    routing_cfg: dict[str, Any] | None,
) -> dict[str, list[str]]:
    routing = routing_cfg or {}
    mode = str(routing.get("mode", "default")).lower()
    if mode != "auto_split" or len(broker_names) <= 1:
        return {"": list(symbols)}
    buckets: dict[str, list[str]] = {}
    for symbol in symbols:
        broker_name = auto_split_broker(symbol, broker_names)
        buckets.setdefault(broker_name, []).append(symbol)
    return buckets
