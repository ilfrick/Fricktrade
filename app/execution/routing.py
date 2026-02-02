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


def parallel_partition_symbols(
    symbols: list[str],
    broker_names: list[str],
    broker_buying_power: dict[str, float],
    routing_cfg: dict[str, Any] | None,
    min_symbols: int = 1,
) -> dict[str, list[str]]:
    """
    Partition symbols for parallel routing where each broker gets symbols
    proportional to their buying power.

    Args:
        symbols: Ranked list of symbols (best first from AI filter)
        broker_names: List of broker names
        broker_buying_power: Dict mapping broker name to buying power
        routing_cfg: Routing configuration
        min_symbols: Minimum symbols per broker (default 1)

    Returns:
        Dict mapping broker name to list of symbols they should trade
    """
    if not broker_names or not symbols:
        return {}

    routing = routing_cfg or {}
    scaling_enabled = bool(routing.get("buying_power_scaling", True))

    if not scaling_enabled or len(broker_names) == 1:
        # No scaling - all brokers get all symbols
        return {name: list(symbols) for name in broker_names}

    # Calculate total buying power
    total_bp = sum(broker_buying_power.get(name, 0.0) for name in broker_names)
    if total_bp <= 0:
        # Fallback: equal distribution
        return {name: list(symbols) for name in broker_names}

    # Calculate symbol count for each broker based on buying power ratio
    buckets: dict[str, list[str]] = {}
    for name in broker_names:
        bp = broker_buying_power.get(name, 0.0)
        if bp <= 0:
            buckets[name] = symbols[:min_symbols] if symbols else []
            continue

        # Ratio of this broker's BP to total
        ratio = bp / total_bp
        # Number of symbols proportional to BP, but at least min_symbols
        symbol_count = max(min_symbols, int(len(symbols) * ratio))
        # Cap at total symbols
        symbol_count = min(symbol_count, len(symbols))
        # Broker gets top N symbols from the ranked list
        buckets[name] = symbols[:symbol_count]

    return buckets


def calculate_broker_symbol_limits(
    broker_names: list[str],
    broker_buying_power: dict[str, float],
    total_symbols: int,
    min_symbols: int = 1,
    max_symbols: int | None = None,
) -> dict[str, int]:
    """
    Calculate how many symbols each broker should process based on buying power.

    Args:
        broker_names: List of broker names
        broker_buying_power: Dict mapping broker name to buying power
        total_symbols: Total number of available symbols
        min_symbols: Minimum symbols per broker
        max_symbols: Maximum symbols per broker (None = no limit)

    Returns:
        Dict mapping broker name to number of symbols they should process
    """
    if not broker_names:
        return {}

    total_bp = sum(broker_buying_power.get(name, 0.0) for name in broker_names)
    if total_bp <= 0:
        # Equal distribution
        count = max(min_symbols, total_symbols // len(broker_names))
        if max_symbols:
            count = min(count, max_symbols)
        return {name: count for name in broker_names}

    limits: dict[str, int] = {}
    for name in broker_names:
        bp = broker_buying_power.get(name, 0.0)
        if bp <= 0:
            limits[name] = min_symbols
            continue

        ratio = bp / total_bp
        count = max(min_symbols, int(total_symbols * ratio))
        count = min(count, total_symbols)
        if max_symbols:
            count = min(count, max_symbols)
        limits[name] = count

    return limits
