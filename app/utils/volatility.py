# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick


def realized_volatility_pct(prices: list[float] | tuple[float, ...]) -> float:
    """Realized volatility from price series, as percentage."""
    if len(prices) < 3:
        return 0.0
    returns = []
    for idx in range(1, len(prices)):
        prev = prices[idx - 1]
        curr = prices[idx]
        if not prev:
            continue
        returns.append((curr - prev) / prev)
    if not returns:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / max(len(returns) - 1, 1)
    return (var ** 0.5) * 100.0
