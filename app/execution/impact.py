# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ImpactEstimate:
    impact_bps: float
    participation: float
    volatility_pct: float


def estimate_market_impact(notional: float, price: float, market_state: dict, cfg: dict | None = None) -> ImpactEstimate:
    cfg = cfg or {}
    if notional <= 0 or price <= 0:
        return ImpactEstimate(impact_bps=0.0, participation=0.0, volatility_pct=0.0)
    session_volume = float(market_state.get("session_volume", 0.0) or 0.0)
    dollar_volume = session_volume * price
    participation = (notional / dollar_volume) if dollar_volume > 0 else 0.0
    volatility_pct = _realized_volatility_pct(market_state)

    base_bps = float(cfg.get("base_bps", 1.0))
    volume_scale_bps = float(cfg.get("volume_scale_bps", 50.0))
    min_vol_pct = float(cfg.get("min_vol_pct", 0.5))
    vol = max(volatility_pct, min_vol_pct)
    impact = base_bps + participation * volume_scale_bps * (vol / 100.0)
    return ImpactEstimate(impact_bps=impact, participation=participation, volatility_pct=volatility_pct)


def _realized_volatility_pct(market_state: dict) -> float:
    prices = market_state.get("prices", []) or []
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
    return (var**0.5) * 100.0
