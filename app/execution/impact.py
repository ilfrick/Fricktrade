# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from dataclasses import dataclass

from app.utils.volatility import realized_volatility_pct


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
    prices = market_state.get("prices", []) or []
    volatility_pct = realized_volatility_pct(prices)

    base_bps = float(cfg.get("base_bps", 1.0))
    volume_scale_bps = float(cfg.get("volume_scale_bps", 50.0))
    min_vol_pct = float(cfg.get("min_vol_pct", 0.5))
    vol = max(volatility_pct, min_vol_pct)
    impact = base_bps + participation * volume_scale_bps * (vol / 100.0)
    return ImpactEstimate(impact_bps=impact, participation=participation, volatility_pct=volatility_pct)
