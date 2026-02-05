# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""
Execution quality and smart order routing.

This module provides:
- Smart order routing with market impact minimization
- Execution algorithms (TWAP, VWAP, POV)
- Transaction cost analysis (TCA)
- Almgren-Chriss market impact modeling

Usage:
    from app.execution import SmartOrderRouter, TCAAnalyzer

    # Route an order
    router = SmartOrderRouter()
    decision = router.route(OrderContext(
        symbol="AAPL",
        side="buy",
        qty=100,
        price=150.0,
        urgency=0.5,
        market_state={"session_volume": 1000000},
    ))

    # Analyze execution costs
    analyzer = TCAAnalyzer()
    metrics = analyzer.analyze_fills(fills, decision_prices)
"""

from app.execution.algos import AlgoSlice, twap_slices, vwap_slices, pov_slices
from app.execution.impact import ImpactEstimate, estimate_market_impact
from app.execution.smart_router import (
    OrderContext,
    RoutingDecision,
    SmartOrderRouter,
    AlmgrenChrissParams,
    almgren_chriss_optimal_trajectory,
    estimate_almgren_chriss_cost,
)
from app.execution.tca import (
    Fill,
    TCAMetrics,
    TCAReport,
    TCAAnalyzer,
    compute_vwap_slippage,
)

__all__ = [
    # Algorithms
    "AlgoSlice",
    "twap_slices",
    "vwap_slices",
    "pov_slices",
    # Impact
    "ImpactEstimate",
    "estimate_market_impact",
    # Smart routing
    "OrderContext",
    "RoutingDecision",
    "SmartOrderRouter",
    "AlmgrenChrissParams",
    "almgren_chriss_optimal_trajectory",
    "estimate_almgren_chriss_cost",
    # TCA
    "Fill",
    "TCAMetrics",
    "TCAReport",
    "TCAAnalyzer",
    "compute_vwap_slippage",
]
