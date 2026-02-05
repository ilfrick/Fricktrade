# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""
Transaction Cost Analysis (TCA) for execution quality measurement.

Measures and analyzes execution costs including:
- Slippage
- Market impact
- Timing costs
- Opportunity costs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Fill:
    """Represents a single order fill."""
    symbol: str
    side: str  # "buy" or "sell"
    qty: int
    price: float
    timestamp: datetime
    order_id: str = ""
    venue: str = ""


@dataclass
class TCAMetrics:
    """Transaction cost metrics for a trade or group of trades."""
    # Slippage metrics
    slippage_bps: float = 0.0
    implementation_shortfall_bps: float = 0.0

    # Market impact
    realized_impact_bps: float = 0.0
    temporary_impact_bps: float = 0.0
    permanent_impact_bps: float = 0.0

    # Timing
    timing_cost_bps: float = 0.0
    delay_cost_bps: float = 0.0

    # Spread
    spread_capture_pct: float = 0.0  # How much of spread we captured (negative = paid)

    # Summary
    total_cost_bps: float = 0.0
    notional: float = 0.0
    n_fills: int = 0

    def __add__(self, other: "TCAMetrics") -> "TCAMetrics":
        """Combine TCA metrics (weighted by notional)."""
        total_notional = self.notional + other.notional
        if total_notional == 0:
            return TCAMetrics()

        w1 = self.notional / total_notional
        w2 = other.notional / total_notional

        return TCAMetrics(
            slippage_bps=w1 * self.slippage_bps + w2 * other.slippage_bps,
            implementation_shortfall_bps=w1 * self.implementation_shortfall_bps + w2 * other.implementation_shortfall_bps,
            realized_impact_bps=w1 * self.realized_impact_bps + w2 * other.realized_impact_bps,
            temporary_impact_bps=w1 * self.temporary_impact_bps + w2 * other.temporary_impact_bps,
            permanent_impact_bps=w1 * self.permanent_impact_bps + w2 * other.permanent_impact_bps,
            timing_cost_bps=w1 * self.timing_cost_bps + w2 * other.timing_cost_bps,
            delay_cost_bps=w1 * self.delay_cost_bps + w2 * other.delay_cost_bps,
            spread_capture_pct=w1 * self.spread_capture_pct + w2 * other.spread_capture_pct,
            total_cost_bps=w1 * self.total_cost_bps + w2 * other.total_cost_bps,
            notional=total_notional,
            n_fills=self.n_fills + other.n_fills,
        )


@dataclass
class TCAReport:
    """Complete TCA report for a trading period."""
    period_start: datetime
    period_end: datetime
    aggregate: TCAMetrics
    by_symbol: dict[str, TCAMetrics] = field(default_factory=dict)
    by_algo: dict[str, TCAMetrics] = field(default_factory=dict)
    by_venue: dict[str, TCAMetrics] = field(default_factory=dict)
    by_side: dict[str, TCAMetrics] = field(default_factory=dict)


class TCAAnalyzer:
    """
    Analyzes transaction costs for executed orders.

    Computes various cost metrics relative to benchmark prices.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize TCA analyzer.

        Args:
            config: Analysis configuration
        """
        self.config = config or {}

    def analyze_fills(
        self,
        fills: list[Fill],
        decision_prices: dict[str, float],
        arrival_prices: dict[str, float] | None = None,
        close_prices: dict[str, float] | None = None,
        spreads: dict[str, float] | None = None,
    ) -> TCAMetrics:
        """
        Analyze a set of fills.

        Args:
            fills: List of fills to analyze
            decision_prices: Price at decision time per symbol
            arrival_prices: Price at order arrival per symbol
            close_prices: Price at interval close per symbol
            spreads: Bid-ask spread per symbol (in price units)

        Returns:
            TCA metrics for the fills
        """
        if not fills:
            return TCAMetrics()

        arrival_prices = arrival_prices or decision_prices
        close_prices = close_prices or arrival_prices
        spreads = spreads or {}

        total_slippage = 0.0
        total_is = 0.0
        total_timing = 0.0
        total_spread_capture = 0.0
        total_notional = 0.0

        for fill in fills:
            decision_price = decision_prices.get(fill.symbol, fill.price)
            arrival_price = arrival_prices.get(fill.symbol, decision_price)
            close_price = close_prices.get(fill.symbol, arrival_price)
            spread = spreads.get(fill.symbol, 0.0)

            notional = fill.qty * fill.price
            total_notional += notional

            # Slippage: difference from decision price
            side_mult = 1 if fill.side == "buy" else -1
            slippage = side_mult * (fill.price - decision_price)
            slippage_bps = (slippage / decision_price) * 10000 if decision_price > 0 else 0
            total_slippage += slippage_bps * notional

            # Implementation shortfall: difference from arrival price
            is_cost = side_mult * (fill.price - arrival_price)
            is_bps = (is_cost / arrival_price) * 10000 if arrival_price > 0 else 0
            total_is += is_bps * notional

            # Timing cost: price movement during execution
            timing = side_mult * (close_price - arrival_price)
            timing_bps = (timing / arrival_price) * 10000 if arrival_price > 0 else 0
            total_timing += timing_bps * notional

            # Spread capture
            if spread > 0:
                mid = arrival_price
                if fill.side == "buy":
                    # Buying: paid spread if executed above mid
                    spread_cost = fill.price - mid
                else:
                    # Selling: captured spread if executed above mid
                    spread_cost = mid - fill.price
                spread_capture = -spread_cost / (spread / 2) * 100  # % of half-spread
                total_spread_capture += spread_capture * notional

        # Compute weighted averages
        if total_notional > 0:
            avg_slippage = total_slippage / total_notional
            avg_is = total_is / total_notional
            avg_timing = total_timing / total_notional
            avg_spread = total_spread_capture / total_notional
        else:
            avg_slippage = avg_is = avg_timing = avg_spread = 0.0

        return TCAMetrics(
            slippage_bps=avg_slippage,
            implementation_shortfall_bps=avg_is,
            timing_cost_bps=avg_timing,
            spread_capture_pct=avg_spread,
            total_cost_bps=avg_slippage,  # Slippage as total cost
            notional=total_notional,
            n_fills=len(fills),
        )

    def generate_report(
        self,
        fills: list[Fill],
        decision_prices: dict[str, float],
        arrival_prices: dict[str, float] | None = None,
        close_prices: dict[str, float] | None = None,
        spreads: dict[str, float] | None = None,
        algos: dict[str, str] | None = None,  # order_id -> algo
    ) -> TCAReport:
        """
        Generate comprehensive TCA report.

        Args:
            fills: All fills in the period
            decision_prices: Prices at decision time
            arrival_prices: Prices at order arrival
            close_prices: Prices at close
            spreads: Bid-ask spreads
            algos: Map from order_id to algorithm used

        Returns:
            Complete TCA report
        """
        if not fills:
            now = datetime.utcnow()
            return TCAReport(
                period_start=now,
                period_end=now,
                aggregate=TCAMetrics(),
            )

        algos = algos or {}

        # Group fills
        by_symbol: dict[str, list[Fill]] = {}
        by_algo: dict[str, list[Fill]] = {}
        by_venue: dict[str, list[Fill]] = {}
        by_side: dict[str, list[Fill]] = {}

        for fill in fills:
            by_symbol.setdefault(fill.symbol, []).append(fill)
            by_side.setdefault(fill.side, []).append(fill)
            if fill.venue:
                by_venue.setdefault(fill.venue, []).append(fill)
            algo = algos.get(fill.order_id, "unknown")
            by_algo.setdefault(algo, []).append(fill)

        # Compute aggregate
        aggregate = self.analyze_fills(
            fills, decision_prices, arrival_prices, close_prices, spreads
        )

        # Compute by-group metrics
        symbol_metrics = {
            sym: self.analyze_fills(f, decision_prices, arrival_prices, close_prices, spreads)
            for sym, f in by_symbol.items()
        }
        algo_metrics = {
            algo: self.analyze_fills(f, decision_prices, arrival_prices, close_prices, spreads)
            for algo, f in by_algo.items()
        }
        venue_metrics = {
            venue: self.analyze_fills(f, decision_prices, arrival_prices, close_prices, spreads)
            for venue, f in by_venue.items()
        }
        side_metrics = {
            side: self.analyze_fills(f, decision_prices, arrival_prices, close_prices, spreads)
            for side, f in by_side.items()
        }

        # Determine period
        timestamps = [f.timestamp for f in fills]
        period_start = min(timestamps)
        period_end = max(timestamps)

        return TCAReport(
            period_start=period_start,
            period_end=period_end,
            aggregate=aggregate,
            by_symbol=symbol_metrics,
            by_algo=algo_metrics,
            by_venue=venue_metrics,
            by_side=side_metrics,
        )

    def estimate_market_impact(
        self,
        fills: list[Fill],
        pre_trade_prices: dict[str, float],
        post_trade_prices: dict[str, float],
    ) -> dict[str, float]:
        """
        Estimate realized market impact from price changes.

        Args:
            fills: Executed fills
            pre_trade_prices: Prices before trading
            post_trade_prices: Prices after trading

        Returns:
            Dict with permanent and temporary impact estimates
        """
        if not fills:
            return {"permanent_impact_bps": 0.0, "total_impact_bps": 0.0}

        permanent_impacts = []
        total_impacts = []

        for fill in fills:
            pre = pre_trade_prices.get(fill.symbol, fill.price)
            post = post_trade_prices.get(fill.symbol, fill.price)

            if pre <= 0:
                continue

            # Permanent impact: post-trade price change
            side_mult = 1 if fill.side == "buy" else -1
            permanent = side_mult * (post - pre) / pre * 10000
            permanent_impacts.append(permanent)

            # Total impact: slippage from pre-trade
            total = side_mult * (fill.price - pre) / pre * 10000
            total_impacts.append(total)

        return {
            "permanent_impact_bps": np.mean(permanent_impacts) if permanent_impacts else 0.0,
            "temporary_impact_bps": np.mean(total_impacts) - np.mean(permanent_impacts) if permanent_impacts else 0.0,
            "total_impact_bps": np.mean(total_impacts) if total_impacts else 0.0,
        }


def compute_vwap_slippage(
    fills: list[Fill],
    market_vwaps: dict[str, float],
) -> float:
    """
    Compute VWAP slippage (execution price vs market VWAP).

    Args:
        fills: Executed fills
        market_vwaps: Market VWAP per symbol

    Returns:
        VWAP slippage in bps
    """
    if not fills:
        return 0.0

    total_slippage = 0.0
    total_notional = 0.0

    for fill in fills:
        vwap = market_vwaps.get(fill.symbol)
        if vwap is None or vwap <= 0:
            continue

        notional = fill.qty * fill.price
        side_mult = 1 if fill.side == "buy" else -1
        slippage = side_mult * (fill.price - vwap) / vwap * 10000
        total_slippage += slippage * notional
        total_notional += notional

    return total_slippage / total_notional if total_notional > 0 else 0.0
