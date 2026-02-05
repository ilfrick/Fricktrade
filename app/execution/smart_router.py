# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""
Smart order routing with market impact minimization.

Selects optimal execution algorithm based on order characteristics
and current market conditions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.execution.impact import estimate_market_impact, ImpactEstimate
from app.execution.algos import twap_slices, vwap_slices, pov_slices, AlgoSlice

logger = logging.getLogger(__name__)


@dataclass
class OrderContext:
    """Context for routing decision."""
    symbol: str
    side: str  # "buy" or "sell"
    qty: int
    price: float
    urgency: float = 0.5  # 0=patient, 1=urgent
    market_state: dict = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    algo: str  # "twap", "vwap", "pov", "market"
    slices: list[AlgoSlice]
    estimated_impact_bps: float
    participation_rate: float
    reason: str


class SmartOrderRouter:
    """
    Intelligent order router that minimizes market impact.

    Selects execution algorithm based on:
    - Order size relative to volume
    - Market volatility
    - Bid-ask spread
    - Urgency requirements
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize router.

        Args:
            config: Router configuration with thresholds
        """
        self.config = config or {}

        # Default thresholds
        self.max_participation = self.config.get("max_participation", 0.10)
        self.urgency_threshold_high = self.config.get("urgency_threshold_high", 0.8)
        self.urgency_threshold_low = self.config.get("urgency_threshold_low", 0.3)
        self.spread_threshold_bps = self.config.get("spread_threshold_bps", 20)
        self.volatility_threshold_pct = self.config.get("volatility_threshold_pct", 2.0)
        self.default_duration_seconds = self.config.get("default_duration_seconds", 300)
        self.default_slices = self.config.get("default_slices", 10)

    def route(self, ctx: OrderContext) -> RoutingDecision:
        """
        Determine optimal execution strategy.

        Args:
            ctx: Order context with market state

        Returns:
            Routing decision with algorithm and slices
        """
        # Estimate market impact
        notional = ctx.qty * ctx.price
        impact = estimate_market_impact(
            notional=notional,
            price=ctx.price,
            market_state=ctx.market_state,
            cfg=self.config.get("impact", {}),
        )

        # Get market conditions
        spread_bps = ctx.market_state.get("spread_bps", 5.0)
        volume_profile = ctx.market_state.get("volume_profile", [])
        session_volume = ctx.market_state.get("session_volume", 0)

        # Select algorithm based on conditions
        algo, reason = self._select_algo(ctx, impact, spread_bps, session_volume)

        # Generate slices
        slices = self._generate_slices(algo, ctx, volume_profile, session_volume)

        # Calculate effective participation rate
        participation = self._calc_participation(slices, session_volume)

        logger.debug(
            "Routed %s %d %s: algo=%s impact=%.2fbps participation=%.2f%%",
            ctx.side, ctx.qty, ctx.symbol, algo, impact.impact_bps, participation * 100,
        )

        return RoutingDecision(
            algo=algo,
            slices=slices,
            estimated_impact_bps=impact.impact_bps,
            participation_rate=participation,
            reason=reason,
        )

    def _select_algo(
        self,
        ctx: OrderContext,
        impact: ImpactEstimate,
        spread_bps: float,
        session_volume: float,
    ) -> tuple[str, str]:
        """Select execution algorithm based on conditions."""

        # High urgency -> market order
        if ctx.urgency >= self.urgency_threshold_high:
            return "market", "High urgency requires immediate execution"

        # Wide spread -> TWAP to average
        if spread_bps > self.spread_threshold_bps:
            return "twap", f"Wide spread ({spread_bps:.1f}bps) - using TWAP"

        # High volatility -> POV to adapt
        if impact.volatility_pct > self.volatility_threshold_pct:
            return "pov", f"High volatility ({impact.volatility_pct:.1f}%) - using POV"

        # Large order relative to volume -> VWAP
        if impact.participation > 0.05:
            return "vwap", f"Large order ({impact.participation:.1%} of volume) - using VWAP"

        # Low urgency -> TWAP for patient execution
        if ctx.urgency <= self.urgency_threshold_low:
            return "twap", "Low urgency - using TWAP for better price"

        # Default to TWAP
        return "twap", "Default execution"

    def _generate_slices(
        self,
        algo: str,
        ctx: OrderContext,
        volume_profile: list[float],
        session_volume: float,
    ) -> list[AlgoSlice]:
        """Generate execution slices for selected algorithm."""

        if algo == "market":
            # Single immediate slice
            return [AlgoSlice(qty=ctx.qty, earliest_at=datetime.utcnow())]

        elif algo == "twap":
            duration = int(self.default_duration_seconds * (1 - ctx.urgency))
            duration = max(duration, 60)  # Minimum 1 minute
            return twap_slices(ctx.qty, duration, self.default_slices)

        elif algo == "vwap":
            if not volume_profile:
                # Fallback to uniform TWAP
                return twap_slices(ctx.qty, self.default_duration_seconds, self.default_slices)
            return vwap_slices(ctx.qty, volume_profile, self.default_duration_seconds)

        elif algo == "pov":
            est_volume = session_volume / 6.5  # Hourly estimate from session
            participation = min(self.max_participation, 0.05 + 0.1 * ctx.urgency)
            return pov_slices(ctx.qty, participation, est_volume)

        else:
            logger.warning("Unknown algo %s, using TWAP", algo)
            return twap_slices(ctx.qty, self.default_duration_seconds, self.default_slices)

    def _calc_participation(self, slices: list[AlgoSlice], session_volume: float) -> float:
        """Calculate participation rate from slices."""
        if not slices or session_volume <= 0:
            return 0.0
        total_qty = sum(s.qty for s in slices)
        # Assume slices execute over the session
        return total_qty / session_volume


@dataclass
class AlmgrenChrissParams:
    """Parameters for Almgren-Chriss market impact model."""
    sigma: float  # Daily volatility
    eta: float  # Temporary impact coefficient
    gamma: float  # Permanent impact coefficient
    lambda_: float  # Risk aversion
    tau: float  # Trading horizon (days)


def almgren_chriss_optimal_trajectory(
    total_shares: int,
    params: AlmgrenChrissParams,
    n_intervals: int = 10,
) -> list[float]:
    """
    Compute optimal execution trajectory using Almgren-Chriss model.

    Minimizes expected cost + risk penalty for large order execution.

    Args:
        total_shares: Total shares to execute
        params: Model parameters
        n_intervals: Number of trading intervals

    Returns:
        List of shares to trade in each interval
    """
    import numpy as np

    sigma = params.sigma
    eta = params.eta
    gamma = params.gamma
    lambda_ = params.lambda_
    tau = params.tau

    # Time step
    dt = tau / n_intervals

    # Compute kappa (urgency parameter)
    kappa_sq = lambda_ * sigma ** 2 / eta
    kappa = np.sqrt(kappa_sq) if kappa_sq > 0 else 0.01

    # Optimal trading rate (continuous approximation)
    # x(t) = X * sinh(kappa * (T - t)) / sinh(kappa * T)
    trajectory = []
    for i in range(n_intervals):
        t = i * dt
        remaining_time = tau - t
        if remaining_time <= 0:
            trajectory.append(0)
        else:
            # Fraction remaining
            frac = np.sinh(kappa * remaining_time) / np.sinh(kappa * tau)
            # Trade this interval
            if i == 0:
                prev_frac = 1.0
            else:
                prev_time = (i - 1) * dt
                prev_remaining = tau - prev_time
                prev_frac = np.sinh(kappa * prev_remaining) / np.sinh(kappa * tau)
            trade_frac = prev_frac - frac
            trajectory.append(int(total_shares * trade_frac))

    # Adjust for rounding
    total_traded = sum(trajectory)
    if total_traded < total_shares:
        trajectory[-1] += (total_shares - total_traded)

    return trajectory


def estimate_almgren_chriss_cost(
    total_shares: int,
    price: float,
    params: AlmgrenChrissParams,
    trajectory: list[float] | None = None,
) -> dict:
    """
    Estimate execution cost using Almgren-Chriss model.

    Args:
        total_shares: Total shares to execute
        price: Current price
        params: Model parameters
        trajectory: Optional custom trajectory

    Returns:
        Dict with permanent_impact, temporary_impact, variance, total_cost
    """
    import numpy as np

    if trajectory is None:
        trajectory = almgren_chriss_optimal_trajectory(total_shares, params)

    X = total_shares
    sigma = params.sigma
    eta = params.eta
    gamma = params.gamma
    tau = params.tau
    n = len(trajectory)
    dt = tau / n

    # Permanent impact: proportional to total traded
    permanent_impact = gamma * X * price

    # Temporary impact: depends on trading rate
    temp_impact = 0.0
    for trade_qty in trajectory:
        if trade_qty > 0:
            # Temporary impact per interval
            rate = trade_qty / dt if dt > 0 else trade_qty
            temp_impact += eta * rate * trade_qty * price / X if X > 0 else 0

    # Execution variance
    variance = sigma ** 2 * tau * X ** 2 * price ** 2

    return {
        "permanent_impact": permanent_impact,
        "temporary_impact": temp_impact,
        "variance": variance,
        "total_cost": permanent_impact + temp_impact,
        "cost_bps": (permanent_impact + temp_impact) / (X * price) * 10000 if X > 0 else 0,
    }
