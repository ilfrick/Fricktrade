# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""
Dynamic portfolio rebalancing.

Determines when and how to rebalance the portfolio based on:
- Drift from target weights
- Time-based scheduling
- Risk-based triggers
- Transaction cost awareness
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RebalanceTrade:
    """A trade required for rebalancing."""
    symbol: str
    action: str  # "buy" or "sell"
    qty: int
    notional: float
    current_weight: float
    target_weight: float
    drift_pct: float


@dataclass
class RebalanceDecision:
    """Result of rebalance check."""
    should_rebalance: bool
    reason: str
    total_drift_pct: float
    trades: list[RebalanceTrade] = field(default_factory=list)
    estimated_cost: float = 0.0


class RebalanceEngine:
    """
    Determines when and how to rebalance the portfolio.

    Supports multiple rebalancing strategies:
    - Threshold-based: rebalance when drift exceeds threshold
    - Periodic: rebalance on schedule
    - Risk-based: rebalance when risk metrics trigger
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize rebalance engine.

        Args:
            config: Rebalancing configuration
        """
        self.config = config or {}

        # Thresholds
        self.drift_threshold_pct = self.config.get("drift_threshold_pct", 5.0)
        self.min_trade_pct = self.config.get("min_trade_pct", 0.5)
        self.min_interval_minutes = self.config.get("min_interval_minutes", 30)

        # Cost parameters
        self.commission_bps = self.config.get("commission_bps", 5.0)
        self.spread_bps = self.config.get("spread_bps", 5.0)

        # State
        self.last_rebalance: datetime | None = None

    def check_rebalance(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        prices: dict[str, float],
        nav: float,
    ) -> RebalanceDecision:
        """
        Check if rebalancing is needed.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            prices: Current prices per symbol
            nav: Current net asset value

        Returns:
            Rebalance decision with trades if needed
        """
        # Check minimum interval
        if self.last_rebalance:
            elapsed = datetime.utcnow() - self.last_rebalance
            if elapsed < timedelta(minutes=self.min_interval_minutes):
                return RebalanceDecision(
                    should_rebalance=False,
                    reason=f"Min interval not elapsed ({elapsed.total_seconds()/60:.1f}m)",
                    total_drift_pct=0.0,
                )

        # Compute drift
        total_drift, symbol_drifts = self._compute_drift(current_weights, target_weights)

        # Check threshold
        if total_drift < self.drift_threshold_pct:
            return RebalanceDecision(
                should_rebalance=False,
                reason=f"Drift {total_drift:.2f}% below threshold {self.drift_threshold_pct}%",
                total_drift_pct=total_drift,
            )

        # Compute trades needed
        trades = self._compute_trades(
            current_weights, target_weights, prices, nav, symbol_drifts
        )

        # Estimate cost
        estimated_cost = self._estimate_cost(trades)

        # Check if trades are worth the cost
        if not trades:
            return RebalanceDecision(
                should_rebalance=False,
                reason="No trades above minimum threshold",
                total_drift_pct=total_drift,
            )

        return RebalanceDecision(
            should_rebalance=True,
            reason=f"Drift {total_drift:.2f}% exceeds threshold",
            total_drift_pct=total_drift,
            trades=trades,
            estimated_cost=estimated_cost,
        )

    def execute_rebalance(self, decision: RebalanceDecision) -> None:
        """Mark rebalance as executed (update last_rebalance timestamp)."""
        if decision.should_rebalance:
            self.last_rebalance = datetime.utcnow()
            logger.info(
                "Rebalance executed: %d trades, drift=%.2f%%, cost=$%.2f",
                len(decision.trades),
                decision.total_drift_pct,
                decision.estimated_cost,
            )

    def _compute_drift(
        self,
        current: dict[str, float],
        target: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        """Compute total drift and per-symbol drift."""
        all_symbols = set(current.keys()) | set(target.keys())
        symbol_drifts = {}
        total_drift = 0.0

        for sym in all_symbols:
            curr = current.get(sym, 0.0)
            tgt = target.get(sym, 0.0)
            drift = abs(curr - tgt) * 100  # Convert to percentage points
            symbol_drifts[sym] = drift
            total_drift += drift

        # Total drift is sum of absolute deviations / 2 (since they double-count)
        return total_drift / 2, symbol_drifts

    def _compute_trades(
        self,
        current: dict[str, float],
        target: dict[str, float],
        prices: dict[str, float],
        nav: float,
        drifts: dict[str, float],
    ) -> list[RebalanceTrade]:
        """Compute trades needed for rebalancing."""
        trades = []
        all_symbols = set(current.keys()) | set(target.keys())

        for sym in all_symbols:
            curr_weight = current.get(sym, 0.0)
            tgt_weight = target.get(sym, 0.0)
            drift = drifts.get(sym, 0.0)

            # Skip small drifts
            if drift < self.min_trade_pct:
                continue

            price = prices.get(sym, 0.0)
            if price <= 0:
                continue

            # Calculate trade
            weight_delta = tgt_weight - curr_weight
            notional = weight_delta * nav
            qty = int(abs(notional) / price)

            if qty == 0:
                continue

            trades.append(RebalanceTrade(
                symbol=sym,
                action="buy" if weight_delta > 0 else "sell",
                qty=qty,
                notional=abs(notional),
                current_weight=curr_weight,
                target_weight=tgt_weight,
                drift_pct=drift,
            ))

        return trades

    def _estimate_cost(self, trades: list[RebalanceTrade]) -> float:
        """Estimate transaction costs for trades."""
        total_notional = sum(t.notional for t in trades)
        cost_bps = self.commission_bps + self.spread_bps / 2
        return total_notional * cost_bps / 10000


class AdaptiveRebalancer:
    """
    Adaptive rebalancing that considers market conditions.

    Adjusts rebalancing behavior based on:
    - Volatility (widen thresholds in volatile markets)
    - Liquidity (delay rebalancing in illiquid conditions)
    - Transaction costs (avoid rebalancing when costs exceed benefit)
    """

    def __init__(self, base_engine: RebalanceEngine | None = None, config: dict | None = None):
        """
        Initialize adaptive rebalancer.

        Args:
            base_engine: Base rebalance engine
            config: Adaptive configuration
        """
        self.base = base_engine or RebalanceEngine()
        self.config = config or {}

        self.vol_adjustment = self.config.get("vol_adjustment", True)
        self.vol_threshold_low = self.config.get("vol_threshold_low", 0.15)
        self.vol_threshold_high = self.config.get("vol_threshold_high", 0.30)

    def check_rebalance(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        prices: dict[str, float],
        nav: float,
        market_vol: float | None = None,
        liquidity_score: float | None = None,
    ) -> RebalanceDecision:
        """
        Check if adaptive rebalancing is needed.

        Args:
            current_weights: Current weights
            target_weights: Target weights
            prices: Current prices
            nav: Net asset value
            market_vol: Current market volatility (annualized)
            liquidity_score: Liquidity score (0-1, higher = more liquid)

        Returns:
            Rebalance decision
        """
        # Adjust threshold based on volatility
        if self.vol_adjustment and market_vol is not None:
            if market_vol > self.vol_threshold_high:
                # High volatility - widen threshold
                self.base.drift_threshold_pct *= 1.5
            elif market_vol < self.vol_threshold_low:
                # Low volatility - tighten threshold
                self.base.drift_threshold_pct *= 0.75

        # Check base rebalance
        decision = self.base.check_rebalance(
            current_weights, target_weights, prices, nav
        )

        # Adjust for liquidity
        if liquidity_score is not None and decision.should_rebalance:
            if liquidity_score < 0.3:
                # Low liquidity - delay unless drift is severe
                if decision.total_drift_pct < self.base.drift_threshold_pct * 2:
                    return RebalanceDecision(
                        should_rebalance=False,
                        reason=f"Low liquidity ({liquidity_score:.2f}) - delaying",
                        total_drift_pct=decision.total_drift_pct,
                    )

        return decision


def compute_optimal_rebalance_schedule(
    volatility: float,
    transaction_cost_bps: float,
    expected_drift_rate: float = 0.01,
) -> int:
    """
    Compute optimal rebalancing frequency.

    Based on the Almgren model for optimal rebalancing interval.

    Args:
        volatility: Portfolio volatility (annualized)
        transaction_cost_bps: Round-trip transaction cost in bps
        expected_drift_rate: Expected daily drift from target

    Returns:
        Optimal rebalancing interval in days
    """
    # Simplified model: balance drift cost vs transaction cost
    # Optimal interval ~ sqrt(transaction_cost / (volatility * drift_rate))
    tc = transaction_cost_bps / 10000
    drift = expected_drift_rate

    if drift <= 0 or volatility <= 0:
        return 5  # Default: weekly

    optimal_days = np.sqrt(tc / (volatility * drift))
    return max(1, min(30, int(optimal_days)))
