# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""
Portfolio optimization and risk management.

This module provides:
- Portfolio optimization (mean-variance, risk parity, max Sharpe, Black-Litterman)
- Risk models (covariance estimation, factor risk decomposition)
- Dynamic rebalancing with cost awareness

Usage:
    from app.portfolio import (
        PortfolioOptimizer,
        RiskModel,
        RebalanceEngine,
    )

    # Optimize portfolio
    optimizer = PortfolioOptimizer()
    result = optimizer.max_sharpe(
        expected_returns={"AAPL": 0.15, "GOOGL": 0.12},
        cov_matrix=cov,
        symbols=["AAPL", "GOOGL"],
    )

    # Estimate risk
    risk_model = RiskModel()
    cov = risk_model.compute_covariance(returns, method="ledoit_wolf")

    # Check rebalance
    engine = RebalanceEngine()
    decision = engine.check_rebalance(current, target, prices, nav)
"""

from app.portfolio.optimizer import (
    PortfolioConstraints,
    OptimizationResult,
    PortfolioOptimizer,
)
from app.portfolio.risk_model import (
    RiskMetrics,
    RiskModel,
    FactorModel,
)
from app.portfolio.rebalance import (
    RebalanceTrade,
    RebalanceDecision,
    RebalanceEngine,
    AdaptiveRebalancer,
    compute_optimal_rebalance_schedule,
)

__all__ = [
    # Optimizer
    "PortfolioConstraints",
    "OptimizationResult",
    "PortfolioOptimizer",
    # Risk
    "RiskMetrics",
    "RiskModel",
    "FactorModel",
    # Rebalance
    "RebalanceTrade",
    "RebalanceDecision",
    "RebalanceEngine",
    "AdaptiveRebalancer",
    "compute_optimal_rebalance_schedule",
]
