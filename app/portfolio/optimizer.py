# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""
Portfolio optimization methods.

Implements various portfolio construction approaches:
- Mean-variance optimization (Markowitz)
- Risk parity
- Maximum Sharpe ratio
- Black-Litterman

All optimizers return weight dictionaries mapping symbols to allocations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConstraints:
    """Constraints for portfolio optimization."""
    max_position_pct: float = 0.20  # Max weight per position
    min_position_pct: float = 0.0   # Min weight (0 = allow zero positions)
    max_sector_pct: dict[str, float] = field(default_factory=dict)  # Sector limits
    max_turnover_pct: float = 1.0   # Max daily turnover
    long_only: bool = True          # No short positions
    factor_neutral: list[str] = field(default_factory=list)  # Neutralize these factors


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""
    weights: dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    method: str
    converged: bool = True
    message: str = ""


class PortfolioOptimizer:
    """
    Multi-method portfolio optimizer.

    Supports mean-variance, risk parity, max Sharpe, and other methods.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize optimizer.

        Args:
            config: Optimizer configuration
        """
        self.config = config or {}
        self.risk_free_rate = self.config.get("risk_free_rate", 0.05)

    def mean_variance(
        self,
        expected_returns: dict[str, float],
        cov_matrix: np.ndarray,
        symbols: list[str],
        risk_aversion: float = 1.0,
        constraints: PortfolioConstraints | None = None,
    ) -> OptimizationResult:
        """
        Mean-variance optimization (Markowitz).

        Maximizes: E[R] - (lambda/2) * Var[R]

        Args:
            expected_returns: Expected return per symbol
            cov_matrix: Covariance matrix (n x n)
            symbols: List of symbols (in cov_matrix order)
            risk_aversion: Risk aversion parameter (higher = more conservative)
            constraints: Portfolio constraints

        Returns:
            Optimization result with weights
        """
        constraints = constraints or PortfolioConstraints()
        n = len(symbols)
        mu = np.array([expected_returns.get(s, 0.0) for s in symbols])

        try:
            # Analytical solution for unconstrained case
            # w* = (1/lambda) * Sigma^-1 * mu
            cov_inv = np.linalg.inv(cov_matrix + np.eye(n) * 1e-6)
            raw_weights = (1 / risk_aversion) * cov_inv @ mu

            # Apply constraints
            weights = self._apply_constraints(raw_weights, symbols, constraints)

            # Compute portfolio metrics
            w = np.array([weights.get(s, 0.0) for s in symbols])
            exp_ret = float(w @ mu)
            exp_vol = float(np.sqrt(w @ cov_matrix @ w))
            sharpe = (exp_ret - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0.0

            return OptimizationResult(
                weights=weights,
                expected_return=exp_ret,
                expected_volatility=exp_vol,
                sharpe_ratio=sharpe,
                method="mean_variance",
            )

        except Exception as e:
            logger.warning("Mean-variance optimization failed: %s", e)
            return self._equal_weight_fallback(symbols, "mean_variance", str(e))

    def risk_parity(
        self,
        cov_matrix: np.ndarray,
        symbols: list[str],
        constraints: PortfolioConstraints | None = None,
    ) -> OptimizationResult:
        """
        Risk parity optimization.

        Equalizes risk contribution from each asset.

        Args:
            cov_matrix: Covariance matrix
            symbols: List of symbols
            constraints: Portfolio constraints

        Returns:
            Optimization result with weights
        """
        constraints = constraints or PortfolioConstraints()
        n = len(symbols)

        try:
            # Iterative risk parity using Newton's method
            weights = np.ones(n) / n
            for _ in range(100):
                # Portfolio volatility
                port_vol = np.sqrt(weights @ cov_matrix @ weights)
                if port_vol < 1e-10:
                    break

                # Marginal risk contribution
                mrc = cov_matrix @ weights / port_vol

                # Risk contribution
                rc = weights * mrc

                # Target: equal risk contribution
                target_rc = port_vol / n

                # Update weights
                adjustment = rc - target_rc
                weights = weights * np.exp(-0.5 * adjustment / (rc + 1e-10))
                weights = weights / weights.sum()

            # Apply constraints
            weight_dict = self._apply_constraints(weights, symbols, constraints)

            # Compute metrics (no expected return for risk parity)
            w = np.array([weight_dict.get(s, 0.0) for s in symbols])
            exp_vol = float(np.sqrt(w @ cov_matrix @ w))

            return OptimizationResult(
                weights=weight_dict,
                expected_return=0.0,  # Not using expected returns
                expected_volatility=exp_vol,
                sharpe_ratio=0.0,
                method="risk_parity",
            )

        except Exception as e:
            logger.warning("Risk parity optimization failed: %s", e)
            return self._equal_weight_fallback(symbols, "risk_parity", str(e))

    def max_sharpe(
        self,
        expected_returns: dict[str, float],
        cov_matrix: np.ndarray,
        symbols: list[str],
        constraints: PortfolioConstraints | None = None,
    ) -> OptimizationResult:
        """
        Maximum Sharpe ratio optimization.

        Args:
            expected_returns: Expected returns per symbol
            cov_matrix: Covariance matrix
            symbols: List of symbols
            constraints: Portfolio constraints

        Returns:
            Optimization result with weights
        """
        constraints = constraints or PortfolioConstraints()
        n = len(symbols)
        mu = np.array([expected_returns.get(s, 0.0) for s in symbols])
        excess_returns = mu - self.risk_free_rate

        try:
            # Analytical solution: w* proportional to Sigma^-1 * (mu - rf)
            cov_inv = np.linalg.inv(cov_matrix + np.eye(n) * 1e-6)
            raw_weights = cov_inv @ excess_returns
            raw_weights = raw_weights / raw_weights.sum()  # Normalize

            # Apply constraints
            weights = self._apply_constraints(raw_weights, symbols, constraints)

            # Compute metrics
            w = np.array([weights.get(s, 0.0) for s in symbols])
            exp_ret = float(w @ mu)
            exp_vol = float(np.sqrt(w @ cov_matrix @ w))
            sharpe = (exp_ret - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0.0

            return OptimizationResult(
                weights=weights,
                expected_return=exp_ret,
                expected_volatility=exp_vol,
                sharpe_ratio=sharpe,
                method="max_sharpe",
            )

        except Exception as e:
            logger.warning("Max Sharpe optimization failed: %s", e)
            return self._equal_weight_fallback(symbols, "max_sharpe", str(e))

    def minimum_variance(
        self,
        cov_matrix: np.ndarray,
        symbols: list[str],
        constraints: PortfolioConstraints | None = None,
    ) -> OptimizationResult:
        """
        Minimum variance portfolio.

        Args:
            cov_matrix: Covariance matrix
            symbols: List of symbols
            constraints: Portfolio constraints

        Returns:
            Optimization result with weights
        """
        constraints = constraints or PortfolioConstraints()
        n = len(symbols)

        try:
            # Analytical solution: w* = Sigma^-1 * 1 / (1' * Sigma^-1 * 1)
            cov_inv = np.linalg.inv(cov_matrix + np.eye(n) * 1e-6)
            ones = np.ones(n)
            raw_weights = cov_inv @ ones
            raw_weights = raw_weights / (ones @ raw_weights)

            weights = self._apply_constraints(raw_weights, symbols, constraints)

            w = np.array([weights.get(s, 0.0) for s in symbols])
            exp_vol = float(np.sqrt(w @ cov_matrix @ w))

            return OptimizationResult(
                weights=weights,
                expected_return=0.0,
                expected_volatility=exp_vol,
                sharpe_ratio=0.0,
                method="minimum_variance",
            )

        except Exception as e:
            logger.warning("Minimum variance optimization failed: %s", e)
            return self._equal_weight_fallback(symbols, "minimum_variance", str(e))

    def black_litterman(
        self,
        market_weights: dict[str, float],
        cov_matrix: np.ndarray,
        symbols: list[str],
        views: dict[str, float] | None = None,
        view_confidence: dict[str, float] | None = None,
        tau: float = 0.05,
        constraints: PortfolioConstraints | None = None,
    ) -> OptimizationResult:
        """
        Black-Litterman optimization.

        Combines market equilibrium with investor views.

        Args:
            market_weights: Market cap weights
            cov_matrix: Covariance matrix
            symbols: List of symbols
            views: Investor views on expected returns
            view_confidence: Confidence in each view (0-1)
            tau: Uncertainty in equilibrium (typically 0.01-0.1)
            constraints: Portfolio constraints

        Returns:
            Optimization result with weights
        """
        constraints = constraints or PortfolioConstraints()
        views = views or {}
        view_confidence = view_confidence or {}
        n = len(symbols)

        try:
            # Market equilibrium returns (reverse optimization)
            w_mkt = np.array([market_weights.get(s, 1/n) for s in symbols])
            delta = 2.5  # Risk aversion (typical value)
            pi = delta * cov_matrix @ w_mkt  # Equilibrium returns

            if not views:
                # No views - return market weights
                weights = {s: float(w_mkt[i]) for i, s in enumerate(symbols)}
                exp_ret = float(w_mkt @ pi)
                exp_vol = float(np.sqrt(w_mkt @ cov_matrix @ w_mkt))
                sharpe = (exp_ret - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0.0

                return OptimizationResult(
                    weights=weights,
                    expected_return=exp_ret,
                    expected_volatility=exp_vol,
                    sharpe_ratio=sharpe,
                    method="black_litterman",
                )

            # Build view matrix P and view vector Q
            view_symbols = [s for s in views if s in symbols]
            k = len(view_symbols)
            P = np.zeros((k, n))
            Q = np.zeros(k)
            omega_diag = np.zeros(k)

            for i, sym in enumerate(view_symbols):
                j = symbols.index(sym)
                P[i, j] = 1.0
                Q[i] = views[sym]
                conf = view_confidence.get(sym, 0.5)
                omega_diag[i] = (1 - conf) / conf * tau * cov_matrix[j, j]

            Omega = np.diag(omega_diag)

            # Black-Litterman formula
            tau_sigma = tau * cov_matrix
            tau_sigma_inv = np.linalg.inv(tau_sigma + np.eye(n) * 1e-8)
            omega_inv = np.linalg.inv(Omega + np.eye(k) * 1e-8)

            # Posterior expected returns
            M = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
            mu_bl = M @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)

            # Optimize with BL returns
            return self.mean_variance(
                expected_returns={s: float(mu_bl[i]) for i, s in enumerate(symbols)},
                cov_matrix=cov_matrix,
                symbols=symbols,
                risk_aversion=delta,
                constraints=constraints,
            )

        except Exception as e:
            logger.warning("Black-Litterman optimization failed: %s", e)
            return self._equal_weight_fallback(symbols, "black_litterman", str(e))

    def _apply_constraints(
        self,
        weights: np.ndarray,
        symbols: list[str],
        constraints: PortfolioConstraints,
    ) -> dict[str, float]:
        """Apply portfolio constraints to raw weights."""
        n = len(symbols)
        w = weights.copy()

        # Long-only constraint
        if constraints.long_only:
            w = np.maximum(w, 0.0)

        # Position limits
        w = np.clip(w, constraints.min_position_pct, constraints.max_position_pct)

        # Renormalize
        if w.sum() > 0:
            w = w / w.sum()
        else:
            w = np.ones(n) / n

        return {s: float(w[i]) for i, s in enumerate(symbols)}

    def _equal_weight_fallback(
        self,
        symbols: list[str],
        method: str,
        error: str,
    ) -> OptimizationResult:
        """Return equal-weight portfolio as fallback."""
        n = len(symbols)
        weights = {s: 1.0 / n for s in symbols}
        return OptimizationResult(
            weights=weights,
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            method=method,
            converged=False,
            message=f"Fallback to equal weight: {error}",
        )
