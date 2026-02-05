# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""
Risk models for portfolio optimization.

Provides covariance matrix estimation methods:
- Sample covariance
- Ledoit-Wolf shrinkage
- Exponentially weighted
- Factor models

Also includes factor exposure computation and risk decomposition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Risk metrics for a portfolio."""
    volatility: float
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional VaR (95%)
    max_drawdown: float
    beta: float  # Market beta
    tracking_error: float


class RiskModel:
    """
    Covariance estimation and risk decomposition.

    Supports multiple estimation methods for robust covariance matrices.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize risk model.

        Args:
            config: Risk model configuration
        """
        self.config = config or {}
        self.shrinkage_target = self.config.get("shrinkage_target", "constant_correlation")
        self.ewma_halflife = self.config.get("ewma_halflife", 60)

    def compute_covariance(
        self,
        returns: np.ndarray,
        method: str = "ledoit_wolf",
    ) -> np.ndarray:
        """
        Compute covariance matrix.

        Args:
            returns: Return matrix (n_samples, n_assets)
            method: Estimation method (sample, ledoit_wolf, ewma, factor)

        Returns:
            Covariance matrix (n_assets, n_assets)
        """
        if method == "sample":
            return self._sample_covariance(returns)
        elif method == "ledoit_wolf":
            return self._ledoit_wolf_covariance(returns)
        elif method == "ewma":
            return self._ewma_covariance(returns)
        else:
            logger.warning("Unknown method %s, using Ledoit-Wolf", method)
            return self._ledoit_wolf_covariance(returns)

    def _sample_covariance(self, returns: np.ndarray) -> np.ndarray:
        """Standard sample covariance."""
        return np.cov(returns.T)

    def _ledoit_wolf_covariance(self, returns: np.ndarray) -> np.ndarray:
        """
        Ledoit-Wolf shrinkage estimator.

        Shrinks sample covariance toward a structured target to reduce
        estimation error.
        """
        n, p = returns.shape

        # Sample covariance
        sample_cov = np.cov(returns.T)

        # Shrinkage target: constant correlation
        sample_var = np.diag(sample_cov)
        sqrt_var = np.sqrt(sample_var)
        unit_cor = sample_cov / np.outer(sqrt_var, sqrt_var + 1e-10)
        np.fill_diagonal(unit_cor, 1.0)
        avg_cor = (unit_cor.sum() - p) / (p * (p - 1))
        target = avg_cor * np.outer(sqrt_var, sqrt_var)
        np.fill_diagonal(target, sample_var)

        # Compute optimal shrinkage intensity
        # Simplified Ledoit-Wolf formula
        X = returns - returns.mean(axis=0)
        X2 = X ** 2

        # Frobenius norm of sample - target
        delta = sample_cov - target
        delta_norm = (delta ** 2).sum()

        # Asymptotic mean squared error terms
        sum1 = 0.0
        sum2 = 0.0
        for i in range(p):
            for j in range(p):
                sum1 += np.mean(X2[:, i] * X2[:, j]) - sample_cov[i, j] ** 2
                sum2 += (sample_cov[i, j] - target[i, j]) ** 2

        # Shrinkage intensity
        if delta_norm > 0:
            kappa = (sum1 / n) / delta_norm
            shrinkage = max(0.0, min(1.0, kappa))
        else:
            shrinkage = 1.0

        # Shrunk covariance
        result = shrinkage * target + (1 - shrinkage) * sample_cov

        logger.debug("Ledoit-Wolf shrinkage: %.3f", shrinkage)
        return result

    def _ewma_covariance(self, returns: np.ndarray) -> np.ndarray:
        """
        Exponentially weighted moving average covariance.

        Recent observations get higher weight.
        """
        n, p = returns.shape
        halflife = min(self.ewma_halflife, n - 1)
        decay = 0.5 ** (1 / halflife) if halflife > 0 else 0.94

        # Compute weights
        weights = np.array([decay ** i for i in range(n - 1, -1, -1)])
        weights = weights / weights.sum()

        # Weighted covariance
        mean = np.average(returns, axis=0, weights=weights)
        centered = returns - mean
        weighted_centered = centered * np.sqrt(weights)[:, np.newaxis]
        cov = weighted_centered.T @ weighted_centered

        return cov

    def compute_factor_risk(
        self,
        weights: np.ndarray,
        factor_loadings: np.ndarray,
        factor_cov: np.ndarray,
        idio_var: np.ndarray,
    ) -> dict:
        """
        Decompose portfolio risk into factor and idiosyncratic components.

        Args:
            weights: Portfolio weights (n_assets,)
            factor_loadings: Factor exposures (n_assets, n_factors)
            factor_cov: Factor covariance (n_factors, n_factors)
            idio_var: Idiosyncratic variance per asset (n_assets,)

        Returns:
            Dict with factor_risk, idio_risk, total_risk, risk_contributions
        """
        # Portfolio factor exposure
        port_exposure = weights @ factor_loadings  # (n_factors,)

        # Factor risk
        factor_var = port_exposure @ factor_cov @ port_exposure
        factor_risk = np.sqrt(factor_var)

        # Idiosyncratic risk
        idio_risk = np.sqrt((weights ** 2) @ idio_var)

        # Total risk
        total_var = factor_var + (weights ** 2) @ idio_var
        total_risk = np.sqrt(total_var)

        # Risk contribution by factor
        n_factors = factor_loadings.shape[1]
        factor_contrib = np.zeros(n_factors)
        for i in range(n_factors):
            factor_contrib[i] = port_exposure[i] ** 2 * factor_cov[i, i] / total_var

        return {
            "factor_risk": float(factor_risk),
            "idio_risk": float(idio_risk),
            "total_risk": float(total_risk),
            "factor_contributions": factor_contrib.tolist(),
            "idio_contribution": float((weights ** 2) @ idio_var / total_var),
        }

    def compute_var_cvar(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """
        Compute Value at Risk and Conditional VaR.

        Args:
            returns: Historical returns (n_samples, n_assets)
            weights: Portfolio weights
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Tuple of (VaR, CVaR) as positive numbers (losses)
        """
        # Portfolio returns
        port_returns = returns @ weights

        # VaR: quantile of losses
        var = -np.percentile(port_returns, (1 - confidence) * 100)

        # CVaR: expected loss beyond VaR
        losses = -port_returns
        cvar = losses[losses >= var].mean() if (losses >= var).any() else var

        return float(var), float(cvar)

    def compute_risk_metrics(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        benchmark_returns: np.ndarray | None = None,
    ) -> RiskMetrics:
        """
        Compute comprehensive risk metrics.

        Args:
            returns: Asset returns (n_samples, n_assets)
            weights: Portfolio weights
            benchmark_returns: Optional benchmark returns for beta/TE

        Returns:
            RiskMetrics dataclass
        """
        port_returns = returns @ weights

        # Volatility (annualized, assuming daily returns)
        volatility = float(np.std(port_returns) * np.sqrt(252))

        # VaR and CVaR
        var_95, cvar_95 = self.compute_var_cvar(returns, weights, 0.95)

        # Max drawdown
        cumulative = np.cumprod(1 + port_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = float(-drawdowns.min())

        # Beta and tracking error
        beta = 0.0
        tracking_error = 0.0
        if benchmark_returns is not None and len(benchmark_returns) == len(port_returns):
            # Beta
            cov_port_bench = np.cov(port_returns, benchmark_returns)[0, 1]
            var_bench = np.var(benchmark_returns)
            beta = float(cov_port_bench / var_bench) if var_bench > 0 else 0.0

            # Tracking error
            active_returns = port_returns - benchmark_returns
            tracking_error = float(np.std(active_returns) * np.sqrt(252))

        return RiskMetrics(
            volatility=volatility,
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            beta=beta,
            tracking_error=tracking_error,
        )


class FactorModel:
    """
    Factor model for risk and return decomposition.

    Supports common factors: market, size, value, momentum, volatility.
    """

    COMMON_FACTORS = ["market", "size", "value", "momentum", "volatility"]

    def __init__(self, factors: list[str] | None = None):
        """
        Initialize factor model.

        Args:
            factors: List of factors to use
        """
        self.factors = factors or self.COMMON_FACTORS
        self.factor_returns: dict[str, np.ndarray] = {}
        self.factor_loadings: dict[str, dict[str, float]] = {}

    def estimate_loadings(
        self,
        asset_returns: np.ndarray,
        factor_returns: np.ndarray,
        symbols: list[str],
    ) -> np.ndarray:
        """
        Estimate factor loadings via regression.

        Args:
            asset_returns: Asset returns (n_samples, n_assets)
            factor_returns: Factor returns (n_samples, n_factors)
            symbols: Asset symbols

        Returns:
            Factor loadings matrix (n_assets, n_factors)
        """
        n_assets = len(symbols)
        n_factors = factor_returns.shape[1]
        loadings = np.zeros((n_assets, n_factors))

        for i in range(n_assets):
            # OLS regression: asset_return = alpha + beta * factor_returns + epsilon
            X = np.column_stack([np.ones(len(factor_returns)), factor_returns])
            y = asset_returns[:, i]
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                loadings[i, :] = beta[1:]  # Exclude intercept
            except Exception:
                loadings[i, :] = 0.0

        return loadings

    def compute_factor_exposures(
        self,
        weights: np.ndarray,
        loadings: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute portfolio factor exposures.

        Args:
            weights: Portfolio weights
            loadings: Factor loadings matrix

        Returns:
            Dict mapping factor name to exposure
        """
        exposures = weights @ loadings
        return {f: float(exposures[i]) for i, f in enumerate(self.factors)}

    def neutralize_factor(
        self,
        weights: np.ndarray,
        loadings: np.ndarray,
        target_exposures: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Adjust weights to neutralize factor exposures.

        Args:
            weights: Original weights
            loadings: Factor loadings matrix
            target_exposures: Target factor exposures (default: zeros)

        Returns:
            Adjusted weights with neutralized factor exposures
        """
        n_factors = loadings.shape[1]
        if target_exposures is None:
            target_exposures = np.zeros(n_factors)

        # Current exposure
        current = weights @ loadings

        # Adjustment needed
        delta = target_exposures - current

        # Find minimal weight adjustment to achieve target
        # Solve: loadings.T @ w_adj = delta, minimize ||w_adj||
        # Using pseudoinverse
        loadings_pinv = np.linalg.pinv(loadings.T)
        adjustment = loadings_pinv @ delta

        # Apply adjustment
        new_weights = weights + adjustment

        # Renormalize to sum to 1
        if new_weights.sum() > 0:
            new_weights = new_weights / new_weights.sum()
        else:
            new_weights = weights

        return new_weights
