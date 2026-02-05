# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""
Advanced regime detection using Hidden Markov Models and changepoint detection.

Provides:
- HMM-based regime inference with Gaussian emissions
- Changepoint detection for structural breaks
- Regime-conditional strategy selection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Any

import numpy as np

logger = logging.getLogger(__name__)

# Optional HMM library
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    GaussianHMM = None


@dataclass
class RegimeState:
    """Current regime state with probabilities."""
    regime: int
    regime_name: str
    probability: float
    regime_probs: list[float]
    features: dict[str, float]


class RegimeHMM:
    """
    Hidden Markov Model for market regime detection.

    Uses Gaussian emissions on return and volatility features
    to infer underlying market regimes.

    Typical regimes:
    - 0: Low volatility / trending up
    - 1: Medium volatility / mean-reverting
    - 2: High volatility / crisis
    """

    REGIME_NAMES = {
        0: "low_vol_trending",
        1: "medium_vol_normal",
        2: "high_vol_crisis",
    }

    def __init__(self, n_regimes: int = 3, config: dict | None = None):
        """
        Initialize HMM regime detector.

        Args:
            n_regimes: Number of hidden regimes
            config: Model configuration
        """
        if not HMM_AVAILABLE:
            logger.warning("hmmlearn not available; using fallback regime detection")

        self.n_regimes = n_regimes
        self.config = config or {}
        self.model: GaussianHMM | None = None
        self._is_fitted = False

        # Feature parameters
        self.vol_window = self.config.get("vol_window", 20)
        self.skew_window = self.config.get("skew_window", 20)

    def fit(self, returns: np.ndarray, n_iter: int = 100) -> None:
        """
        Fit HMM to historical returns.

        Args:
            returns: Historical returns (n_samples,)
            n_iter: Maximum EM iterations
        """
        if not HMM_AVAILABLE:
            logger.warning("hmmlearn not available; skipping HMM fit")
            self._is_fitted = True
            return

        features = self._compute_features(returns)

        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=n_iter,
            random_state=42,
        )

        try:
            self.model.fit(features)
            self._is_fitted = True
            logger.info(
                "RegimeHMM fitted: n_regimes=%d converged=%s",
                self.n_regimes,
                self.model.monitor_.converged,
            )
        except Exception as e:
            logger.warning("HMM fitting failed: %s", e)
            self._is_fitted = False

    def predict_regime(self, returns: np.ndarray) -> int:
        """
        Predict current regime.

        Args:
            returns: Recent returns (n_samples,)

        Returns:
            Regime label (0 to n_regimes-1)
        """
        if not self._is_fitted:
            return self._fallback_regime(returns)

        if not HMM_AVAILABLE or self.model is None:
            return self._fallback_regime(returns)

        features = self._compute_features(returns)
        if len(features) == 0:
            return 1  # Default to medium regime

        try:
            regimes = self.model.predict(features)
            return int(regimes[-1])
        except Exception as e:
            logger.warning("Regime prediction failed: %s", e)
            return self._fallback_regime(returns)

    def regime_probabilities(self, returns: np.ndarray) -> np.ndarray:
        """
        Get probability distribution over regimes.

        Args:
            returns: Recent returns

        Returns:
            Probabilities array (n_regimes,)
        """
        if not self._is_fitted or not HMM_AVAILABLE or self.model is None:
            return self._fallback_probs(returns)

        features = self._compute_features(returns)
        if len(features) == 0:
            return np.ones(self.n_regimes) / self.n_regimes

        try:
            probs = self.model.predict_proba(features)
            return probs[-1]
        except Exception:
            return self._fallback_probs(returns)

    def get_state(self, returns: np.ndarray) -> RegimeState:
        """
        Get comprehensive regime state.

        Args:
            returns: Recent returns

        Returns:
            RegimeState with regime, probabilities, and features
        """
        regime = self.predict_regime(returns)
        probs = self.regime_probabilities(returns)

        # Compute display features
        vol = np.std(returns[-self.vol_window:]) * np.sqrt(252) if len(returns) >= self.vol_window else 0.0
        ret = np.mean(returns[-self.vol_window:]) * 252 if len(returns) >= self.vol_window else 0.0

        return RegimeState(
            regime=regime,
            regime_name=self.REGIME_NAMES.get(regime, f"regime_{regime}"),
            probability=float(probs[regime]),
            regime_probs=[float(p) for p in probs],
            features={"volatility": vol, "return": ret},
        )

    def _compute_features(self, returns: np.ndarray) -> np.ndarray:
        """Compute features for HMM: [return, volatility, skewness]."""
        if len(returns) < max(self.vol_window, self.skew_window):
            return np.array([]).reshape(-1, 3)

        # Rolling features
        n = len(returns)
        features = []

        for i in range(self.vol_window, n):
            window = returns[i - self.vol_window:i]
            ret = returns[i]
            vol = np.std(window)
            skew = self._rolling_skewness(window)
            features.append([ret, vol, skew])

        return np.array(features)

    def _rolling_skewness(self, window: np.ndarray) -> float:
        """Compute skewness of a window."""
        n = len(window)
        if n < 3:
            return 0.0
        mean = np.mean(window)
        std = np.std(window)
        if std < 1e-10:
            return 0.0
        return float(np.mean(((window - mean) / std) ** 3))

    def _fallback_regime(self, returns: np.ndarray) -> int:
        """Simple fallback regime detection without HMM."""
        if len(returns) < 20:
            return 1

        vol = np.std(returns[-20:])
        vol_percentile = min(vol / 0.02, 1.0)  # Normalize by typical vol

        if vol_percentile < 0.33:
            return 0  # Low vol
        elif vol_percentile < 0.67:
            return 1  # Medium vol
        else:
            return 2  # High vol

    def _fallback_probs(self, returns: np.ndarray) -> np.ndarray:
        """Fallback probability distribution."""
        regime = self._fallback_regime(returns)
        probs = np.ones(self.n_regimes) * 0.1 / (self.n_regimes - 1)
        probs[regime] = 0.9
        return probs


class ChangepointDetector:
    """
    Detects structural breaks (changepoints) in time series.

    Uses simple methods that don't require external libraries.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize changepoint detector.

        Args:
            config: Detector configuration
        """
        self.config = config or {}
        self.min_segment = self.config.get("min_segment", 20)
        self.threshold = self.config.get("threshold", 2.0)  # Z-score threshold

    def detect(self, series: np.ndarray) -> list[int]:
        """
        Detect changepoints in a time series.

        Uses CUSUM-based detection.

        Args:
            series: Time series values

        Returns:
            List of changepoint indices
        """
        n = len(series)
        if n < 2 * self.min_segment:
            return []

        changepoints = []

        # CUSUM detection
        mean = np.mean(series)
        std = np.std(series) + 1e-10
        cusum_pos = np.zeros(n)
        cusum_neg = np.zeros(n)

        for i in range(1, n):
            z = (series[i] - mean) / std
            cusum_pos[i] = max(0, cusum_pos[i - 1] + z - 0.5)
            cusum_neg[i] = max(0, cusum_neg[i - 1] - z - 0.5)

            # Check for changepoint
            if cusum_pos[i] > self.threshold or cusum_neg[i] > self.threshold:
                # Ensure minimum segment length
                if not changepoints or i - changepoints[-1] >= self.min_segment:
                    changepoints.append(i)
                    # Reset CUSUM
                    cusum_pos[i] = 0
                    cusum_neg[i] = 0

        return changepoints

    def detect_mean_shift(self, series: np.ndarray, window: int = 20) -> list[int]:
        """
        Detect mean shifts using rolling window comparison.

        Args:
            series: Time series values
            window: Window size for comparison

        Returns:
            Indices where significant mean shifts occurred
        """
        n = len(series)
        if n < 2 * window:
            return []

        changepoints = []

        for i in range(window, n - window):
            left_mean = np.mean(series[i - window:i])
            right_mean = np.mean(series[i:i + window])
            pooled_std = np.std(series[i - window:i + window]) + 1e-10

            # T-test statistic
            t_stat = abs(right_mean - left_mean) / (pooled_std * np.sqrt(2 / window))

            if t_stat > self.threshold:
                if not changepoints or i - changepoints[-1] >= self.min_segment:
                    changepoints.append(i)

        return changepoints

    def recent_changepoint(self, series: np.ndarray, lookback: int = 100) -> bool:
        """
        Check if there was a recent changepoint.

        Args:
            series: Time series values
            lookback: Lookback window

        Returns:
            True if recent changepoint detected
        """
        if len(series) < lookback:
            recent = series
        else:
            recent = series[-lookback:]

        cps = self.detect(recent)
        # Check if any changepoint in last 20% of lookback
        threshold = int(0.8 * len(recent))
        return any(cp > threshold for cp in cps)


class RegimeAwareStrategy:
    """
    Adapts strategy behavior based on detected market regime.

    Wraps multiple strategy variants and selects based on regime.
    """

    def __init__(
        self,
        regime_detector: RegimeHMM,
        strategies: dict[int, Callable],
        config: dict | None = None,
    ):
        """
        Initialize regime-aware strategy.

        Args:
            regime_detector: Regime detection model
            strategies: Dict mapping regime ID to strategy function
            config: Strategy configuration
        """
        self.regime_detector = regime_detector
        self.strategies = strategies
        self.config = config or {}

        # Regime override
        self.force_regime: int | None = None

    def generate_signal(self, market_state: dict) -> dict:
        """
        Generate trading signal using regime-appropriate strategy.

        Args:
            market_state: Current market state with prices, etc.

        Returns:
            Signal dict with action, size, regime info
        """
        prices = market_state.get("prices", [])
        if len(prices) < 2:
            return {"action": "hold", "regime": -1}

        returns = np.diff(prices) / prices[:-1]

        # Get regime
        if self.force_regime is not None:
            regime = self.force_regime
        else:
            regime = self.regime_detector.predict_regime(returns)

        # Select strategy
        strategy = self.strategies.get(regime)
        if strategy is None:
            # Fallback to hold
            return {"action": "hold", "regime": regime, "reason": "no_strategy_for_regime"}

        # Generate signal with selected strategy
        signal = strategy(market_state)
        signal["regime"] = regime
        signal["regime_name"] = RegimeHMM.REGIME_NAMES.get(regime, f"regime_{regime}")

        return signal

    def get_regime_state(self, market_state: dict) -> RegimeState:
        """Get current regime state."""
        prices = market_state.get("prices", [])
        if len(prices) < 2:
            return RegimeState(
                regime=1,
                regime_name="unknown",
                probability=0.0,
                regime_probs=[1/3, 1/3, 1/3],
                features={},
            )

        returns = np.diff(prices) / prices[:-1]
        return self.regime_detector.get_state(returns)


def create_regime_strategies() -> dict[int, Callable]:
    """
    Create default regime-specific strategy variants.

    Returns:
        Dict mapping regime to strategy function
    """
    def low_vol_strategy(state: dict) -> dict:
        """Trending/momentum strategy for low volatility."""
        return {
            "action": "follow_trend",
            "position_scale": 1.2,
            "stop_loss_pct": 2.0,
            "take_profit_pct": 5.0,
        }

    def medium_vol_strategy(state: dict) -> dict:
        """Mean reversion strategy for normal conditions."""
        return {
            "action": "mean_revert",
            "position_scale": 1.0,
            "stop_loss_pct": 1.5,
            "take_profit_pct": 3.0,
        }

    def high_vol_strategy(state: dict) -> dict:
        """Defensive strategy for high volatility/crisis."""
        return {
            "action": "reduce_exposure",
            "position_scale": 0.5,
            "stop_loss_pct": 1.0,
            "take_profit_pct": 2.0,
        }

    return {
        0: low_vol_strategy,
        1: medium_vol_strategy,
        2: high_vol_strategy,
    }
