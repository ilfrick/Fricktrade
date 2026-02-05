# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""
Base predictor interface for ensemble methods.

All ensemble predictors implement this interface to ensure consistent
prediction and training APIs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BasePredictor(ABC):
    """
    Abstract base class for ensemble predictors.

    Each predictor provides:
    - fit(): Train on historical data
    - predict(): Generate predictions for new data
    - predict_proba(): Generate probability distributions (optional)
    """

    def __init__(self, name: str, config: dict | None = None):
        self.name = name
        self.config = config or {}
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the predictor on historical data.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for new data.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predictions of shape (n_samples,)
        """
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """
        Generate probability distributions for classification.

        Default implementation returns None (not supported).
        Override in subclasses for classifiers.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Probabilities of shape (n_samples, n_classes) or None
        """
        return None

    @property
    def is_fitted(self) -> bool:
        """Check if the predictor has been trained."""
        return self._is_fitted

    def get_feature_importance(self) -> dict[str, float] | None:
        """
        Get feature importance scores if available.

        Returns:
            Dict mapping feature names to importance scores, or None
        """
        return None

    def save(self, path: str) -> None:
        """Save the predictor to disk."""
        raise NotImplementedError(f"{self.name} does not support save()")

    def load(self, path: str) -> None:
        """Load the predictor from disk."""
        raise NotImplementedError(f"{self.name} does not support load()")


class ReturnPredictor(BasePredictor):
    """
    Base class for return prediction models.

    Predicts future returns (continuous values).
    """

    @abstractmethod
    def predict_return(self, X: np.ndarray, horizon: int = 1) -> np.ndarray:
        """
        Predict future returns.

        Args:
            X: Feature matrix
            horizon: Prediction horizon in bars

        Returns:
            Predicted returns
        """
        pass


class DirectionPredictor(BasePredictor):
    """
    Base class for direction prediction models.

    Predicts price direction (up/down/flat).
    """

    DIRECTIONS = {0: "down", 1: "flat", 2: "up"}

    @abstractmethod
    def predict_direction(self, X: np.ndarray) -> np.ndarray:
        """
        Predict price direction.

        Args:
            X: Feature matrix

        Returns:
            Direction labels (0=down, 1=flat, 2=up)
        """
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Get probability distribution over directions."""
        return None


class SignalPredictor(BasePredictor):
    """
    Base class for trading signal models.

    Generates trading signals (buy/hold/sell).
    """

    SIGNALS = {0: "sell", 1: "hold", 2: "buy"}

    @abstractmethod
    def predict_signal(self, X: np.ndarray) -> np.ndarray:
        """
        Generate trading signals.

        Args:
            X: Feature matrix

        Returns:
            Signal labels (0=sell, 1=hold, 2=buy)
        """
        pass

    def predict_confidence(self, X: np.ndarray) -> np.ndarray | None:
        """
        Get confidence scores for signals.

        Returns:
            Confidence in [0, 1] or None
        """
        return None
