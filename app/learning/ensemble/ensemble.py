# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""
Ensemble aggregation with online meta-learning.

Combines predictions from multiple models using adaptive weighting
based on recent performance.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Callable

import numpy as np

from app.learning.ensemble.base import BasePredictor

logger = logging.getLogger(__name__)


class MetaLearner:
    """
    Online learning of ensemble weights based on recent performance.

    Uses inverse-error weighting with exponential smoothing to adapt
    weights as market conditions change.
    """

    def __init__(
        self,
        predictor_names: list[str],
        lookback: int = 100,
        smoothing: float = 0.1,
        min_weight: float = 0.05,
    ):
        """
        Initialize meta-learner.

        Args:
            predictor_names: Names of predictors to weight
            lookback: Number of recent samples to consider
            smoothing: Exponential smoothing factor (0-1, higher = faster adaptation)
            min_weight: Minimum weight for any predictor
        """
        self.predictor_names = predictor_names
        self.lookback = lookback
        self.smoothing = smoothing
        self.min_weight = min_weight

        # Performance tracking
        self._errors: dict[str, deque] = {
            name: deque(maxlen=lookback) for name in predictor_names
        }
        self._weights: dict[str, float] = {
            name: 1.0 / len(predictor_names) for name in predictor_names
        }

    def update(self, predictions: dict[str, float], actual: float) -> dict[str, float]:
        """
        Update weights based on prediction errors.

        Args:
            predictions: Dict mapping predictor name to prediction
            actual: Actual observed value

        Returns:
            Updated weights
        """
        # Track errors
        for name, pred in predictions.items():
            if name in self._errors:
                error = (pred - actual) ** 2
                self._errors[name].append(error)

        # Compute new weights using inverse error
        new_weights = self._compute_weights()

        # Exponential smoothing
        for name in self.predictor_names:
            self._weights[name] = (
                self.smoothing * new_weights[name]
                + (1 - self.smoothing) * self._weights[name]
            )

        # Normalize
        total = sum(self._weights.values())
        if total > 0:
            self._weights = {k: v / total for k, v in self._weights.items()}

        return self._weights.copy()

    def _compute_weights(self) -> dict[str, float]:
        """Compute weights using inverse error."""
        avg_errors = {}
        for name, errors in self._errors.items():
            if len(errors) > 0:
                avg_errors[name] = np.mean(errors) + 1e-8
            else:
                avg_errors[name] = 1.0

        # Inverse error weighting
        inv_errors = {name: 1.0 / err for name, err in avg_errors.items()}
        total = sum(inv_errors.values())

        weights = {}
        for name in self.predictor_names:
            w = inv_errors[name] / total if total > 0 else 1.0 / len(self.predictor_names)
            weights[name] = max(w, self.min_weight)

        # Renormalize after applying min_weight
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    @property
    def weights(self) -> dict[str, float]:
        """Current ensemble weights."""
        return self._weights.copy()

    def reset(self) -> None:
        """Reset to uniform weights."""
        for name in self.predictor_names:
            self._errors[name].clear()
            self._weights[name] = 1.0 / len(self.predictor_names)


class EnsemblePredictor:
    """
    Ensemble that combines multiple predictors with adaptive weighting.

    Supports:
    - Static weights (fixed proportions)
    - Dynamic weights (meta-learning based on performance)
    - Various aggregation methods (weighted mean, median, voting)
    """

    def __init__(
        self,
        predictors: list[BasePredictor],
        weights: dict[str, float] | None = None,
        use_meta_learning: bool = True,
        aggregation: str = "weighted_mean",
        meta_lookback: int = 100,
    ):
        """
        Initialize ensemble.

        Args:
            predictors: List of predictor instances
            weights: Optional fixed weights (use uniform if None)
            use_meta_learning: Whether to adapt weights online
            aggregation: Aggregation method (weighted_mean, median, voting)
            meta_lookback: Lookback window for meta-learning
        """
        self.predictors = {p.name: p for p in predictors}
        self.aggregation = aggregation
        self.use_meta_learning = use_meta_learning

        # Initialize weights
        names = list(self.predictors.keys())
        if weights:
            self._weights = weights.copy()
        else:
            self._weights = {name: 1.0 / len(names) for name in names}

        # Meta-learner for adaptive weights
        self._meta_learner = MetaLearner(names, lookback=meta_lookback) if use_meta_learning else None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train all predictors in the ensemble.

        Args:
            X: Feature matrix
            y: Target values
            **kwargs: Additional arguments passed to each predictor
        """
        for name, predictor in self.predictors.items():
            logger.info("Training ensemble member: %s", name)
            predictor.fit(X, y, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate ensemble predictions.

        Args:
            X: Feature matrix

        Returns:
            Aggregated predictions
        """
        predictions = {}
        for name, predictor in self.predictors.items():
            if predictor.is_fitted:
                predictions[name] = predictor.predict(X)

        if not predictions:
            raise RuntimeError("No fitted predictors in ensemble")

        return self._aggregate(predictions)

    def predict_with_details(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Generate predictions with individual predictor outputs.

        Returns:
            Tuple of (aggregated predictions, dict of individual predictions)
        """
        predictions = {}
        for name, predictor in self.predictors.items():
            if predictor.is_fitted:
                predictions[name] = predictor.predict(X)

        aggregated = self._aggregate(predictions)
        return aggregated, predictions

    def _aggregate(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        """Aggregate predictions using configured method."""
        if self.aggregation == "weighted_mean":
            return self._weighted_mean(predictions)
        elif self.aggregation == "median":
            return self._median(predictions)
        elif self.aggregation == "voting":
            return self._voting(predictions)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

    def _weighted_mean(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        """Compute weighted mean of predictions."""
        result = None
        total_weight = 0.0

        for name, pred in predictions.items():
            weight = self._weights.get(name, 0.0)
            if weight > 0:
                if result is None:
                    result = weight * pred
                else:
                    result += weight * pred
                total_weight += weight

        if result is None or total_weight == 0:
            # Fallback to simple mean
            return np.mean(list(predictions.values()), axis=0)

        return result / total_weight

    def _median(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        """Compute median of predictions."""
        stacked = np.stack(list(predictions.values()), axis=0)
        return np.median(stacked, axis=0)

    def _voting(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        """Majority voting for classification."""
        stacked = np.stack(list(predictions.values()), axis=0)
        # Use weighted voting
        n_samples = stacked.shape[1]
        result = np.zeros(n_samples)

        for i in range(n_samples):
            votes = {}
            for name, pred in predictions.items():
                vote = int(pred[i])
                weight = self._weights.get(name, 1.0)
                votes[vote] = votes.get(vote, 0.0) + weight
            result[i] = max(votes.keys(), key=lambda k: votes[k])

        return result

    def update_weights(self, predictions: dict[str, float], actual: float) -> None:
        """
        Update ensemble weights based on observed outcome.

        Args:
            predictions: Dict mapping predictor name to its prediction
            actual: Actual observed value
        """
        if self._meta_learner:
            self._weights = self._meta_learner.update(predictions, actual)

    @property
    def weights(self) -> dict[str, float]:
        """Current ensemble weights."""
        return self._weights.copy()

    def get_feature_importance(self) -> dict[str, dict[str, float]]:
        """Get feature importance from all predictors."""
        importance = {}
        for name, predictor in self.predictors.items():
            imp = predictor.get_feature_importance()
            if imp:
                importance[name] = imp
        return importance


class StackingEnsemble:
    """
    Stacking ensemble with a meta-model.

    Uses base predictor outputs as features for a meta-model
    that learns optimal combination.
    """

    def __init__(
        self,
        base_predictors: list[BasePredictor],
        meta_predictor: BasePredictor,
        use_original_features: bool = True,
    ):
        """
        Initialize stacking ensemble.

        Args:
            base_predictors: First-level predictors
            meta_predictor: Second-level model that combines base predictions
            use_original_features: Include original features in meta-model input
        """
        self.base_predictors = {p.name: p for p in base_predictors}
        self.meta_predictor = meta_predictor
        self.use_original_features = use_original_features

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the stacking ensemble.

        Uses cross-validation to generate out-of-fold predictions
        for training the meta-model.
        """
        from sklearn.model_selection import KFold

        n_samples = len(y)
        n_predictors = len(self.base_predictors)

        # Generate OOF predictions
        oof_predictions = np.zeros((n_samples, n_predictors))
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            for i, (name, predictor) in enumerate(self.base_predictors.items()):
                # Clone and train on fold
                predictor.fit(X_train, y_train)
                oof_predictions[val_idx, i] = predictor.predict(X_val)

        # Train base predictors on full data
        for predictor in self.base_predictors.values():
            predictor.fit(X, y)

        # Prepare meta-features
        if self.use_original_features:
            meta_X = np.hstack([X, oof_predictions])
        else:
            meta_X = oof_predictions

        # Train meta-model
        self.meta_predictor.fit(meta_X, y)

        logger.info(
            "StackingEnsemble trained: base=%d meta_features=%d",
            len(self.base_predictors),
            meta_X.shape[1],
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate stacked predictions."""
        # Get base predictions
        base_preds = []
        for predictor in self.base_predictors.values():
            base_preds.append(predictor.predict(X))
        base_preds = np.column_stack(base_preds)

        # Prepare meta-features
        if self.use_original_features:
            meta_X = np.hstack([X, base_preds])
        else:
            meta_X = base_preds

        return self.meta_predictor.predict(meta_X)
