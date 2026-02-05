# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""
LightGBM-based predictors for ensemble methods.

LightGBM provides fast, accurate gradient boosting that complements
neural network approaches with different inductive biases.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from app.learning.ensemble.base import ReturnPredictor, DirectionPredictor, SignalPredictor

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    lgb = None

logger = logging.getLogger(__name__)


class LGBMReturnPredictor(ReturnPredictor):
    """
    LightGBM regressor for return prediction.

    Predicts future returns using gradient boosting on technical features.
    """

    def __init__(self, name: str = "lgbm_return", config: dict | None = None):
        super().__init__(name, config)

        if not LGBM_AVAILABLE:
            raise ImportError("lightgbm is required for LGBMReturnPredictor")

        self.model: lgb.LGBMRegressor | None = None
        self._feature_names: list[str] | None = None

        # Default hyperparameters optimized for financial data
        self._params = {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "max_depth": -1,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }
        self._params.update(config.get("lgbm_params", {}) if config else {})

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """
        Train the LightGBM regressor.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target returns (n_samples,)
            feature_names: Optional feature names for interpretability
            eval_set: Optional validation set (X_val, y_val)
        """
        self._feature_names = feature_names

        self.model = lgb.LGBMRegressor(**self._params)

        callbacks = []
        if eval_set is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=20, verbose=False))

        fit_params: dict[str, Any] = {}
        if eval_set is not None:
            fit_params["eval_set"] = [eval_set]

        self.model.fit(
            X, y,
            feature_name=feature_names or "auto",
            callbacks=callbacks if callbacks else None,
            **fit_params,
        )
        self._is_fitted = True

        logger.info(
            "LGBMReturnPredictor trained: samples=%d features=%d best_iter=%s",
            len(y),
            X.shape[1],
            getattr(self.model, "best_iteration_", "N/A"),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict returns."""
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def predict_return(self, X: np.ndarray, horizon: int = 1) -> np.ndarray:
        """Predict future returns (alias for predict)."""
        return self.predict(X)

    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importance scores."""
        if not self._is_fitted or self.model is None:
            return None

        importance = self.model.feature_importances_
        if self._feature_names:
            return dict(zip(self._feature_names, importance))
        return {f"feature_{i}": v for i, v in enumerate(importance)}

    def save(self, path: str) -> None:
        """Save model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "params": self._params,
                "feature_names": self._feature_names,
            }, f)
        logger.info("Saved LGBMReturnPredictor to %s", path)

    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self._params = data["params"]
        self._feature_names = data["feature_names"]
        self._is_fitted = True
        logger.info("Loaded LGBMReturnPredictor from %s", path)


class LGBMDirectionPredictor(DirectionPredictor):
    """
    LightGBM classifier for direction prediction.

    Predicts whether price will go up, down, or stay flat.
    """

    def __init__(self, name: str = "lgbm_direction", config: dict | None = None):
        super().__init__(name, config)

        if not LGBM_AVAILABLE:
            raise ImportError("lightgbm is required for LGBMDirectionPredictor")

        self.model: lgb.LGBMClassifier | None = None
        self._feature_names: list[str] | None = None

        # Default hyperparameters
        self._params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "max_depth": -1,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }
        self._params.update(config.get("lgbm_params", {}) if config else {})

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> None:
        """
        Train the direction classifier.

        Args:
            X: Feature matrix
            y: Direction labels (0=down, 1=flat, 2=up)
            feature_names: Optional feature names
        """
        self._feature_names = feature_names
        self.model = lgb.LGBMClassifier(**self._params)
        self.model.fit(X, y, feature_name=feature_names or "auto")
        self._is_fitted = True

        logger.info(
            "LGBMDirectionPredictor trained: samples=%d features=%d",
            len(y),
            X.shape[1],
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict direction labels."""
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X)

    def predict_direction(self, X: np.ndarray) -> np.ndarray:
        """Predict price direction (alias)."""
        return self.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability distribution over directions."""
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        """Save model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "params": self._params,
                "feature_names": self._feature_names,
            }, f)

    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self._params = data["params"]
        self._feature_names = data["feature_names"]
        self._is_fitted = True


class LGBMSignalPredictor(SignalPredictor):
    """
    LightGBM classifier for trading signal prediction.

    Directly predicts buy/hold/sell signals.
    """

    def __init__(self, name: str = "lgbm_signal", config: dict | None = None):
        super().__init__(name, config)

        if not LGBM_AVAILABLE:
            raise ImportError("lightgbm is required for LGBMSignalPredictor")

        self.model: lgb.LGBMClassifier | None = None
        self._feature_names: list[str] | None = None

        self._params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "class_weight": "balanced",  # Handle imbalanced signals
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }
        self._params.update(config.get("lgbm_params", {}) if config else {})

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> None:
        """Train signal classifier."""
        self._feature_names = feature_names
        self.model = lgb.LGBMClassifier(**self._params)
        self.model.fit(X, y, feature_name=feature_names or "auto")
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict signal labels."""
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X)

    def predict_signal(self, X: np.ndarray) -> np.ndarray:
        """Predict trading signals (alias)."""
        return self.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability distribution over signals."""
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict_proba(X)

    def predict_confidence(self, X: np.ndarray) -> np.ndarray:
        """Get confidence scores (max probability)."""
        proba = self.predict_proba(X)
        return np.max(proba, axis=1)

    def save(self, path: str) -> None:
        """Save model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "params": self._params,
                "feature_names": self._feature_names,
            }, f)

    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self._params = data["params"]
        self._feature_names = data["feature_names"]
        self._is_fitted = True
