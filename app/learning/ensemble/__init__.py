# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""
Ensemble learning methods for trading signal aggregation.

This module provides:
- Base predictor interfaces (ReturnPredictor, DirectionPredictor, SignalPredictor)
- LightGBM-based predictors for return/direction/signal prediction
- Ensemble aggregation with adaptive meta-learning
- Stacking ensemble with meta-model

Usage:
    from app.learning.ensemble import (
        LGBMReturnPredictor,
        LGBMDirectionPredictor,
        EnsemblePredictor,
        MetaLearner,
    )

    # Create predictors
    lgbm_return = LGBMReturnPredictor()
    lgbm_direction = LGBMDirectionPredictor()

    # Create ensemble with meta-learning
    ensemble = EnsemblePredictor(
        predictors=[lgbm_return, lgbm_direction],
        use_meta_learning=True,
    )

    # Train
    ensemble.fit(X_train, y_train)

    # Predict
    predictions = ensemble.predict(X_test)
"""

from app.learning.ensemble.base import (
    BasePredictor,
    ReturnPredictor,
    DirectionPredictor,
    SignalPredictor,
)
from app.learning.ensemble.ensemble import (
    MetaLearner,
    EnsemblePredictor,
    StackingEnsemble,
)

# Conditional imports for optional dependencies
try:
    from app.learning.ensemble.lgbm_predictor import (
        LGBMReturnPredictor,
        LGBMDirectionPredictor,
        LGBMSignalPredictor,
    )
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    LGBMReturnPredictor = None
    LGBMDirectionPredictor = None
    LGBMSignalPredictor = None

__all__ = [
    # Base classes
    "BasePredictor",
    "ReturnPredictor",
    "DirectionPredictor",
    "SignalPredictor",
    # Ensemble
    "MetaLearner",
    "EnsemblePredictor",
    "StackingEnsemble",
    # LightGBM (optional)
    "LGBMReturnPredictor",
    "LGBMDirectionPredictor",
    "LGBMSignalPredictor",
    "LGBM_AVAILABLE",
]
