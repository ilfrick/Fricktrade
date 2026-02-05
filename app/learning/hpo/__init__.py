# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""
Hyperparameter optimization for RL trading models.

Uses Optuna for Bayesian optimization of PPO, TCN, reward, and
environment parameters.

Usage:
    from app.learning.hpo import run_hpo, apply_best_params

    # Run optimization
    results = run_hpo(
        cfg=config,
        datasets=train_data,
        n_trials=50,
        study_name="ppo_tcn_hpo",
    )

    # Apply best params
    optimized_cfg = apply_best_params(config, results["best_params"])
"""

from app.learning.hpo.search_space import (
    suggest_ppo_params,
    suggest_tcn_params,
    suggest_reward_params,
    suggest_env_params,
    suggest_all_params,
    SEARCH_SPACES,
)
from app.learning.hpo.optimize import (
    HPOObjective,
    run_hpo,
    apply_best_params,
)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

__all__ = [
    # Search space
    "suggest_ppo_params",
    "suggest_tcn_params",
    "suggest_reward_params",
    "suggest_env_params",
    "suggest_all_params",
    "SEARCH_SPACES",
    # Optimization
    "HPOObjective",
    "run_hpo",
    "apply_best_params",
    # Availability
    "OPTUNA_AVAILABLE",
]
