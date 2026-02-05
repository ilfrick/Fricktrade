# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""
Hyperparameter search space definitions for Optuna optimization.

Defines search spaces for PPO, TCN, and reward parameters.
"""

from __future__ import annotations

from typing import Any

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None


def suggest_ppo_params(trial: "optuna.Trial") -> dict[str, Any]:
    """
    Suggest PPO hyperparameters.

    Args:
        trial: Optuna trial object

    Returns:
        Dict of PPO hyperparameters
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999, log=True),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 0.1, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
        "n_epochs": trial.suggest_int("n_epochs", 3, 20),
    }


def suggest_tcn_params(trial: "optuna.Trial") -> dict[str, Any]:
    """
    Suggest TCN architecture hyperparameters.

    Args:
        trial: Optuna trial object

    Returns:
        Dict of TCN hyperparameters
    """
    return {
        "features_dim": trial.suggest_categorical("tcn_features_dim", [32, 64, 128, 256]),
        "num_layers": trial.suggest_int("tcn_num_layers", 2, 5),
        "kernel_size": trial.suggest_categorical("tcn_kernel_size", [2, 3, 5]),
        "dropout": trial.suggest_float("tcn_dropout", 0.0, 0.3),
    }


def suggest_reward_params(trial: "optuna.Trial") -> dict[str, Any]:
    """
    Suggest reward function hyperparameters.

    Args:
        trial: Optuna trial object

    Returns:
        Dict of reward hyperparameters
    """
    return {
        "nav_weight": trial.suggest_float("nav_weight", 0.5, 2.0),
        "time_penalty_weight": trial.suggest_float("time_penalty_weight", 0.5, 3.0),
        "profit_bonus_weight": trial.suggest_float("profit_bonus_weight", 1.0, 5.0),
        "velocity_weight": trial.suggest_float("velocity_weight", 0.0, 1.0),
        "time_normalizer": trial.suggest_categorical("time_normalizer", [60, 120, 240, 390]),
    }


def suggest_env_params(trial: "optuna.Trial") -> dict[str, Any]:
    """
    Suggest environment hyperparameters.

    Args:
        trial: Optuna trial object

    Returns:
        Dict of environment hyperparameters
    """
    return {
        "window_size": trial.suggest_categorical("window_size", [20, 30, 50, 100]),
        "commission_pct": trial.suggest_float("commission_pct", 0.01, 0.1),
        "slippage_bps": trial.suggest_int("slippage_bps", 1, 10),
    }


def suggest_all_params(
    trial: "optuna.Trial",
    include_ppo: bool = True,
    include_tcn: bool = True,
    include_reward: bool = True,
    include_env: bool = True,
) -> dict[str, Any]:
    """
    Suggest all hyperparameters for full optimization.

    Args:
        trial: Optuna trial object
        include_ppo: Include PPO parameters
        include_tcn: Include TCN parameters
        include_reward: Include reward parameters
        include_env: Include environment parameters

    Returns:
        Combined dict of all hyperparameters
    """
    params = {}

    if include_ppo:
        params["ppo"] = suggest_ppo_params(trial)

    if include_tcn:
        params["tcn"] = suggest_tcn_params(trial)

    if include_reward:
        params["reward"] = suggest_reward_params(trial)

    if include_env:
        params["env"] = suggest_env_params(trial)

    return params


# Pre-defined search space configurations
SEARCH_SPACES = {
    "minimal": {
        "description": "Quick search over most impactful parameters",
        "params": ["learning_rate", "n_steps", "gamma", "ent_coef"],
    },
    "ppo_only": {
        "description": "Full PPO hyperparameter search",
        "params": list(suggest_ppo_params.__code__.co_varnames)[:10],
    },
    "architecture": {
        "description": "TCN architecture search",
        "params": ["tcn_features_dim", "tcn_num_layers", "tcn_kernel_size", "tcn_dropout"],
    },
    "reward": {
        "description": "Reward function tuning",
        "params": ["nav_weight", "time_penalty_weight", "profit_bonus_weight", "velocity_weight"],
    },
    "full": {
        "description": "Full hyperparameter search (slow)",
        "params": "all",
    },
}
