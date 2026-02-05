# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""
Optuna-based hyperparameter optimization for RL trading models.

Provides automated tuning of PPO, TCN, reward, and environment parameters
using Bayesian optimization with Optuna.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

from app.learning.hpo.search_space import (
    suggest_ppo_params,
    suggest_tcn_params,
    suggest_reward_params,
    suggest_env_params,
)

logger = logging.getLogger(__name__)


class HPOObjective:
    """
    Optuna objective function for RL model optimization.

    Trains a model with suggested hyperparameters and evaluates
    using Sharpe ratio as the primary metric.
    """

    def __init__(
        self,
        base_cfg: dict,
        datasets: list,
        eval_datasets: list | None = None,
        timesteps: int = 10000,
        n_eval_episodes: int = 5,
        optimize_tcn: bool = True,
        optimize_reward: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize HPO objective.

        Args:
            base_cfg: Base configuration dict
            datasets: Training datasets
            eval_datasets: Evaluation datasets (use train if None)
            timesteps: Training timesteps per trial
            n_eval_episodes: Evaluation episodes
            optimize_tcn: Include TCN parameters in search
            optimize_reward: Include reward parameters in search
            device: Training device
        """
        self.base_cfg = base_cfg
        self.datasets = datasets
        self.eval_datasets = eval_datasets or datasets
        self.timesteps = timesteps
        self.n_eval_episodes = n_eval_episodes
        self.optimize_tcn = optimize_tcn
        self.optimize_reward = optimize_reward
        self.device = device

    def __call__(self, trial: "optuna.Trial") -> float:
        """
        Evaluate a set of hyperparameters.

        Args:
            trial: Optuna trial with suggested parameters

        Returns:
            Sharpe ratio (higher is better)
        """
        try:
            # Suggest hyperparameters
            ppo_params = suggest_ppo_params(trial)
            tcn_params = suggest_tcn_params(trial) if self.optimize_tcn else None
            reward_params = suggest_reward_params(trial) if self.optimize_reward else None

            # Build config
            cfg = self._build_config(ppo_params, tcn_params, reward_params)

            # Train model
            model, metrics = self._train_and_evaluate(cfg, trial)

            # Report intermediate values for pruning
            sharpe = metrics.get("sharpe", 0.0)
            trial.report(sharpe, step=self.timesteps)

            if trial.should_prune():
                raise optuna.TrialPruned()

            # Store additional metrics as user attributes
            trial.set_user_attr("return_pct", metrics.get("return_pct", 0.0))
            trial.set_user_attr("max_drawdown", metrics.get("max_drawdown_pct", 0.0))
            trial.set_user_attr("win_rate", metrics.get("win_rate", 0.0))

            return sharpe

        except Exception as e:
            logger.warning("Trial %d failed: %s", trial.number, e)
            return float("-inf")

    def _build_config(
        self,
        ppo_params: dict,
        tcn_params: dict | None,
        reward_params: dict | None,
    ) -> dict:
        """Build configuration from suggested parameters."""
        import copy
        cfg = copy.deepcopy(self.base_cfg)

        # Apply PPO params
        learning_cfg = cfg.setdefault("learning", {})
        learning_cfg["learning_rate"] = ppo_params["learning_rate"]
        learning_cfg["n_steps"] = ppo_params["n_steps"]
        learning_cfg["batch_size"] = ppo_params["batch_size"]
        learning_cfg["gamma"] = ppo_params["gamma"]
        learning_cfg["gae_lambda"] = ppo_params["gae_lambda"]
        learning_cfg["ent_coef"] = ppo_params["ent_coef"]
        learning_cfg["clip_range"] = ppo_params["clip_range"]

        # Apply TCN params
        if tcn_params:
            tcn_cfg = learning_cfg.setdefault("tcn", {})
            tcn_cfg["enabled"] = True
            tcn_cfg.update(tcn_params)

        # Apply reward params
        if reward_params:
            reward_cfg = learning_cfg.setdefault("reward", {})
            reward_cfg.update(reward_params)

        return cfg

    def _train_and_evaluate(
        self,
        cfg: dict,
        trial: "optuna.Trial",
    ) -> tuple[Any, dict]:
        """Train model and evaluate performance."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        from app.learning.env import TradingEnv
        from app.learning.networks import create_tcn_policy_kwargs

        learning_cfg = cfg.get("learning", {})
        feature_config = learning_cfg.get("features", {})
        reward_config = learning_cfg.get("reward", {})
        window_size = learning_cfg.get("window_size", 50)

        # Create environments
        envs = []
        for df in self.datasets:
            envs.append(
                lambda data=df: TradingEnv(
                    data=data,
                    window_size=window_size,
                    initial_cash=cfg.get("backtest", {}).get("initial_cash", 10000),
                    commission_pct=cfg.get("backtest", {}).get("commission_pct", 0.05),
                    slippage_bps=cfg.get("backtest", {}).get("slippage_bps", 2),
                    feature_config=feature_config,
                    reward_config=reward_config,
                )
            )
        vec_env = DummyVecEnv(envs)

        # Build policy kwargs
        tcn_cfg = learning_cfg.get("tcn", {})
        if tcn_cfg.get("enabled", False):
            policy_kwargs = create_tcn_policy_kwargs(
                features_dim=tcn_cfg.get("features_dim", 64),
                num_layers=tcn_cfg.get("num_layers", 3),
                kernel_size=tcn_cfg.get("kernel_size", 3),
                dropout=tcn_cfg.get("dropout", 0.1),
            )
        else:
            policy_kwargs = None

        # Create and train model
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=learning_cfg.get("learning_rate", 3e-4),
            n_steps=learning_cfg.get("n_steps", 2048),
            batch_size=learning_cfg.get("batch_size", 64),
            gamma=learning_cfg.get("gamma", 0.99),
            gae_lambda=learning_cfg.get("gae_lambda", 0.95),
            ent_coef=learning_cfg.get("ent_coef", 0.01),
            clip_range=learning_cfg.get("clip_range", 0.2),
            verbose=0,
            device=self.device,
            policy_kwargs=policy_kwargs,
        )

        model.learn(total_timesteps=self.timesteps)

        # Evaluate
        metrics = self._evaluate(model, window_size, feature_config, reward_config, cfg)
        return model, metrics

    def _evaluate(
        self,
        model,
        window_size: int,
        feature_config: dict,
        reward_config: dict,
        cfg: dict,
    ) -> dict:
        """Evaluate model on evaluation datasets."""
        from app.learning.env import TradingEnv

        returns = []
        for df in self.eval_datasets:
            env = TradingEnv(
                data=df,
                window_size=window_size,
                initial_cash=cfg.get("backtest", {}).get("initial_cash", 10000),
                commission_pct=cfg.get("backtest", {}).get("commission_pct", 0.05),
                slippage_bps=cfg.get("backtest", {}).get("slippage_bps", 2),
                feature_config=feature_config,
                reward_config=reward_config,
            )

            for _ in range(self.n_eval_episodes):
                obs, _ = env.reset()
                done = False
                episode_return = 0.0

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    episode_return += reward

                returns.append(episode_return)

        returns = np.array(returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns) + 1e-8

        return {
            "sharpe": mean_return / std_return,
            "return_pct": mean_return,
            "max_drawdown_pct": 0.0,  # TODO: compute from episode info
            "win_rate": np.mean(returns > 0),
        }


def run_hpo(
    cfg: dict,
    datasets: list,
    n_trials: int = 50,
    study_name: str = "ppo_hpo",
    storage: str | None = None,
    timeout: int | None = None,
    n_jobs: int = 1,
    optimize_tcn: bool = True,
    optimize_reward: bool = True,
    timesteps_per_trial: int = 10000,
    results_dir: str = "/app/models/hpo",
) -> dict:
    """
    Run hyperparameter optimization.

    Args:
        cfg: Base configuration
        datasets: Training datasets
        n_trials: Number of optimization trials
        study_name: Optuna study name
        storage: Optuna storage URL (None for in-memory)
        timeout: Maximum optimization time in seconds
        n_jobs: Number of parallel jobs
        optimize_tcn: Include TCN parameters
        optimize_reward: Include reward parameters
        timesteps_per_trial: Training timesteps per trial
        results_dir: Directory to save results

    Returns:
        Dict with best parameters and study statistics
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("optuna is required for HPO. Install with: pip install optuna")

    # Create study
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1000)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    # Create objective
    objective = HPOObjective(
        base_cfg=cfg,
        datasets=datasets,
        timesteps=timesteps_per_trial,
        optimize_tcn=optimize_tcn,
        optimize_reward=optimize_reward,
    )

    # Run optimization
    logger.info(
        "Starting HPO: trials=%d timeout=%s tcn=%s reward=%s",
        n_trials,
        timeout,
        optimize_tcn,
        optimize_reward,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    # Save results
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    best_params = study.best_params
    results = {
        "best_params": best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "study_name": study_name,
    }

    with open(results_path / f"{study_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(
        "HPO complete: best_sharpe=%.4f trials=%d",
        study.best_value,
        len(study.trials),
    )
    logger.info("Best parameters: %s", best_params)

    return results


def apply_best_params(cfg: dict, best_params: dict) -> dict:
    """
    Apply best HPO parameters to configuration.

    Args:
        cfg: Base configuration
        best_params: Best parameters from HPO

    Returns:
        Updated configuration
    """
    import copy
    new_cfg = copy.deepcopy(cfg)
    learning_cfg = new_cfg.setdefault("learning", {})

    # Map HPO params to config
    param_mapping = {
        "learning_rate": ("learning", "learning_rate"),
        "n_steps": ("learning", "n_steps"),
        "batch_size": ("learning", "batch_size"),
        "gamma": ("learning", "gamma"),
        "gae_lambda": ("learning", "gae_lambda"),
        "ent_coef": ("learning", "ent_coef"),
        "clip_range": ("learning", "clip_range"),
        "tcn_features_dim": ("learning.tcn", "features_dim"),
        "tcn_num_layers": ("learning.tcn", "num_layers"),
        "tcn_kernel_size": ("learning.tcn", "kernel_size"),
        "tcn_dropout": ("learning.tcn", "dropout"),
        "nav_weight": ("learning.reward", "nav_weight"),
        "time_penalty_weight": ("learning.reward", "time_penalty_weight"),
        "profit_bonus_weight": ("learning.reward", "profit_bonus_weight"),
        "velocity_weight": ("learning.reward", "velocity_weight"),
        "time_normalizer": ("learning.reward", "time_normalizer"),
    }

    for param_name, value in best_params.items():
        if param_name in param_mapping:
            path, key = param_mapping[param_name]
            parts = path.split(".")
            target = new_cfg
            for part in parts:
                target = target.setdefault(part, {})
            target[key] = value

    return new_cfg
