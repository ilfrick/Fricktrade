# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import tensorflow as tf # Added tensorflow import
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from app.learning.data import load_csv_data
from app.learning.drift import compute_feature_stats
from app.learning.env import TradingEnv
from app.learning.live_rewards import load_live_reward_overrides
from app.learning.evaluate import evaluate_model
from app.learning.registry import build_active_record, register_model, set_active_model
from app.utils.gpu_state import is_gpu_disabled, disable_gpu_until_restart

logger = logging.getLogger(__name__)


def train_from_config(cfg: dict, resume: bool | None = None) -> str:
    learning_cfg = cfg.get("learning", {})
    training_cfg = learning_cfg.get("training", {})
    model_path = learning_cfg.get("model_path", "/app/models/ppo_policy.zip")
    best_model_path = learning_cfg.get("best_model_path", "/app/models/ppo_policy_best.zip")
    device = _resolve_device(learning_cfg.get("device", "auto"))
    feature_config = learning_cfg.get("features", {})

    data_dir = training_cfg.get("data_dir", cfg["backtest"]["data_dir"])
    interval = training_cfg.get("interval", cfg["data"].get("interval"))
    window_size = learning_cfg.get("window_size", 50)
    timesteps = int(training_cfg.get("timesteps", 200_000))
    eval_split = float(training_cfg.get("eval_split", 0.2))
    checkpoint_best_only = bool(training_cfg.get("checkpoint_best_only", True))
    time_penalty = float(
        training_cfg.get("reward_time_penalty_per_step", learning_cfg.get("reward_time_penalty_per_step", 0.0))
    )

    if resume is None:
        resume = bool(training_cfg.get("resume", True))

    datasets = load_csv_data(data_dir, interval=interval)
    live_overrides = {}
    if training_cfg.get("use_live_rewards") and learning_cfg.get("live_rewards", {}).get("enabled", False):
        try:
            live_overrides = load_live_reward_overrides(cfg, float(learning_cfg.get("bar_interval_minutes", 5.0)))
        except Exception as exc:
            logging.warning("Failed to load live reward overrides: %s", exc)
    envs = []
    eval_sets = []
    train_sets = []
    for df in datasets:
        split_idx = int(len(df) * (1.0 - eval_split))
        if split_idx <= 0 or split_idx >= len(df):
            train_df = df
            eval_df = df.iloc[:0]  # Empty DataFrame preserving columns
        else:
            train_df = df.iloc[:split_idx]
            eval_df = df.iloc[split_idx:]
        symbol = train_df.attrs.get("symbol")
        overrides = live_overrides.get(str(symbol), {}) if symbol and live_overrides else {}
        envs.append(
            lambda data=train_df, sym=symbol, ov=overrides: TradingEnv(
                data=data,
                window_size=window_size,
                initial_cash=training_cfg.get("initial_cash", cfg["backtest"]["initial_cash"]),
                commission_pct=training_cfg.get("commission_pct", cfg["backtest"]["commission_pct"]),
                slippage_bps=training_cfg.get("slippage_bps", cfg["backtest"]["slippage_bps"]),
                time_penalty_per_step=time_penalty,
                feature_config=feature_config,
                # New reward shaping parameters
                win_trade_bonus=float(learning_cfg.get("reward_win_trade_bonus", 0.5)),
                loss_trade_penalty=float(learning_cfg.get("reward_loss_trade_penalty", 0.2)),
                win_streak_bonus_scale=float(learning_cfg.get("reward_win_streak_bonus_scale", 0.1)),
                loss_streak_penalty_scale=float(learning_cfg.get("reward_loss_streak_penalty_scale", 0.15)),
                max_streak_bonus=float(learning_cfg.get("reward_max_streak_bonus", 1.0)),
                max_streak_penalty=float(learning_cfg.get("reward_max_streak_penalty", 2.0)),
                sharpe_bonus_scale=float(learning_cfg.get("reward_sharpe_bonus_scale", 0.1)),
                sharpe_window_size=int(learning_cfg.get("reward_sharpe_window_size", 100)),
                target_trade_frequency=float(learning_cfg.get("reward_target_trade_frequency", 0.1)),
                frequency_penalty_scale=float(learning_cfg.get("reward_frequency_penalty_scale", 0.5)),
                enable_time_aware_penalty=bool(learning_cfg.get("enable_time_aware_penalty", False)),
                base_time_penalty_per_minute=float(learning_cfg.get("base_time_penalty_per_minute", 0.01)),
                bar_interval_minutes=float(learning_cfg.get("bar_interval_minutes", 5.0)),
                reward_pnl_mode=str(learning_cfg.get("reward_pnl_mode", "abs")),
                reward_pnl_scale=float(learning_cfg.get("reward_pnl_scale", 1.0)),
                symbol=str(sym) if sym else None,
                reward_overrides=ov,
                reward_override_mode=str(learning_cfg.get("live_rewards", {}).get("mode", "add")),
            )
        )
        eval_sets.append(eval_df)
        train_sets.append(train_df)
    vec_env = DummyVecEnv(envs)

    model = None
    # Try with initial device, then fall back to CPU if GPU error occurs
    for attempt in range(2):
        current_device = device if attempt == 0 else "cpu"
        if is_gpu_disabled(): # If already globally disabled, just use CPU
            current_device = "cpu"

        try:
            if resume and Path(model_path).exists():
                try:
                    model = PPO.load(model_path, env=vec_env, device=current_device, custom_objects=_sb3_custom_objects())
                except ValueError as exc:
                    logging.warning("RL model shape mismatch; rebuilding model: %s", exc)
                    model = None
            if model is None:
                model = PPO("MlpPolicy", vec_env, verbose=1, device=current_device)
            
            # If successful, break out of retry loop
            break

        except (torch.cuda.OutOfMemoryError, tf.errors.ResourceExhaustedError) as exc:
            if current_device != "cpu":
                logging.warning("CUDA out of memory during RL training: %s. Falling back to CPU for current and future runs.", exc)
                disable_gpu_until_restart()
                device = "cpu" # Update device for subsequent PPO.learn call
                continue # Retry with CPU
            else:
                logging.error("RL training failed on CPU after GPU error: %s", exc)
                raise # Re-raise if fails even on CPU

        except Exception as exc:
            logging.error("Unknown error during RL training setup: %s", exc)
            raise

    if model is None:
        logging.error("Failed to initialize PPO model for training after retries.")
        raise RuntimeError("Failed to initialize PPO model.")

    checkpoint_interval = int(training_cfg.get("checkpoint_interval_steps", 0))
    publish_in_progress = bool(training_cfg.get("publish_in_progress", False))
    if checkpoint_best_only and publish_in_progress:
        logging.info("Checkpoint best-only enabled; disabling in-progress publishing.")
        publish_in_progress = False
    callback = None
    if checkpoint_interval > 0:
        registry_cfg = learning_cfg.get("registry", {}) or {}
        active_path = str(registry_cfg.get("active_path", "/app/models/model_active.json"))
        callback = _InProgressCheckpointCallback(
            model_path=model_path,
            active_path=active_path,
            save_freq=checkpoint_interval,
            publish_active=publish_in_progress,
        )
    logging.info("Starting RL training for %d timesteps", timesteps)
    model.learn(total_timesteps=timesteps, callback=callback)

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path = model_path
    if model_path.endswith(".zip"):
        candidate_path = model_path[:-4] + ".candidate.zip"
    else:
        candidate_path = model_path + ".candidate"
    model.save(candidate_path)
    candidate_file = Path(candidate_path)
    if not candidate_file.exists() and not str(candidate_file).endswith(".zip"):
        candidate_file = Path(f"{candidate_path}.zip")
    if not candidate_file.exists():
        raise FileNotFoundError(f"Saved RL model not found at {candidate_file}")
    logging.info("Saved candidate model to %s", candidate_file)

    report_path = training_cfg.get("report_path", "/app/models/training_report.json")
    report_plot_dir = training_cfg.get("report_plot_dir", "/app/models/reports")
    report = evaluate_model(
        model=model,
        datasets=eval_sets,
        window_size=window_size,
        training_cfg=training_cfg,
        report_path=report_path,
        feature_config=feature_config,
        plot_dir=report_plot_dir,
    )
    best_report_path = training_cfg.get("best_report_path", "/app/models/training_report_best.json")
    is_best = _is_better_report(report, best_report_path)
    model_path_for_registry = str(candidate_file)
    if is_best:
        best_model_file = Path(best_model_path)
        best_model_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_file, best_model_file)
        best_report_file = Path(best_report_path)
        best_report_file.parent.mkdir(parents=True, exist_ok=True)
        best_report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        shutil.copy2(candidate_file, output_path)
        model_path_for_registry = str(output_path)
        logging.info("Updated best model: %s", best_model_file)
    elif checkpoint_best_only and Path(best_model_path).exists():
        shutil.copy2(best_model_path, output_path)
        model_path_for_registry = str(best_model_path)
        logging.info("Candidate did not beat best; keeping best model at %s", output_path)
    else:
        shutil.copy2(candidate_file, output_path)
        model_path_for_registry = str(output_path)
        logging.info("Candidate did not beat best; keeping latest at %s", output_path)
    registry_cfg = learning_cfg.get("registry", {}) or {}
    if registry_cfg.get("enabled", True):
        drift_cfg = learning_cfg.get("drift", {}) or {}
        stats = {}
        if drift_cfg.get("baseline_enabled", True):
            stats = compute_feature_stats(
                train_sets,
                window_size=window_size,
                feature_config=feature_config,
                max_samples=int(drift_cfg.get("baseline_max_samples", 5000)),
                stride=int(drift_cfg.get("baseline_stride", 5)),
            )
        metadata = {
            "window_size": window_size,
            "timesteps": timesteps,
            "eval_split": eval_split,
            "interval": interval,
            "device": device,
            "feature_config": feature_config,
        }
        record = register_model(
            model_path=model_path_for_registry,
            best_model_path=best_model_path,
            report=report,
            registry_path=str(registry_cfg.get("path", "/app/models/model_registry.json")),
            feature_stats=stats,
            metadata=metadata,
            artifact_dir=registry_cfg.get("artifact_dir"),
            artifact_prefix=str(registry_cfg.get("artifact_prefix", "ppo_policy")),
        )
        if registry_cfg.get("use_active", True):
            publish_mode = str(registry_cfg.get("publish_mode", "best")).lower()
            active_path = registry_cfg.get("active_path", "/app/models/model_active.json")
            if publish_mode == "latest" or (publish_mode == "best" and is_best):
                set_active_model(active_path, record, reason=publish_mode)
    try:
        candidate_file.unlink(missing_ok=True)
    except Exception:
        logging.debug("Failed to remove candidate model at %s", candidate_file)
    return str(output_path)


class _InProgressCheckpointCallback(BaseCallback):
    def __init__(self, model_path: str, active_path: str, save_freq: int, publish_active: bool):
        super().__init__()
        self._model_path = model_path
        self._active_path = active_path
        self._save_freq = max(int(save_freq), 1)
        self._publish_active = publish_active

    def _on_step(self) -> bool:
        if self.num_timesteps and self.num_timesteps % self._save_freq == 0:
            self._save_checkpoint()
        return True

    def _save_checkpoint(self) -> None:
        tmp_path = f"{self._model_path}.tmp"
        self.model.save(tmp_path)
        tmp_file = tmp_path if tmp_path.endswith(".zip") else f"{tmp_path}.zip"
        if os.path.exists(tmp_file):
            os.replace(tmp_file, self._model_path)
        elif os.path.exists(tmp_path):
            os.replace(tmp_path, self._model_path)
        else:
            logger.warning("Checkpoint temp file missing; skip replace: tmp=%s model=%s", tmp_file, self._model_path)
            return
        if self._publish_active:
            record = build_active_record(self._model_path)
            set_active_model(self._active_path, record, reason="checkpoint")


def _resolve_device(device: str) -> str:
    if is_gpu_disabled():
        logging.warning("GPU globally disabled. Forcing CPU for RL training.")
        return "cpu"
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available() and device != "cpu":
        return "cuda"
    return "cpu"


def _sb3_custom_objects() -> dict:
    return {
        "clip_range": lambda _: 0.2,
        "lr_schedule": lambda _: 0.0,
    }


def _is_better_report(report: dict, best_report_path: str) -> bool:
    avg = report.get("average", {})
    score = (float(avg.get("sharpe", 0.0)), float(avg.get("return_pct", 0.0)))
    best_file = Path(best_report_path)
    if not best_file.exists():
        return True
    try:
        best = json.loads(best_file.read_text(encoding="utf-8"))
    except Exception:
        return True
    best_avg = best.get("average", {})
    best_score = (float(best_avg.get("sharpe", 0.0)), float(best_avg.get("return_pct", 0.0)))
    return score > best_score
