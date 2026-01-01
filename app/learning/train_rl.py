from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from app.learning.data import load_csv_data
from app.learning.drift import compute_feature_stats
from app.learning.env import TradingEnv
from app.learning.evaluate import evaluate_model
from app.learning.registry import register_model, set_active_model


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
    time_penalty = float(
        training_cfg.get("reward_time_penalty_per_step", learning_cfg.get("reward_time_penalty_per_step", 0.0))
    )

    if resume is None:
        resume = bool(training_cfg.get("resume", True))

    datasets = load_csv_data(data_dir, interval=interval)
    envs = []
    eval_sets = []
    train_sets = []
    for df in datasets:
        split_idx = int(len(df) * (1.0 - eval_split))
        train_df = df.iloc[:split_idx] if split_idx > 0 else df
        eval_df = df.iloc[split_idx:] if split_idx > 0 else df
        envs.append(
            lambda data=train_df: TradingEnv(
                data=data,
                window_size=window_size,
                initial_cash=training_cfg.get("initial_cash", cfg["backtest"]["initial_cash"]),
                commission_pct=training_cfg.get("commission_pct", cfg["backtest"]["commission_pct"]),
                slippage_bps=training_cfg.get("slippage_bps", cfg["backtest"]["slippage_bps"]),
                time_penalty_per_step=time_penalty,
                feature_config=feature_config,
            )
        )
        eval_sets.append(eval_df)
        train_sets.append(train_df)
    vec_env = DummyVecEnv(envs)

    if resume and Path(model_path).exists():
        model = PPO.load(model_path, env=vec_env, device=device)
    else:
        model = PPO("MlpPolicy", vec_env, verbose=1, device=device)
    logging.info("Starting RL training for %d timesteps", timesteps)
    model.learn(total_timesteps=timesteps)

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    logging.info("Saved model to %s", output_path)

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
    if is_best:
        best_model_file = Path(best_model_path)
        best_model_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, best_model_file)
        best_report_file = Path(best_report_path)
        best_report_file.parent.mkdir(parents=True, exist_ok=True)
        best_report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logging.info("Updated best model: %s", best_model_file)
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
            model_path=model_path,
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
    return str(output_path)


def _resolve_device(device: str) -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if device != "auto":
        return device
    return "cpu"


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
