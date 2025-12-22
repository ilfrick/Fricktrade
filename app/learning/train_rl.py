from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from app.learning.data import load_csv_data
from app.learning.env import TradingEnv
from app.learning.evaluate import evaluate_model


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

    if resume is None:
        resume = bool(training_cfg.get("resume", True))

    datasets = load_csv_data(data_dir, interval=interval)
    envs = []
    eval_sets = []
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
                feature_config=feature_config,
            )
        )
        eval_sets.append(eval_df)
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
    if _is_better_report(report, best_report_path):
        best_model_file = Path(best_model_path)
        best_model_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, best_model_file)
        best_report_file = Path(best_report_path)
        best_report_file.parent.mkdir(parents=True, exist_ok=True)
        best_report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logging.info("Updated best model: %s", best_model_file)
    return str(output_path)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


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
