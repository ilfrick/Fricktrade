# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from app.learning.env import TradingEnv
from app.learning.data import load_csv_data


def _ensure_matplotlib_backend() -> None:
    return


def _compute_metrics(values: list[float]) -> dict:
    if not values:
        return {"return_pct": 0.0, "sharpe": 0.0, "max_drawdown_pct": 0.0}
    start = values[0]
    end = values[-1]
    returns = np.diff(values)
    ret_pct = (end - start) / start * 100.0 if start else 0.0
    if returns.size == 0:
        sharpe = 0.0
    else:
        mean = float(np.mean(returns))
        std = float(np.std(returns)) or 1e-9
        sharpe = mean / std * np.sqrt(252)
    peak = values[0]
    max_dd = 0.0
    for val in values:
        peak = max(peak, val)
        drawdown = (peak - val) / peak if peak else 0.0
        max_dd = max(max_dd, drawdown)
    return {
        "return_pct": ret_pct,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100.0,
    }


def evaluate_model(
    model: PPO,
    datasets: list[pd.DataFrame],
    window_size: int,
    training_cfg: dict,
    report_path: str,
    feature_config: dict | None = None,
    plot_dir: str | None = None,
) -> dict:
    results = []
    equity_curves = []
    for df in datasets:
        time_penalty = float(training_cfg.get("reward_time_penalty_per_step", 0.0))
        env = DummyVecEnv(
            [
                lambda data=df: TradingEnv(
                    data=data,
                    window_size=window_size,
                    initial_cash=training_cfg.get("initial_cash", 100000),
                    commission_pct=training_cfg.get("commission_pct", 0.05),
                    slippage_bps=training_cfg.get("slippage_bps", 2),
                    time_penalty_per_step=time_penalty,
                    feature_config=feature_config,
                )
            ]
        )
        obs = env.reset()
        done = False
        values = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = env.step(action)
            done = bool(dones[0]) if isinstance(dones, (list, np.ndarray)) else bool(dones)
            last_value = env.get_attr("last_value")[0]
            values.append(float(last_value))
        results.append(_compute_metrics(values))
        equity_curves.append(values)

    if results:
        avg_return = float(np.mean([r["return_pct"] for r in results]))
        avg_sharpe = float(np.mean([r["sharpe"] for r in results]))
        avg_dd = float(np.mean([r["max_drawdown_pct"] for r in results]))
    else:
        avg_return = avg_sharpe = avg_dd = 0.0

    report = {
        "datasets": results,
        "average": {
            "return_pct": avg_return,
            "sharpe": avg_sharpe,
            "max_drawdown_pct": avg_dd,
        },
    }
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logging.info("Saved training report to %s", report_file)
    if plot_dir:
        _ensure_matplotlib_backend()
        plot_path = Path(plot_dir)
        plot_path.mkdir(parents=True, exist_ok=True)
        for idx, values in enumerate(equity_curves):
            if not values:
                continue
            curve_file = plot_path / f"equity_curve_{idx}.csv"
            pd.DataFrame({"portfolio_value": values}).to_csv(curve_file, index=False)
            fig = plt.figure(figsize=(10, 4))
            plt.plot(values, linewidth=1.5)
            plt.title(f"Equity Curve {idx}")
            plt.xlabel("Step")
            plt.ylabel("Portfolio Value")
            fig.tight_layout()
            png_file = plot_path / f"equity_curve_{idx}.png"
            plt.savefig(png_file, dpi=120)
            plt.close(fig)
        logging.info("Saved evaluation charts to %s", plot_path)
    return report


def evaluate_from_config(cfg: dict) -> dict:
    learning_cfg = cfg.get("learning", {})
    training_cfg = learning_cfg.get("training", {})
    model_path = _select_model_path(learning_cfg)
    device = _resolve_device(learning_cfg.get("device", "auto"))
    feature_config = learning_cfg.get("features", {})

    data_dir = training_cfg.get("data_dir", cfg["backtest"]["data_dir"])
    interval = training_cfg.get("interval", cfg["data"].get("interval"))
    window_size = learning_cfg.get("window_size", 50)
    eval_split = float(training_cfg.get("eval_split", 0.2))
    report_path = training_cfg.get("report_path", "/app/models/training_report.json")
    plot_dir = training_cfg.get("report_plot_dir", "/app/models/reports")
    if "reward_time_penalty_per_step" not in training_cfg:
        training_cfg = dict(training_cfg)
        training_cfg["reward_time_penalty_per_step"] = float(
            learning_cfg.get("reward_time_penalty_per_step", 0.0)
        )

    datasets = load_csv_data(data_dir, interval=interval)
    eval_sets = []
    for df in datasets:
        split_idx = int(len(df) * (1.0 - eval_split))
        eval_df = df.iloc[split_idx:] if split_idx > 0 else df
        eval_sets.append(eval_df)

    model = PPO.load(model_path, device=device, custom_objects=_sb3_custom_objects())
    return evaluate_model(
        model=model,
        datasets=eval_sets,
        window_size=window_size,
        training_cfg=training_cfg,
        report_path=report_path,
        feature_config=feature_config,
        plot_dir=plot_dir,
    )


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


def _sb3_custom_objects() -> dict:
    return {
        "clip_range": lambda _: 0.2,
        "lr_schedule": lambda _: 0.0,
    }


def _select_model_path(learning_cfg: dict) -> str:
    model_path = learning_cfg.get("model_path", "/app/models/ppo_policy.zip")
    if not learning_cfg.get("use_best_model", True):
        return model_path
    best_path = learning_cfg.get("best_model_path", "/app/models/ppo_policy_best.zip")
    return best_path if Path(best_path).exists() else model_path
