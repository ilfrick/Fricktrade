#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module(path: Path):
    module_name = "top_movers_same_day"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _args():
    p = argparse.ArgumentParser(description="Tune XGBoost hyperparameters for same-day top movers + entry models.")
    p.add_argument("--repo-root", default="/home/nicola/Fricktrade")
    p.add_argument("--data-dir", default="/home/nicola/Fricktrade/data")
    p.add_argument("--output-dir", default="/home/nicola/Fricktrade/data/reports/modeling/same_day_xgb_tuning")
    p.add_argument("--final-output-dir", default="/home/nicola/Fricktrade/data/reports/modeling/same_day_xgb_tuned")
    p.add_argument("--trials", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--cutoff-bars", type=int, default=30)
    p.add_argument("--entry-k", type=int, default=10)
    p.add_argument("--forecast-date", default="")
    p.add_argument("--python-bin", default="/home/nicola/Fricktrade/testenv/bin/python")
    return p.parse_args()


def _safe_mean(series: pd.Series) -> float | None:
    if series is None or len(series) == 0:
        return None
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.mean())


def main() -> None:
    args = _args()
    repo_root = Path(args.repo_root)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    final_output_dir = Path(args.final_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    module_path = repo_root / "scripts" / "top_movers_same_day.py"
    mod = _load_module(module_path)

    forecast_date = date.fromisoformat(args.forecast_date) if args.forecast_date else None

    base_cfg = mod.Config(
        data_dir=data_dir,
        output_dir=output_dir,
        model_type="xgboost",
        top_n=max(int(args.top_n), 1),
        min_bars=60,
        cutoff_bars=max(int(args.cutoff_bars), 5),
        random_state=42,
        forecast_date=forecast_date,
        entry_horizon_bars=90,
        entry_warmup_bars=20,
        low_zone_tol_pct=0.35,
        rebound_target_pct=2.0,
        entry_k=max(int(args.entry_k), 1),
        xgb_n_estimators_nowcast=450,
        xgb_n_estimators_entry=320,
        xgb_max_depth_nowcast=6,
        xgb_max_depth_entry=5,
        xgb_learning_rate=0.05,
        xgb_subsample=0.9,
        xgb_colsample_bytree=0.9,
        xgb_min_child_weight=1.0,
        xgb_gamma=0.0,
        xgb_reg_lambda=1.0,
    )

    symbol_days = mod._load_symbol_days(base_cfg)
    if not symbol_days:
        raise SystemExit("No valid symbol-day CSV files found for tuning.")

    nowcast_df = mod._build_nowcast_dataset(symbol_days, base_cfg)
    entry_df = mod._build_entry_dataset(symbol_days, base_cfg)
    if nowcast_df.empty or entry_df.empty:
        raise SystemExit("Nowcast/entry dataset empty; cannot tune.")

    nowcast_feats = mod._feature_columns(nowcast_df, "target_top_n")
    entry_feats = mod._feature_columns(entry_df, "target_entry")

    rng = np.random.default_rng(args.seed)
    trials = max(int(args.trials), 1)
    trial_rows: list[dict] = []

    # Always include a baseline first trial.
    trial_params = [
        {
            "xgb_n_estimators_nowcast": 450,
            "xgb_n_estimators_entry": 320,
            "xgb_max_depth_nowcast": 6,
            "xgb_max_depth_entry": 5,
            "xgb_learning_rate": 0.05,
            "xgb_subsample": 0.9,
            "xgb_colsample_bytree": 0.9,
            "xgb_min_child_weight": 1.0,
            "xgb_gamma": 0.0,
            "xgb_reg_lambda": 1.0,
        }
    ]

    while len(trial_params) < trials:
        trial_params.append(
            {
                "xgb_n_estimators_nowcast": int(rng.choice([260, 320, 380, 450, 520, 640])),
                "xgb_n_estimators_entry": int(rng.choice([180, 240, 320, 420, 520])),
                "xgb_max_depth_nowcast": int(rng.choice([3, 4, 5, 6, 7, 8])),
                "xgb_max_depth_entry": int(rng.choice([2, 3, 4, 5, 6])),
                "xgb_learning_rate": float(rng.choice([0.02, 0.03, 0.05, 0.08, 0.12])),
                "xgb_subsample": float(rng.choice([0.7, 0.8, 0.9, 1.0])),
                "xgb_colsample_bytree": float(rng.choice([0.7, 0.8, 0.9, 1.0])),
                "xgb_min_child_weight": float(rng.choice([0.5, 1.0, 2.0, 4.0, 8.0])),
                "xgb_gamma": float(rng.choice([0.0, 0.05, 0.1, 0.2])),
                "xgb_reg_lambda": float(rng.choice([0.5, 1.0, 2.0, 4.0])),
            }
        )

    for idx, params in enumerate(trial_params, start=1):
        cfg = mod.Config(**{**asdict(base_cfg), **params})
        wf_nowcast = mod._walkforward_nowcast(nowcast_df, nowcast_feats, cfg)
        wf_entry = mod._walkforward_entry(entry_df, entry_feats, cfg)

        n_prec = _safe_mean(wf_nowcast["precision_at_k"]) if not wf_nowcast.empty else None
        n_base = _safe_mean(wf_nowcast["baseline_positive_rate"]) if not wf_nowcast.empty else None
        e_prec = _safe_mean(wf_entry["precision_at_k"]) if not wf_entry.empty else None
        e_base = _safe_mean(wf_entry["baseline_positive_rate"]) if not wf_entry.empty else None
        e_rec = _safe_mean(wf_entry["recall_at_k"]) if not wf_entry.empty else None

        n_lift = (n_prec - n_base) if (n_prec is not None and n_base is not None) else None
        e_lift = (e_prec - e_base) if (e_prec is not None and e_base is not None) else None

        # Objective: prioritize same-day top-mover nowcast, then entry quality.
        objective = -1e9
        if n_lift is not None and e_lift is not None:
            objective = (1.0 * n_lift) + (0.55 * e_lift) + (0.08 * (e_rec or 0.0))

        trial_rows.append(
            {
                "trial": idx,
                **params,
                "nowcast_rows": int(len(wf_nowcast)),
                "entry_rows": int(len(wf_entry)),
                "nowcast_precision_mean": n_prec,
                "nowcast_baseline_mean": n_base,
                "nowcast_lift": n_lift,
                "entry_precision_mean": e_prec,
                "entry_baseline_mean": e_base,
                "entry_lift": e_lift,
                "entry_recall_mean": e_rec,
                "objective": objective,
            }
        )

    trials_df = pd.DataFrame(trial_rows).sort_values("objective", ascending=False).reset_index(drop=True)
    best = trials_df.iloc[0].to_dict()

    trials_csv = output_dir / "xgb_tuning_trials.csv"
    best_json = output_dir / "xgb_tuning_best.json"
    trials_df.to_csv(trials_csv, index=False)
    best_json.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "trials": int(trials),
                "best": best,
                "artifacts": {"trials_csv": str(trials_csv)},
            },
            indent=2,
        )
    )

    # Produce final full artifact set with best params via the main same-day script.
    cmd = [
        str(args.python_bin),
        str(module_path),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(final_output_dir),
        "--model",
        "xgboost",
        "--top-n",
        str(base_cfg.top_n),
        "--cutoff-bars",
        str(base_cfg.cutoff_bars),
        "--entry-k",
        str(base_cfg.entry_k),
        "--xgb-n-estimators-nowcast",
        str(int(best["xgb_n_estimators_nowcast"])),
        "--xgb-n-estimators-entry",
        str(int(best["xgb_n_estimators_entry"])),
        "--xgb-max-depth-nowcast",
        str(int(best["xgb_max_depth_nowcast"])),
        "--xgb-max-depth-entry",
        str(int(best["xgb_max_depth_entry"])),
        "--xgb-learning-rate",
        str(float(best["xgb_learning_rate"])),
        "--xgb-subsample",
        str(float(best["xgb_subsample"])),
        "--xgb-colsample-bytree",
        str(float(best["xgb_colsample_bytree"])),
        "--xgb-min-child-weight",
        str(float(best["xgb_min_child_weight"])),
        "--xgb-gamma",
        str(float(best["xgb_gamma"])),
        "--xgb-reg-lambda",
        str(float(best["xgb_reg_lambda"])),
    ]
    if forecast_date:
        cmd.extend(["--forecast-date", forecast_date.isoformat()])
    subprocess.run(cmd, check=True)

    print(f"Tuning trials: {trials_csv}")
    print(f"Best params: {best_json}")
    print(f"Final output dir: {final_output_dir}")


if __name__ == "__main__":
    main()
