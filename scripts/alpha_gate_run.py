#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""Credible alpha gate runner using the production TradingAgent backtest engine.

Unlike alpha_validation.py (screening only), this script uses the configured
TradingAgent, order queue, risk checks, exits, and SimBroker. A record is failed
closed unless the untouched calendar holdout has >=100 completed, non-crossing
trades and block-bootstrap lower bounds above zero for mean and trimmed expectancy.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.backtest.agent_engine import BacktestPlanResult, run_agent_backtest
from app.backtest.splits import block_bootstrap_ci
from app.risk.alpha_gate import (
    config_fingerprint, git_is_dirty, git_sha, implementation_hash, write_record,
)
from app.risk.trials import append_trial, count_trials, deflated_sharpe_probability
from app.utils.config import load_config


def _dataset_id(data_dir: Path, interval: str) -> tuple[str, list[str]]:
    files = sorted(data_dir.glob(f"*_{interval}.csv"))
    if not files:
        raise SystemExit(f"No *_{interval}.csv data in {data_dir}")
    h = hashlib.sha256()
    for path in files:
        h.update(path.name.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest(), [p.name for p in files]


def _common_calendar_cut(data_dir: Path, filenames: list[str], fraction: float) -> pd.Timestamp:
    indexes = []
    for name in filenames:
        frame = pd.read_csv(data_dir / name, usecols=[0], parse_dates=[0])
        index = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0], utc=True, errors="coerce"))
        indexes.append(index.dropna().drop_duplicates().sort_values())
    common = indexes[0]
    for index in indexes[1:]:
        common = common.intersection(index)
    if len(common) < 4:
        raise SystemExit("Need at least four timestamps common to every symbol")
    pos = min(max(int(len(common) * fraction), 1), len(common) - 1)
    return pd.Timestamp(common[pos])


def _trade_returns(
    details: list[dict], cutoff: pd.Timestamp
) -> tuple[list[float], list[int], timedelta, list[pd.Timestamp]]:
    """Aggregate fills into completed positions; purge trades crossing the cut."""
    positions: dict[tuple[str, str], dict] = {}
    training_durations = []
    cut = cutoff.tz_convert("UTC") if cutoff.tzinfo else cutoff.tz_localize("UTC")
    for rec in details:
        try:
            side = str(rec.get("side", "")).lower()
            if side not in {"buy", "sell"}:
                continue
            symbol = str(rec.get("symbol", ""))
            entry = pd.Timestamp(rec.get("entry_time"))
            exit_ = pd.Timestamp(rec.get("timestamp"))
            if entry.tzinfo is None:
                entry = entry.tz_localize("UTC")
            else:
                entry = entry.tz_convert("UTC")
            if exit_.tzinfo is None:
                exit_ = exit_.tz_localize("UTC")
            else:
                exit_ = exit_.tz_convert("UTC")
            if not symbol or entry.tzinfo is None or exit_.tzinfo is None:
                continue
            key = (symbol, entry.isoformat())
            group = positions.setdefault(key, {
                "entry": entry, "last_exit": exit_, "buy_qty": 0.0,
                "buy_notional": 0.0, "buy_fees": 0.0, "sell_qty": 0.0,
                "sell_notional": 0.0, "sell_fees": 0.0,
            })
            group["last_exit"] = max(group["last_exit"], exit_)
            qty = float(rec.get("qty", 0.0))
            notional = float(rec.get("notional", 0.0))
            commission = float(rec.get("commission", 0.0))
            if side == "buy":
                group["buy_qty"] += qty
                group["buy_notional"] += notional
                group["buy_fees"] += commission
            else:
                group["sell_qty"] += qty
                group["sell_notional"] += notional
                group["sell_fees"] += commission
        except (TypeError, ValueError, OverflowError):
            continue
    candidates = []
    for group in positions.values():
        if group["buy_qty"] <= 0 or group["sell_qty"] + 1e-9 < group["buy_qty"]:
            continue  # incomplete open position, including partial exits
        entry, exit_ = group["entry"], group["last_exit"]
        if exit_ <= entry:
            continue
        basis = group["buy_notional"] + group["buy_fees"]
        if basis <= 0:
            continue
        net_pct = (
            group["sell_notional"] - group["sell_fees"] - basis
        ) / basis * 100.0
        candidates.append((entry, exit_, net_pct,
                           max(int((exit_ - entry).total_seconds() // 60), 1)))
    for entry, exit_, _, duration in candidates:
        if entry < cut and exit_ < cut:
            training_durations.append(duration)
    embargo = timedelta(minutes=max(int(np.median(training_durations)), 1)) if training_durations else timedelta(0)
    holdout_start = cut + embargo
    # Drop trades spanning the cutoff and trades wholly inside the embargo.
    kept = [row for row in candidates if row[0] >= holdout_start and row[1] >= holdout_start]
    return ([r[2] for r in kept], [r[3] for r in kept], embargo, [r[1] for r in kept])


def _run(args) -> int:
    if args.record and git_is_dirty():
        raise SystemExit(
            "Refusing to write an alpha-gate record from a dirty worktree. "
            "Commit the exact implementation/config first, then rerun."
        )
    cfg = load_config(args.config)
    bt = cfg.setdefault("backtest", {})
    data_dir = Path(args.data_dir or bt.get("data_dir", "data")).resolve()
    interval = str(cfg.get("data", {}).get("interval", "1m"))
    dataset_id, files = _dataset_id(data_dir, interval)
    cutoff = _common_calendar_cut(data_dir, files, args.train_fraction)
    cfg["strategy"].setdefault("alpha_gate", {})["dataset_id"] = dataset_id

    # Keep the live config/fingerprint untouched. The research copy bypasses only
    # the entry gate so it can measure the candidate; it retains all production
    # strategy, risk, and order handling behavior.
    sim_cfg = copy.deepcopy(cfg)
    sim_cfg.setdefault("strategy", {}).setdefault("alpha_gate", {})["enabled"] = False
    sim_cfg["backtest"]["data_dir"] = str(data_dir)
    sim_cfg["backtest"]["symbols_source"] = "data_dir"
    sim_cfg["backtest"]["plan"] = {"enabled": False}
    sim_cfg["backtest"]["start"] = str(pd.Timestamp.min.date())
    last_time = max(pd.Timestamp(pd.read_csv(data_dir / f, usecols=[0], parse_dates=[0]).iloc[-1, 0])
                    for f in files)
    sim_cfg["backtest"]["end"] = last_time.strftime("%Y-%m-%d")

    print(f"engine=agent_engine")
    print(f"git_sha={git_sha()}")
    print(f"implementation_hash={implementation_hash()}")
    print(f"dataset_id={dataset_id} symbols={len(files)}")
    print(f"holdout_start={cutoff.isoformat()} (calendar intersection; train_fraction={args.train_fraction})")
    result = run_agent_backtest(sim_cfg)
    if isinstance(result, BacktestPlanResult):
        raise SystemExit("Unexpected plan result; disable backtest.plan")
    values, durations, embargo, exit_times = _trade_returns(result.trade_details, cutoff)
    n = len(values)
    gaps = np.diff(np.sort(pd.to_datetime(exit_times, utc=True).asi8)) if len(exit_times) > 1 else []
    median_gap_minutes = float(np.median(gaps)) / 60_000_000_000 if len(gaps) else 0.0
    median_hold_minutes = float(np.median(durations)) if durations else 1.0
    block = max(int(round(median_hold_minutes / max(median_gap_minutes, 1.0))), 1)
    ci_raw = block_bootstrap_ci(values, block, iterations=args.bootstrap_iterations,
                                seed=args.seed)
    ci = {k: (v if math.isfinite(v) else None) for k, v in ci_raw.items()}
    mean = float(np.mean(values)) if values else 0.0
    median = float(np.median(values)) if values else 0.0
    trimmed_values = np.sort(values)[:max(int(math.ceil(n * 0.99)), 1)] if n else []
    trimmed = float(np.mean(trimmed_values)) if n else 0.0
    passed = (n >= args.min_trades and mean > 0 and median > 0 and trimmed > 0
              and (ci["mean_low"] or 0) > 0 and (ci["trimmed_low"] or 0) > 0)
    trials_path = Path(args.trials_path)
    n_trials_before = count_trials(trials_path)
    sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
    sharpe = mean / sd if sd > 0 else 0.0
    dsr = deflated_sharpe_probability(sharpe, trials=n_trials_before + 1,
                                      observations=n)
    metrics = {
        "trades": n, "mean_net_pct": mean, "median_net_pct": median,
        "trimmed_net_pct": trimmed, "block_length_trades": block,
        "bootstrap_ci_95": ci, "trials_before": n_trials_before,
        "deflated_sharpe_probability": dsr,
    }
    provenance = {
        "engine": "agent_engine", "dataset_id": dataset_id,
        "data_dir": str(data_dir), "symbols": len(files),
        "calendar_cutoff": cutoff.isoformat(),
        "holdout_start": (cutoff + embargo).isoformat(),
        "embargo_minutes": int(embargo.total_seconds() // 60),
        "holdout_end": last_time.isoformat(), "costs": {
            k: sim_cfg["backtest"].get(k) for k in (
                "commission_pct", "maker_commission_pct", "slippage_bps", "spread_bps"
            )
        },
        "min_trades": args.min_trades, "bootstrap_iterations": args.bootstrap_iterations,
        "train_fraction": args.train_fraction,
    }
    print(json.dumps({"passed": passed, **metrics}, indent=2, allow_nan=False))
    trial_path = append_trial({
        "git_sha": git_sha(), "config_hash": config_fingerprint(cfg),
        "dataset_id": dataset_id, "params": cfg.get("strategy", {}),
        "implementation_hash": implementation_hash(),
        "working_tree_clean": not git_is_dirty(),
        "metrics": metrics,
    }, trials_path)
    print(f"trial_appended={trial_path} total_trials={n_trials_before + 1}")
    if args.record:
        path = write_record(cfg, passed=passed, metrics=metrics, provenance=provenance)
        print(f"record={path} passed={passed}")
    return 0 if passed else 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config/config.yaml"))
    parser.add_argument("--data-dir")
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--min-trades", type=int, default=100)
    parser.add_argument("--bootstrap-iterations", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--trials-path", default="data/reports/alpha_trials.jsonl")
    parser.add_argument("--record", action="store_true",
                        help="write a versioned alpha gate record (failed verdicts also record)")
    args = parser.parse_args()
    if args.min_trades < 100:
        parser.error("--min-trades cannot be below the safety floor of 100")
    if not 0 < args.train_fraction < 1:
        parser.error("--train-fraction must be between 0 and 1")
    try:
        raise SystemExit(_run(args))
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"alpha gate run failed: {exc}") from exc


if __name__ == "__main__":
    main()
