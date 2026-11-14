# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""Append-only experiment registry and trial-count-aware Sharpe diagnostics."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_TRIALS_PATH = Path("data/reports/alpha_trials.jsonl")


def append_trial(record: dict[str, Any], path: Path | str = DEFAULT_TRIALS_PATH) -> Path:
    """Append exactly one durable JSON line; never replace prior experiment history."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "git_sha": record.get("git_sha", "unknown"),
        "config_hash": record.get("config_hash", ""),
        "dataset_id": record.get("dataset_id", ""),
        "implementation_hash": record.get("implementation_hash", ""),
        "working_tree_clean": bool(record.get("working_tree_clean", False)),
        "params": record.get("params", {}),
        "metrics": record.get("metrics", {}),
    }
    fd = os.open(target, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        os.write(fd, (json.dumps(row, sort_keys=True, default=str) + "\n").encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    return target


def count_trials(path: Path | str = DEFAULT_TRIALS_PATH) -> int:
    target = Path(path)
    if not target.exists():
        return 0
    with target.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def deflated_sharpe_probability(
    sharpe: float, *, trials: int, observations: int, skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Approximate probability the observed Sharpe exceeds the multiple-test hurdle.

    Uses the expected maximum of N standard-normal trials (Euler approximation) as
    the null hurdle, then a finite-sample Sharpe standard error with skew/kurtosis.
    This is deliberately a diagnostic; the promotion gate uses bootstrap expectancy
    CIs rather than this normal approximation.
    """
    if observations < 2 or trials < 1 or not math.isfinite(sharpe):
        return 0.0
    if trials == 1:
        hurdle = 0.0
    else:
        euler = 0.5772156649015329
        z = (1.0 - euler) * _normal_ppf(1.0 - 1.0 / trials) + euler * _normal_ppf(
            1.0 - 1.0 / (trials * math.e)
        )
        hurdle = z / math.sqrt(observations)
    sr = sharpe / math.sqrt(observations)
    variance = max(
        (1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr * sr)
        / max(observations - 1, 1),
        1e-15,
    )
    return float(0.5 * math.erfc((hurdle - sr) / math.sqrt(2.0 * variance)))


def _normal_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam approximation; dependency-free)."""
    p = min(max(p, 1e-15), 1.0 - 1e-15)
    a = [-39.6968302866538, 220.946098424521, -275.928510446969,
         138.357751867269, -30.6647980661472, 2.50662827745924]
    b = [-54.4760987982241, 161.585836858041, -155.698979859887,
         66.8013118877197, -13.2806815528857]
    c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184,
         -2.54973253934373, 4.37466414146497, 2.93816398269878]
    d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742]
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - 0.02425:
        q = math.sqrt(-2 * math.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           ((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1
