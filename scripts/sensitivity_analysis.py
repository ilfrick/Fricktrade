# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""Sensitivity analysis for MR strategy parameters and ATR stop multiplier.

Tests each parameter independently at multiple values while holding others fixed.
Runs two test periods (crash + recovery) per variation.

Usage:
    python3 -m scripts.sensitivity_analysis [--param PARAM_NAME]

Without --param, runs all parameters sequentially.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from app.backtest.agent_engine import run_agent_backtest
from app.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = Path("/data/reports/sensitivity_results.json")

# Two test periods: crash (Jan 28 - Feb 9) and recovery (Feb 21 - Mar 3)
TEST_PERIODS = [
    ("crash", "2026-01-28", "2026-02-09"),
    ("recovery", "2026-02-21", "2026-03-03"),
]

# Parameter grid: name → (config_path, values, baseline)
# config_path uses dots for nested keys
PARAM_GRID = {
    "bb_period": {
        "path": "strategy.params.crypto_mean_reversion.bb_period",
        "values": [50, 75, 100, 150, 200],
        "baseline": 100,
    },
    "bb_period_low": {
        "path": "strategy.params.crypto_mean_reversion.bb_period",
        "values": [20, 25, 30, 35, 40, 45],
        "baseline": 50,
    },
    "bb_std": {
        "path": "strategy.params.crypto_mean_reversion.bb_std",
        "values": [1.5, 1.75, 2.0, 2.5, 3.0],
        "baseline": 2.0,
    },
    "rsi_period": {
        "path": "strategy.params.crypto_mean_reversion.rsi_period",
        "values": [30, 50, 70, 100, 140],
        "baseline": 70,
    },
    "rsi_oversold": {
        "path": "strategy.params.crypto_mean_reversion.rsi_oversold",
        "values": [20.0, 25.0, 30.0, 35.0, 40.0],
        "baseline": 30.0,
    },
    "drop_window_bars": {
        "path": "strategy.params.crypto_mean_reversion.drop_window_bars",
        "values": [30, 50, 75, 100, 150],
        "baseline": 75,
    },
    "atr_stop_mult": {
        "path": "risk.crypto.atr_stop_mult",
        "values": [1.5, 2.0, 2.5, 3.5, 5.0],
        "baseline": 2.5,
    },
}


def _set_nested(cfg: dict, dotted_key: str, value) -> None:
    """Set a nested config value using dot-separated key."""
    keys = dotted_key.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _run_single(cfg: dict, start: str, end: str) -> dict:
    """Run a single backtest and return summary metrics."""
    test_cfg = copy.deepcopy(cfg)
    test_cfg["backtest"]["start"] = start
    test_cfg["backtest"]["end"] = end
    # Disable walk-forward for speed
    test_cfg["backtest"].pop("walk_forward", None)

    try:
        result = run_agent_backtest(test_cfg)
        ret = float(result.return_pct) if hasattr(result, "return_pct") else float(result.average_return_pct)
        trades = int(result.trades) if hasattr(result, "trades") else int(result.total_trades)
        trade_details = getattr(result, "trade_details", [])

        sells = [t for t in trade_details if t.get("side") == "sell" and "pnl_pct" in t]
        wins = [t for t in sells if t["pnl_pct"] > 0.01]
        losses = [t for t in sells if t["pnl_pct"] < -0.01]
        win_rate = len(wins) / len(sells) * 100 if sells else 0
        avg_win = sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0

        # Compute max drawdown from trade sequence
        equity = 10000.0
        peak = equity
        max_dd = 0.0
        for t in trade_details:
            if t.get("side") == "sell":
                pnl = t.get("pnl_pct", 0) / 100.0 * t.get("notional", 0)
                equity += pnl
                peak = max(peak, equity)
                dd = (peak - equity) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)

        return {
            "return_pct": round(ret, 4),
            "trades": trades,
            "sells": len(sells),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 1),
            "avg_win": round(avg_win, 3),
            "avg_loss": round(avg_loss, 3),
            "max_dd_pct": round(max_dd, 2),
        }
    except Exception as exc:
        logger.warning("Backtest failed: %s", exc)
        return {"return_pct": 0, "trades": 0, "error": str(exc)}


def run_sensitivity(cfg: dict, param_name: str | None = None) -> list[dict]:
    """Run sensitivity analysis for one or all parameters."""
    params_to_test = {param_name: PARAM_GRID[param_name]} if param_name else PARAM_GRID
    all_results: list[dict] = []

    # Load existing results if any
    if RESULTS_PATH.exists():
        try:
            all_results = json.loads(RESULTS_PATH.read_text())
        except Exception:
            pass

    total = sum(len(p["values"]) for p in params_to_test.values()) * len(TEST_PERIODS)
    done = 0

    for pname, pspec in params_to_test.items():
        print(f"\n{'='*70}")
        print(f"PARAMETER: {pname} (baseline={pspec['baseline']})")
        print(f"  Testing: {pspec['values']}")
        print(f"{'='*70}")

        header = f"{'Value':>8} {'Period':>10} {'Return%':>8} {'Trades':>6} {'Sells':>5} {'Wins':>4} {'WinRate':>7} {'AvgWin':>7} {'AvgLoss':>8} {'MaxDD':>6}"
        print(header)
        print("-" * len(header))

        for value in pspec["values"]:
            for period_name, start, end in TEST_PERIODS:
                done += 1

                # Check if already computed
                existing = [r for r in all_results
                            if r.get("param") == pname
                            and r.get("value") == value
                            and r.get("period") == period_name]
                if existing:
                    m = existing[0]["metrics"]
                    print(f"{value:>8} {period_name:>10} {m['return_pct']:>+7.2f}% {m['trades']:>6} {m.get('sells',0):>5} {m.get('wins',0):>4} {m.get('win_rate',0):>6.1f}% {m.get('avg_win',0):>+6.3f} {m.get('avg_loss',0):>+7.3f} {m.get('max_dd_pct',0):>5.1f}%")
                    continue

                logger.info("[%d/%d] %s=%s period=%s starting...", done, total, pname, value, period_name)
                t0 = time.time()

                test_cfg = copy.deepcopy(cfg)
                _set_nested(test_cfg, pspec["path"], value)

                metrics = _run_single(test_cfg, start, end)
                elapsed = time.time() - t0

                rec = {
                    "param": pname,
                    "value": value,
                    "baseline": value == pspec["baseline"],
                    "period": period_name,
                    "metrics": metrics,
                    "elapsed_sec": round(elapsed, 1),
                }
                all_results.append(rec)

                # Save incrementally
                try:
                    RESULTS_PATH.write_text(json.dumps(all_results, indent=2, default=str))
                except Exception as exc:
                    logger.warning("Failed to save results: %s", exc)

                m = metrics
                print(f"{value:>8} {period_name:>10} {m['return_pct']:>+7.2f}% {m['trades']:>6} {m.get('sells',0):>5} {m.get('wins',0):>4} {m.get('win_rate',0):>6.1f}% {m.get('avg_win',0):>+6.3f} {m.get('avg_loss',0):>+7.3f} {m.get('max_dd_pct',0):>5.1f}%")

        # Print summary for this parameter
        print(f"\n--- {pname} Summary (avg across periods) ---")
        summary_header = f"{'Value':>8} {'AvgReturn':>10} {'TotalTrades':>11} {'WinRate':>7}"
        print(summary_header)
        print("-" * len(summary_header))
        for value in pspec["values"]:
            recs = [r for r in all_results if r["param"] == pname and r["value"] == value]
            if not recs:
                continue
            avg_ret = sum(r["metrics"]["return_pct"] for r in recs) / len(recs)
            total_trades = sum(r["metrics"]["trades"] for r in recs)
            total_sells = sum(r["metrics"].get("sells", 0) for r in recs)
            total_wins = sum(r["metrics"].get("wins", 0) for r in recs)
            wr = total_wins / total_sells * 100 if total_sells else 0
            marker = " ← baseline" if value == pspec["baseline"] else ""
            print(f"{value:>8} {avg_ret:>+9.3f}% {total_trades:>11} {wr:>6.1f}%{marker}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="MR strategy sensitivity analysis")
    parser.add_argument("--param", choices=list(PARAM_GRID.keys()), help="Run single parameter")
    parser.add_argument("--reset", action="store_true", help="Clear previous results")
    args = parser.parse_args()

    if args.reset and RESULTS_PATH.exists():
        RESULTS_PATH.unlink()
        print("Previous results cleared.")

    cfg = load_config("/app/config/config.yaml")
    results = run_sensitivity(cfg, args.param)

    print(f"\n\nResults saved to {RESULTS_PATH}")
    print(f"Total variations tested: {len(results)}")


if __name__ == "__main__":
    main()
