# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""A/B backtest + Bonferroni: single_sided_conviction_multiplier sweep.

Tests SSCM values [1.0 (baseline), 0.8, 0.6, 0.4] across crash + recovery
periods, then runs Bonferroni walk-forward validation on the best candidates.

Usage (inside Docker):
    python3 -m scripts.backtest_sscm_ab
    python3 -m scripts.backtest_sscm_ab --test       # single value, crash only
    python3 -m scripts.backtest_sscm_ab --no-bonferroni
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from app.backtest.agent_engine import run_agent_backtest
from app.backtest.sampling import build_walkforward_windows
from app.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

RESULTS_PATH = Path("/data/reports/sscm_ab_results.json")

TEST_PERIODS = [
    ("crash", "2026-01-28", "2026-02-09"),
    ("recovery", "2026-02-21", "2026-03-03"),
]

SSCM_VALUES = [1.0, 0.8, 0.6, 0.4]


def _run_single(cfg: dict, start: str, end: str) -> dict:
    test_cfg = copy.deepcopy(cfg)
    test_cfg["backtest"]["start"] = start
    test_cfg["backtest"]["end"] = end
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
        return {"return_pct": 0, "trades": 0, "sells": 0, "wins": 0,
                "losses": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0,
                "max_dd_pct": 0, "error": str(exc)}


def _bootstrap_ci(values: list[float], confidence: float, samples: int = 2000, seed: int = 42) -> dict | None:
    if len(values) < 2:
        return None
    import random
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        batch = [rng.choice(values) for _ in range(len(values))]
        means.append(sum(batch) / len(batch))
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = means[max(int(alpha * len(means)), 0)]
    hi = means[min(int((1.0 - alpha) * len(means)) - 1, len(means) - 1)]
    return {"low": lo, "high": hi, "confidence": confidence}


def run_grid(cfg: dict, test_mode: bool = False) -> list[dict]:
    sscm_values = [0.6] if test_mode else SSCM_VALUES
    periods = [TEST_PERIODS[0]] if test_mode else TEST_PERIODS
    total = len(sscm_values) * len(periods)
    done = 0

    print(f"\n{'='*70}")
    print(f"SSCM A/B Grid: single_sided_conviction_multiplier sweep")
    print(f"  SSCM values: {sscm_values}")
    print(f"  Periods: {[p[0] for p in periods]}")
    print(f"  Total runs: {total}")
    print(f"{'='*70}")

    header = f"{'sscm':>6} {'period':>10} {'return%':>8} {'trades':>6} {'sells':>5} {'wins':>4} {'win%':>5} {'maxDD':>6} {'time':>6}"
    print(header)
    print("-" * len(header))
    sys.stdout.flush()

    all_results = []

    for sscm in sscm_values:
        for period_name, start, end in periods:
            done += 1
            logger.info("[%d/%d] sscm=%.2f period=%s starting...", done, total, sscm, period_name)
            sys.stdout.flush()
            t0 = time.time()

            test_cfg = copy.deepcopy(cfg)
            test_cfg.setdefault("strategy", {})["single_sided_conviction_multiplier"] = sscm

            metrics = _run_single(test_cfg, start, end)
            elapsed = time.time() - t0

            rec = {
                "sscm": sscm,
                "period": period_name,
                "metrics": metrics,
                "elapsed_sec": round(elapsed, 1),
            }
            all_results.append(rec)

            m = metrics
            print(f"{sscm:>6.2f} {period_name:>10} {m['return_pct']:>+7.2f}% "
                  f"{m['trades']:>6} {m['sells']:>5} {m['wins']:>4} "
                  f"{m['win_rate']:>4.1f}% {m['max_dd_pct']:>5.1f}% {elapsed:>5.0f}s")
            sys.stdout.flush()

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY — sorted by avg return (crash + recovery)")
    print(f"{'='*70}")

    combos: dict[float, list[dict]] = {}
    for r in all_results:
        combos.setdefault(r["sscm"], []).append(r)

    summary_rows = []
    for sscm, recs in combos.items():
        avg_ret = sum(r["metrics"]["return_pct"] for r in recs) / len(recs)
        total_trades = sum(r["metrics"]["trades"] for r in recs)
        total_sells = sum(r["metrics"].get("sells", 0) for r in recs)
        total_wins = sum(r["metrics"].get("wins", 0) for r in recs)
        wr = total_wins / total_sells * 100 if total_sells else 0
        max_dd = max(r["metrics"].get("max_dd_pct", 0) for r in recs)
        crash_ret = next((r["metrics"]["return_pct"] for r in recs if r["period"] == "crash"), None)
        recov_ret = next((r["metrics"]["return_pct"] for r in recs if r["period"] == "recovery"), None)
        summary_rows.append({
            "sscm": sscm,
            "avg_return": avg_ret,
            "crash_return": crash_ret,
            "recovery_return": recov_ret,
            "total_trades": total_trades,
            "total_sells": total_sells,
            "total_wins": total_wins,
            "win_rate": wr,
            "max_dd": max_dd,
        })

    summary_rows.sort(key=lambda x: x["avg_return"], reverse=True)

    sh = f"{'rank':>4} {'sscm':>6} {'avg_ret':>8} {'crash':>7} {'recov':>7} {'trades':>6} {'wins':>4} {'win%':>5} {'maxDD':>6}"
    print(sh)
    print("-" * len(sh))
    for i, row in enumerate(summary_rows, 1):
        cs = f"{row['crash_return']:>+6.2f}%" if row["crash_return"] is not None else "   N/A "
        rs = f"{row['recovery_return']:>+6.2f}%" if row["recovery_return"] is not None else "   N/A "
        base = " *" if row["sscm"] == 1.0 else ""
        print(f"{i:>4} {row['sscm']:>6.2f} {row['avg_return']:>+7.2f}% {cs} {rs} "
              f"{row['total_trades']:>6} {row['total_wins']:>4} {row['win_rate']:>4.1f}% "
              f"{row['max_dd']:>5.1f}%{base}")

    print(f"\n* = current baseline (sscm=1.0)")
    sys.stdout.flush()

    # Save grid results
    try:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(all_results, indent=2, default=str))
        print(f"Grid results saved to {RESULTS_PATH}")
    except Exception as exc:
        logger.warning("Failed to save results: %s", exc)

    return all_results


def run_bonferroni(cfg: dict, summary_rows: list[dict], top_n: int = 4) -> list[dict]:
    # Only validate non-baseline candidates that beat baseline
    baseline_ret = next((r["avg_return"] for r in summary_rows if r["sscm"] == 1.0), 0.0)
    candidates = [r for r in summary_rows if r["sscm"] != 1.0 and r["avg_return"] > baseline_ret and r["total_trades"] > 0]
    # Also include baseline for reference
    baseline_row = next((r for r in summary_rows if r["sscm"] == 1.0), None)

    if not candidates:
        print("\nNo SSCM candidates beat baseline. Nothing to validate.")
        return []

    # Include baseline as first candidate for comparison
    all_candidates = []
    if baseline_row:
        all_candidates.append(baseline_row)
    all_candidates.extend(candidates[:top_n])

    n_candidates = len(all_candidates)
    family_alpha = 0.05
    adj_alpha = family_alpha / n_candidates
    adj_confidence = 1.0 - adj_alpha

    bt = cfg.get("backtest", {})
    full_start = datetime.strptime(bt["start"], "%Y-%m-%d")
    full_end = datetime.strptime(bt["end"], "%Y-%m-%d")
    total_days = (full_end - full_start).days
    holdout_days = max(int(total_days * 0.20), 7)
    train_end = full_end - timedelta(days=holdout_days)
    total_train_days = (train_end - full_start).days

    if total_train_days >= 60:
        train_days = max(total_train_days // 3, 14)
        test_days = max(total_train_days // 6, 7)
    else:
        train_days = max(total_train_days // 2, 7)
        test_days = max(total_train_days // 4, 5)
    embargo_days = 2
    step_days = test_days

    folds = build_walkforward_windows(full_start, train_end, train_days=train_days,
                                       embargo_days=embargo_days, test_days=test_days,
                                       step_days=step_days)

    print(f"\n{'='*70}")
    print(f"PHASE 2: BONFERRONI WALK-FORWARD VALIDATION")
    print(f"{'='*70}")
    print(f"Full period:    {full_start:%Y-%m-%d} -> {full_end:%Y-%m-%d} ({total_days}d)")
    print(f"Train/Validate: {full_start:%Y-%m-%d} -> {train_end:%Y-%m-%d} ({total_train_days}d)")
    print(f"Hold-out:       {train_end:%Y-%m-%d} -> {full_end:%Y-%m-%d} ({holdout_days}d)")
    print(f"Walk-forward:   train={train_days}d embargo={embargo_days}d test={test_days}d")
    print(f"Folds: {len(folds)}")
    print(f"Candidates: {n_candidates}  Family alpha: {family_alpha}  Adjusted alpha: {adj_alpha:.4f}")
    print(f"CI confidence: {adj_confidence*100:.2f}%")
    sys.stdout.flush()

    bonf_results = []

    for rank, row in enumerate(all_candidates, 1):
        sscm = row["sscm"]
        tag = " (baseline)" if sscm == 1.0 else ""
        print(f"\n--- Candidate {rank}/{n_candidates}: sscm={sscm:.2f}{tag} "
              f"(grid avg_ret={row['avg_return']:+.2f}%) ---")
        sys.stdout.flush()

        test_cfg = copy.deepcopy(cfg)
        test_cfg.setdefault("strategy", {})["single_sided_conviction_multiplier"] = sscm

        fold_returns: list[float] = []
        fold_trades: list[int] = []

        header = f"  {'Fold':>4}  {'Test period':>23}  {'Return%':>8}  {'Trades':>6}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for idx, (train_start, train_end_f, test_start, test_end) in enumerate(folds, 1):
            fold_cfg = copy.deepcopy(test_cfg)
            fold_cfg["backtest"]["start"] = test_start.strftime("%Y-%m-%d")
            fold_cfg["backtest"]["end"] = test_end.strftime("%Y-%m-%d")
            fold_cfg["backtest"].pop("walk_forward", None)

            try:
                result = run_agent_backtest(fold_cfg)
                ret_pct = float(result.return_pct) if hasattr(result, "return_pct") else float(result.average_return_pct)
                trades = int(result.trades) if hasattr(result, "trades") else int(result.total_trades)
            except Exception as exc:
                logger.warning("Fold %d failed: %s", idx, exc)
                ret_pct, trades = 0.0, 0

            fold_returns.append(ret_pct)
            fold_trades.append(trades)
            test_str = f"{test_start:%Y-%m-%d} -> {test_end:%Y-%m-%d}"
            print(f"  {idx:>4}  {test_str:>23}  {ret_pct:>+7.2f}%  {trades:>6}")
            sys.stdout.flush()

        avg_ret = sum(fold_returns) / len(fold_returns) if fold_returns else 0.0
        total_t = sum(fold_trades)
        ci = _bootstrap_ci(fold_returns, confidence=adj_confidence)

        passes = False
        if ci:
            passes = ci["low"] > 0.0

        print(f"  {'AGG':>4}  {'':>23}  {avg_ret:>+7.2f}%  {total_t:>6}")
        if ci:
            print(f"  CI ({adj_confidence*100:.1f}%): [{ci['low']:+.3f}%, {ci['high']:+.3f}%]")
            print(f"  Passes Bonferroni: {'YES' if passes else 'NO'}")
        else:
            print(f"  Insufficient folds for CI")
        sys.stdout.flush()

        bonf_results.append({
            "rank": rank,
            "sscm": sscm,
            "is_baseline": sscm == 1.0,
            "grid_avg_return": row["avg_return"],
            "wf_avg_return": avg_ret,
            "wf_total_trades": total_t,
            "fold_returns": fold_returns,
            "ci_low": ci["low"] if ci else None,
            "ci_high": ci["high"] if ci else None,
            "passes_bonferroni": passes,
        })

    # Save
    bonf_path = RESULTS_PATH.parent / "sscm_bonferroni.json"
    try:
        bonf_path.write_text(json.dumps(bonf_results, indent=2, default=str))
    except Exception as exc:
        logger.warning("Failed to save Bonferroni results: %s", exc)

    # Summary
    print(f"\n{'='*70}")
    print(f"BONFERRONI SUMMARY")
    print(f"{'='*70}")
    bh = f"{'rank':>4} {'sscm':>6} {'grid_ret':>8} {'wf_ret':>7} {'trades':>6} {'CI_low':>7} {'CI_high':>7} {'pass':>5}"
    print(bh)
    print("-" * len(bh))
    for br in bonf_results:
        ci_lo = f"{br['ci_low']:+.3f}" if br["ci_low"] is not None else "  N/A"
        ci_hi = f"{br['ci_high']:+.3f}" if br["ci_high"] is not None else "  N/A"
        pf = "YES" if br["passes_bonferroni"] else "NO"
        tag = " *" if br["is_baseline"] else ""
        print(f"{br['rank']:>4} {br['sscm']:>6.2f} {br['grid_avg_return']:>+7.2f}% "
              f"{br['wf_avg_return']:>+6.2f}% {br['wf_total_trades']:>6} "
              f"{ci_lo:>7} {ci_hi:>7} {pf:>5}{tag}")

    print(f"\n* = baseline (sscm=1.0)")
    print(f"Bonferroni results saved to {bonf_path}")
    sys.stdout.flush()
    return bonf_results


def main():
    parser = argparse.ArgumentParser(description="SSCM A/B backtest + Bonferroni")
    parser.add_argument("--test", action="store_true", help="Test mode: sscm=0.6, crash only")
    parser.add_argument("--no-bonferroni", action="store_true", help="Skip Bonferroni")
    parser.add_argument("--top", type=int, default=4, help="Top N candidates for Bonferroni")
    args = parser.parse_args()

    cfg = load_config("/app/config/config.yaml")
    results = run_grid(cfg, test_mode=args.test)
    print(f"\nPhase 1 complete: {len(results)} runs.")

    if not args.test and not args.no_bonferroni:
        combos: dict[float, list[dict]] = {}
        for r in results:
            combos.setdefault(r["sscm"], []).append(r)

        summary_rows = []
        for sscm, recs in combos.items():
            avg_ret = sum(r["metrics"]["return_pct"] for r in recs) / len(recs)
            total_trades = sum(r["metrics"]["trades"] for r in recs)
            total_sells = sum(r["metrics"].get("sells", 0) for r in recs)
            total_wins = sum(r["metrics"].get("wins", 0) for r in recs)
            summary_rows.append({
                "sscm": sscm,
                "avg_return": avg_ret,
                "total_trades": total_trades,
                "total_sells": total_sells,
                "total_wins": total_wins,
            })
        summary_rows.sort(key=lambda x: x["avg_return"], reverse=True)

        run_bonferroni(cfg, summary_rows, top_n=args.top)


if __name__ == "__main__":
    main()
