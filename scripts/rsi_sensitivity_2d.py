# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""2D sensitivity grid sweep: rsi_period x rsi_oversold for crypto_mean_reversion.

Phase 1: Sweeps 30 combinations across two test periods (crash + recovery).
Phase 2: Runs Bonferroni walk-forward validation on top winners.
All other MR params held at validated values.

Usage:
    # Full run (inside Docker):
    python3 -m scripts.rsi_sensitivity_2d

    # Test run (single baseline combo, crash period only):
    python3 -m scripts.rsi_sensitivity_2d --test

    # Resume (skips already-computed combos):
    python3 -m scripts.rsi_sensitivity_2d --resume

    # Skip Bonferroni (grid sweep only):
    python3 -m scripts.rsi_sensitivity_2d --no-bonferroni

    # Run Bonferroni on top N (default 5):
    python3 -m scripts.rsi_sensitivity_2d --top 3
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

RESULTS_PATH = Path("/data/reports/rsi_sensitivity_2d_results.json")

# Two test periods
TEST_PERIODS = [
    ("crash", "2026-01-28", "2026-02-09"),
    ("recovery", "2026-02-21", "2026-03-03"),
]

# 2D grid
RSI_PERIODS = [5, 10, 15, 20, 30, 50]
RSI_OVERSOLD = [10, 15, 20, 25, 30]

# Fixed params (validated values)
FIXED_PARAMS = {
    "strategy.params.crypto_mean_reversion.bb_period": 50,
    "strategy.params.crypto_mean_reversion.bb_std": 2.0,
    "strategy.params.crypto_mean_reversion.drop_window_bars": 75,
    "strategy.params.crypto_mean_reversion.hard_stop_pct": 5.0,
    "strategy.params.crypto_mean_reversion.crash_filter_pct": 0,
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
        return {"return_pct": 0, "trades": 0, "sells": 0, "wins": 0,
                "losses": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0,
                "max_dd_pct": 0, "error": str(exc)}


def _load_existing_results() -> list[dict]:
    """Load previously saved results for resume support."""
    if RESULTS_PATH.exists():
        try:
            return json.loads(RESULTS_PATH.read_text())
        except Exception:
            pass
    return []


def _save_results(results: list[dict]) -> None:
    """Save results incrementally."""
    try:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    except Exception as exc:
        logger.warning("Failed to save results: %s", exc)


def _find_existing(results: list[dict], rsi_period: int, rsi_oversold: float, period_name: str) -> dict | None:
    """Find an existing result for a given combo + period."""
    for r in results:
        if (r.get("rsi_period") == rsi_period
                and r.get("rsi_oversold") == rsi_oversold
                and r.get("period") == period_name):
            return r
    return None


def run_grid(cfg: dict, test_mode: bool = False, resume: bool = False) -> list[dict]:
    """Run the full 2D grid sweep."""
    rsi_periods = [30] if test_mode else RSI_PERIODS
    rsi_oversolds = [30] if test_mode else RSI_OVERSOLD
    periods = [TEST_PERIODS[0]] if test_mode else TEST_PERIODS  # crash only for test

    all_results = _load_existing_results() if resume else []
    total = len(rsi_periods) * len(rsi_oversolds) * len(periods)
    done = 0
    skipped = 0
    t_start = time.time()

    print(f"\n{'='*80}")
    print(f"RSI 2D Sensitivity Grid: rsi_period x rsi_oversold")
    print(f"  rsi_period values:  {rsi_periods}")
    print(f"  rsi_oversold values: {rsi_oversolds}")
    print(f"  periods: {[p[0] for p in periods]}")
    print(f"  total runs: {total}")
    if resume:
        print(f"  resume mode: ON (existing results: {len(all_results)})")
    print(f"{'='*80}")
    sys.stdout.flush()

    # Print header
    header = (f"{'rsi_per':>7} {'rsi_os':>6} {'period':>10} {'return%':>8} "
              f"{'trades':>6} {'wins':>4} {'win%':>5} {'maxDD':>6} {'time':>6}")
    print(header)
    print("-" * len(header))
    sys.stdout.flush()

    for rsi_period in rsi_periods:
        for rsi_oversold in rsi_oversolds:
            for period_name, start, end in periods:
                done += 1

                # Check if already computed (resume mode)
                existing = _find_existing(all_results, rsi_period, rsi_oversold, period_name)
                if existing and resume:
                    m = existing["metrics"]
                    print(f"{rsi_period:>7} {rsi_oversold:>6} {period_name:>10} "
                          f"{m['return_pct']:>+7.2f}% {m['trades']:>6} "
                          f"{m.get('wins',0):>4} {m.get('win_rate',0):>4.1f}% "
                          f"{m.get('max_dd_pct',0):>5.1f}% {'skip':>6}")
                    sys.stdout.flush()
                    skipped += 1
                    continue

                logger.info("[%d/%d] rsi_period=%d rsi_oversold=%d period=%s starting...",
                            done, total, rsi_period, rsi_oversold, period_name)
                sys.stdout.flush()
                t0 = time.time()

                # Build config with this combo's params
                test_cfg = copy.deepcopy(cfg)
                _set_nested(test_cfg, "strategy.params.crypto_mean_reversion.rsi_period", rsi_period)
                _set_nested(test_cfg, "strategy.params.crypto_mean_reversion.rsi_oversold", float(rsi_oversold))
                for key, value in FIXED_PARAMS.items():
                    _set_nested(test_cfg, key, value)

                metrics = _run_single(test_cfg, start, end)
                elapsed = time.time() - t0

                rec = {
                    "rsi_period": rsi_period,
                    "rsi_oversold": rsi_oversold,
                    "period": period_name,
                    "metrics": metrics,
                    "elapsed_sec": round(elapsed, 1),
                }
                all_results.append(rec)
                _save_results(all_results)

                m = metrics
                print(f"{rsi_period:>7} {rsi_oversold:>6} {period_name:>10} "
                      f"{m['return_pct']:>+7.2f}% {m['trades']:>6} "
                      f"{m.get('wins',0):>4} {m.get('win_rate',0):>4.1f}% "
                      f"{m.get('max_dd_pct',0):>5.1f}% {elapsed:>5.0f}s")
                sys.stdout.flush()

    total_time = time.time() - t_start

    # --- Summary table: average of crash + recovery, sorted by avg_return ---
    print(f"\n{'='*80}")
    print(f"SUMMARY — sorted by avg return (crash + recovery)")
    print(f"  Total time: {total_time:.0f}s | Runs: {done} | Skipped: {skipped}")
    print(f"{'='*80}")

    # Build summary: for each (rsi_period, rsi_oversold), average across periods
    combos: dict[tuple[int, float], list[dict]] = {}
    for r in all_results:
        key = (r["rsi_period"], r["rsi_oversold"])
        combos.setdefault(key, []).append(r)

    summary_rows = []
    for (rp, ro), recs in combos.items():
        avg_ret = sum(r["metrics"]["return_pct"] for r in recs) / len(recs)
        total_trades = sum(r["metrics"]["trades"] for r in recs)
        total_sells = sum(r["metrics"].get("sells", 0) for r in recs)
        total_wins = sum(r["metrics"].get("wins", 0) for r in recs)
        wr = total_wins / total_sells * 100 if total_sells else 0
        max_dd = max(r["metrics"].get("max_dd_pct", 0) for r in recs)
        # Per-period returns for display
        crash_ret = next((r["metrics"]["return_pct"] for r in recs if r["period"] == "crash"), None)
        recov_ret = next((r["metrics"]["return_pct"] for r in recs if r["period"] == "recovery"), None)
        summary_rows.append({
            "rsi_period": rp,
            "rsi_oversold": ro,
            "avg_return": avg_ret,
            "crash_return": crash_ret,
            "recovery_return": recov_ret,
            "total_trades": total_trades,
            "total_wins": total_wins,
            "win_rate": wr,
            "max_dd": max_dd,
        })

    summary_rows.sort(key=lambda x: x["avg_return"], reverse=True)

    sum_header = (f"{'rank':>4} {'rsi_per':>7} {'rsi_os':>6} {'avg_ret':>8} "
                  f"{'crash':>7} {'recov':>7} {'trades':>6} {'wins':>4} "
                  f"{'win%':>5} {'maxDD':>6}")
    print(sum_header)
    print("-" * len(sum_header))

    for i, row in enumerate(summary_rows, 1):
        crash_str = f"{row['crash_return']:>+6.2f}%" if row["crash_return"] is not None else "   N/A "
        recov_str = f"{row['recovery_return']:>+6.2f}%" if row["recovery_return"] is not None else "   N/A "
        baseline = " *" if row["rsi_period"] == 30 and row["rsi_oversold"] == 30 else ""
        print(f"{i:>4} {row['rsi_period']:>7} {row['rsi_oversold']:>6} "
              f"{row['avg_return']:>+7.2f}% "
              f"{crash_str} {recov_str} "
              f"{row['total_trades']:>6} {row['total_wins']:>4} "
              f"{row['win_rate']:>4.1f}% "
              f"{row['max_dd']:>5.1f}%{baseline}")

    print(f"\n* = current baseline (rsi_period=30, rsi_oversold=30)")
    print(f"Results saved to {RESULTS_PATH}")
    sys.stdout.flush()
    return all_results


def _bootstrap_ci(values: list[float], confidence: float, samples: int = 2000, seed: int = 42) -> dict | None:
    """Bootstrap confidence interval for mean."""
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
    hi = means[min(int((1.0 - confidence) / 2.0 * len(means) + confidence * len(means)) - 1, len(means) - 1)]
    return {"low": lo, "high": hi, "confidence": confidence}


def run_bonferroni(cfg: dict, summary_rows: list[dict], top_n: int = 5) -> list[dict]:
    """Phase 2: Bonferroni walk-forward validation on top grid winners.

    Uses the full backtest data range from config (not just crash+recovery).
    """
    # Filter to positive avg return, take top N
    winners = [r for r in summary_rows if r["avg_return"] > 0 and r["total_trades"] > 0]
    winners = winners[:top_n]  # already sorted desc by avg_return

    if not winners:
        print("\nNo positive-return combos to validate.")
        return []

    n_candidates = len(winners)
    family_alpha = 0.05
    adj_alpha = family_alpha / n_candidates
    adj_confidence = 1.0 - adj_alpha

    # Data split: full period from config
    bt = cfg.get("backtest", {})
    full_start = datetime.strptime(bt["start"], "%Y-%m-%d")
    full_end = datetime.strptime(bt["end"], "%Y-%m-%d")
    total_days = (full_end - full_start).days
    holdout_days = max(int(total_days * 0.20), 7)
    train_end = full_end - timedelta(days=holdout_days)
    total_train_days = (train_end - full_start).days

    # Walk-forward params
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

    print(f"\n{'='*80}")
    print(f"PHASE 2: BONFERRONI WALK-FORWARD VALIDATION")
    print(f"{'='*80}")
    print(f"Full period:    {full_start:%Y-%m-%d} → {full_end:%Y-%m-%d} ({total_days}d)")
    print(f"Train/Validate: {full_start:%Y-%m-%d} → {train_end:%Y-%m-%d} ({total_train_days}d)")
    print(f"Hold-out:       {train_end:%Y-%m-%d} → {full_end:%Y-%m-%d} ({holdout_days}d) [reserved]")
    print(f"Walk-forward:   train={train_days}d embargo={embargo_days}d test={test_days}d")
    print(f"Folds: {len(folds)}")
    print(f"Candidates: {n_candidates}  Family α: {family_alpha}  Adjusted α: {adj_alpha:.4f}")
    print(f"CI confidence: {adj_confidence*100:.2f}%")
    sys.stdout.flush()

    bonf_results = []

    for rank, row in enumerate(winners, 1):
        rp = row["rsi_period"]
        ro = row["rsi_oversold"]
        print(f"\n--- Candidate {rank}/{n_candidates}: rsi_period={rp} rsi_oversold={ro} "
              f"(grid avg_ret={row['avg_return']:+.2f}%) ---")
        sys.stdout.flush()

        # Build config with this combo's params
        test_cfg = copy.deepcopy(cfg)
        _set_nested(test_cfg, "strategy.params.crypto_mean_reversion.rsi_period", rp)
        _set_nested(test_cfg, "strategy.params.crypto_mean_reversion.rsi_oversold", float(ro))
        for key, value in FIXED_PARAMS.items():
            _set_nested(test_cfg, key, value)

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
            test_str = f"{test_start:%Y-%m-%d} → {test_end:%Y-%m-%d}"
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
            "rsi_period": rp,
            "rsi_oversold": ro,
            "grid_avg_return": row["avg_return"],
            "wf_avg_return": avg_ret,
            "wf_total_trades": total_t,
            "fold_returns": fold_returns,
            "ci_low": ci["low"] if ci else None,
            "ci_high": ci["high"] if ci else None,
            "passes_bonferroni": passes,
        })

    # Save Bonferroni results
    bonf_path = RESULTS_PATH.parent / "rsi_sensitivity_2d_bonferroni.json"
    try:
        bonf_path.write_text(json.dumps(bonf_results, indent=2, default=str))
    except Exception as exc:
        logger.warning("Failed to save Bonferroni results: %s", exc)

    # Final summary
    print(f"\n{'='*80}")
    print(f"BONFERRONI SUMMARY")
    print(f"{'='*80}")
    bh = (f"{'rank':>4} {'rsi_per':>7} {'rsi_os':>6} {'grid_ret':>8} "
          f"{'wf_ret':>7} {'trades':>6} {'CI_low':>7} {'CI_high':>7} {'pass':>5}")
    print(bh)
    print("-" * len(bh))
    for br in bonf_results:
        ci_lo = f"{br['ci_low']:+.3f}" if br["ci_low"] is not None else "  N/A"
        ci_hi = f"{br['ci_high']:+.3f}" if br["ci_high"] is not None else "  N/A"
        pf = "YES" if br["passes_bonferroni"] else "NO"
        print(f"{br['rank']:>4} {br['rsi_period']:>7} {br['rsi_oversold']:>6} "
              f"{br['grid_avg_return']:>+7.2f}% {br['wf_avg_return']:>+6.2f}% "
              f"{br['wf_total_trades']:>6} {ci_lo:>7} {ci_hi:>7} {pf:>5}")

    print(f"\nBonferroni results saved to {bonf_path}")
    sys.stdout.flush()
    return bonf_results


def main():
    parser = argparse.ArgumentParser(description="2D RSI sensitivity grid sweep + Bonferroni")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: single baseline combo (rsi_period=30, rsi_oversold=30), crash only")
    parser.add_argument("--resume", action="store_true",
                        help="Resume: skip already-computed combos")
    parser.add_argument("--reset", action="store_true",
                        help="Clear previous results before running")
    parser.add_argument("--no-bonferroni", action="store_true",
                        help="Skip Bonferroni validation (grid sweep only)")
    parser.add_argument("--top", type=int, default=5,
                        help="Number of top winners to validate with Bonferroni (default 5)")
    args = parser.parse_args()

    if args.reset and RESULTS_PATH.exists():
        RESULTS_PATH.unlink()
        print("Previous results cleared.")

    cfg = load_config("/app/config/config.yaml")
    results = run_grid(cfg, test_mode=args.test, resume=args.resume)
    print(f"\nPhase 1 complete: {len(results)} variations tested.")

    if not args.test and not args.no_bonferroni:
        # Build summary for Bonferroni phase
        combos: dict[tuple[int, float], list[dict]] = {}
        for r in results:
            key = (r["rsi_period"], r["rsi_oversold"])
            combos.setdefault(key, []).append(r)

        summary_rows = []
        for (rp, ro), recs in combos.items():
            avg_ret = sum(r["metrics"]["return_pct"] for r in recs) / len(recs)
            total_trades = sum(r["metrics"]["trades"] for r in recs)
            summary_rows.append({
                "rsi_period": rp,
                "rsi_oversold": ro,
                "avg_return": avg_ret,
                "total_trades": total_trades,
            })
        summary_rows.sort(key=lambda x: x["avg_return"], reverse=True)

        run_bonferroni(cfg, summary_rows, top_n=args.top)


if __name__ == "__main__":
    main()
