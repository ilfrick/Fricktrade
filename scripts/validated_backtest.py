# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""Bonferroni-corrected backtesting with walk-forward validation and hold-out gate.

Tracks every config tested, applies multiple-comparison correction, and prevents
overfitting by enforcing a strict train/validate/hold-out split.

Usage:
    python3 -m scripts.validated_backtest register [--name NAME]
    python3 -m scripts.validated_backtest validate [--candidate N]
    python3 -m scripts.validated_backtest holdout  [--candidate N]  (one shot!)
    python3 -m scripts.validated_backtest status
    python3 -m scripts.validated_backtest reject   [--candidate N]

Data split (90-day example, Dec 21 - Mar 21):
    Train/Validate: Dec 21 - Mar 06  (75 days)
    Hold-out:       Mar 06 - Mar 21  (15 days, sacred — never touched until final eval)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

from app.backtest.agent_engine import run_agent_backtest
from app.backtest.sampling import build_walkforward_windows
from app.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REGISTRY_PATH = Path("/data/reports/bonferroni_registry.json")
ALPHA = 0.05  # Family-wise error rate
HOLDOUT_FRACTION = 0.20  # Reserve last 20% of data


# ---------------------------------------------------------------------------
# Config fingerprinting — hash only strategy-relevant keys
# ---------------------------------------------------------------------------

_STRATEGY_KEYS = [
    "strategy.names",
    "strategy.combine",
    "strategy.weights",
    "strategy.min_conviction",
    "strategy.vote_threshold",
    "strategy.params",
    "strategy.strategies",
    "data.interval",
    "risk.vol_targeting",
    "risk.crypto.hard_stop_pct",
    "risk.crypto.trailing_stop_pct",
    "risk.crypto.take_profit_pct",
    "risk.crypto.circuit_breaker_drawdown_pct",
    "risk.hard_stop_pct",
    "risk.take_profit_pct",
    "risk.partial_take_profit_pct",
    "execution.position_exit.peak_detection",
    "order_book",
]


def _extract_key(cfg: dict, dotted: str):
    """Extract a nested config value by dotted path."""
    parts = dotted.split(".")
    node = cfg
    for p in parts:
        if not isinstance(node, dict):
            return None
        node = node.get(p)
    return node


def _config_fingerprint(cfg: dict) -> tuple[str, dict]:
    """Return (sha256_hex, strategy_snapshot) for the strategy-relevant config."""
    snapshot = {}
    for key in _STRATEGY_KEYS:
        val = _extract_key(cfg, key)
        if val is not None:
            snapshot[key] = val
    raw = json.dumps(snapshot, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16], snapshot


# ---------------------------------------------------------------------------
# Registry I/O
# ---------------------------------------------------------------------------

def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {"alpha": ALPHA, "candidates": []}


def _save_registry(reg: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2, default=str) + "\n")


def _active_candidates(reg: dict) -> list[dict]:
    """Return candidates that haven't been rejected."""
    return [c for c in reg["candidates"] if c.get("status") != "rejected"]


def _bonferroni_threshold(reg: dict) -> float:
    """Adjusted alpha = family_alpha / N_active_candidates."""
    n = max(len(_active_candidates(reg)), 1)
    return reg.get("alpha", ALPHA) / n


# ---------------------------------------------------------------------------
# Scorecard (reused from benchmark_runner)
# ---------------------------------------------------------------------------

def _scorecard(returns: list[float]) -> dict:
    if not returns:
        return {"sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "max_dd_pct": 0.0}
    ret = [r / 100.0 for r in returns]
    mean = sum(ret) / len(ret)
    var = sum((r - mean) ** 2 for r in ret) / max(len(ret), 1)
    std = var ** 0.5
    downside = [r for r in ret if r < 0]
    d_std = (sum((r - (sum(downside) / len(downside))) ** 2 for r in downside) / len(downside)) ** 0.5 if downside else 0.0
    sharpe = mean / std if std else 0.0
    sortino = mean / d_std if d_std else 0.0
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in ret:
        equity *= 1.0 + r
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak else 0.0)
    calmar = mean / max_dd if max_dd else 0.0
    return {"sharpe": sharpe, "sortino": sortino, "calmar": calmar, "max_dd_pct": max_dd * 100}


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
    hi = means[min(int((1.0 - alpha) * len(means)) - 1, len(means) - 1)]
    return {"low": lo, "high": hi, "confidence": confidence}


# ---------------------------------------------------------------------------
# Apply config updates (from benchmark_runner)
# ---------------------------------------------------------------------------

def _apply_updates(cfg: dict, updates: dict) -> dict:
    new_cfg = copy.deepcopy(cfg)
    for key, value in updates.items():
        section, field = key.split(".", 1)
        if section not in new_cfg:
            new_cfg[section] = {}
        target = new_cfg[section]
        parts = field.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return new_cfg


# ---------------------------------------------------------------------------
# Date split logic
# ---------------------------------------------------------------------------

def _compute_split(cfg: dict) -> tuple[datetime, datetime, datetime]:
    """Return (full_start, holdout_start, full_end).

    Train/validate: full_start → holdout_start
    Hold-out:       holdout_start → full_end
    """
    bt = cfg.get("backtest", {})
    start = datetime.strptime(bt["start"], "%Y-%m-%d")
    end = datetime.strptime(bt["end"], "%Y-%m-%d")
    total_days = (end - start).days
    holdout_days = max(int(total_days * HOLDOUT_FRACTION), 7)
    holdout_start = end - timedelta(days=holdout_days)
    return start, holdout_start, end


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_register(cfg: dict, name: str | None) -> None:
    """Register current config as a candidate."""
    reg = _load_registry()
    fp, snapshot = _config_fingerprint(cfg)

    # Check for duplicate
    for c in reg["candidates"]:
        if c["fingerprint"] == fp and c.get("status") != "rejected":
            print(f"Config already registered as candidate #{c['id']} (fingerprint {fp})")
            return

    cid = len(reg["candidates"]) + 1
    candidate = {
        "id": cid,
        "name": name or f"candidate_{cid}",
        "fingerprint": fp,
        "registered_at": datetime.now().isoformat(),
        "status": "registered",
        "config_snapshot": snapshot,
        "validation": None,
        "holdout": None,
    }
    reg["candidates"].append(candidate)
    _save_registry(reg)

    n_active = len(_active_candidates(reg))
    adj_alpha = _bonferroni_threshold(reg)
    print(f"Registered candidate #{cid} '{candidate['name']}' (fingerprint {fp})")
    print(f"Active candidates: {n_active} → Bonferroni α = {ALPHA}/{n_active} = {adj_alpha:.4f}")
    print(f"Config snapshot keys: {list(snapshot.keys())}")


def cmd_validate(cfg: dict, candidate_id: int | None) -> None:
    """Run walk-forward validation on the train period (NOT hold-out)."""
    reg = _load_registry()
    active = _active_candidates(reg)
    if not active:
        print("No candidates registered. Run 'register' first.")
        return

    candidate = None
    if candidate_id:
        candidate = next((c for c in reg["candidates"] if c["id"] == candidate_id), None)
    else:
        candidate = active[-1]  # most recent
    if not candidate:
        print(f"Candidate #{candidate_id} not found.")
        return

    start, holdout_start, end = _compute_split(cfg)
    train_end = holdout_start
    total_train_days = (train_end - start).days

    # Walk-forward params — scale to available data
    if total_train_days >= 60:
        train_days = max(total_train_days // 3, 14)
        test_days = max(total_train_days // 6, 7)
    else:
        train_days = max(total_train_days // 2, 7)
        test_days = max(total_train_days // 4, 5)
    embargo_days = 2
    step_days = test_days  # non-overlapping test windows

    print(f"\n{'='*70}")
    print(f"VALIDATED BACKTEST — Candidate #{candidate['id']} '{candidate['name']}'")
    print(f"{'='*70}")
    print(f"Full period:    {start:%Y-%m-%d} → {end:%Y-%m-%d} ({(end-start).days}d)")
    print(f"Train/Validate: {start:%Y-%m-%d} → {train_end:%Y-%m-%d} ({total_train_days}d)")
    print(f"Hold-out:       {train_end:%Y-%m-%d} → {end:%Y-%m-%d} ({(end-train_end).days}d) [SACRED]")
    print(f"Walk-forward:   train={train_days}d embargo={embargo_days}d test={test_days}d step={step_days}d")

    folds = build_walkforward_windows(start, train_end, train_days=train_days,
                                       embargo_days=embargo_days, test_days=test_days,
                                       step_days=step_days)
    if not folds:
        print("No walk-forward folds possible with available data.")
        return

    print(f"Folds: {len(folds)}\n")

    # Run each fold on OOS test window only
    fold_cfg_base = _apply_updates(cfg, {
        "backtest.start": start.strftime("%Y-%m-%d"),
        "backtest.end": train_end.strftime("%Y-%m-%d"),
    })

    fold_returns: list[float] = []
    fold_trades: list[int] = []
    fold_sharpes: list[float] = []
    all_trade_details: list[dict] = []
    all_signal_logs: list[dict] = []

    header = f"{'Fold':>4}  {'Test period':>23}  {'Return%':>8}  {'Trades':>6}  {'Sharpe':>8}"
    print(header)
    print("-" * len(header))

    for idx, (train_start, train_end_f, test_start, test_end) in enumerate(folds, 1):
        fold_cfg = _apply_updates(fold_cfg_base, {
            "backtest.start": test_start.strftime("%Y-%m-%d"),
            "backtest.end": test_end.strftime("%Y-%m-%d"),
        })
        try:
            result = run_agent_backtest(fold_cfg)
            ret_pct = float(result.return_pct) if hasattr(result, "return_pct") else float(result.average_return_pct)
            trades = int(result.trades) if hasattr(result, "trades") else int(result.total_trades)
            sc = _scorecard([ret_pct])
            sharpe = sc["sharpe"]
            # Collect trade details for analysis
            all_trade_details.extend(
                {**t, "fold": idx, "fold_start": test_start.strftime("%Y-%m-%d"),
                 "fold_end": test_end.strftime("%Y-%m-%d")}
                for t in getattr(result, "trade_details", [])
            )
            # Collect per-strategy signal logs
            all_signal_logs.extend(
                {**s, "fold": idx}
                for s in getattr(result, "signal_log", [])
            )
        except Exception as exc:
            logger.warning("Fold %d failed: %s", idx, exc)
            ret_pct, trades, sharpe = 0.0, 0, 0.0

        fold_returns.append(ret_pct)
        fold_trades.append(trades)
        fold_sharpes.append(sharpe)

        test_str = f"{test_start:%Y-%m-%d} → {test_end:%Y-%m-%d}"
        print(f"{idx:>4}  {test_str:>23}  {ret_pct:>+7.2f}%  {trades:>6}  {sharpe:>8.3f}")

    # Aggregate
    print("-" * len(header))
    avg_ret = sum(fold_returns) / len(fold_returns) if fold_returns else 0.0
    total_trades = sum(fold_trades)
    overall_sc = _scorecard(fold_returns)
    print(f"{'AGG':>4}  {'':>23}  {avg_ret:>+7.2f}%  {total_trades:>6}  {overall_sc['sharpe']:>8.3f}")

    # Bonferroni-adjusted significance test
    adj_alpha = _bonferroni_threshold(reg)
    adj_confidence = 1.0 - adj_alpha
    ci = _bootstrap_ci(fold_returns, confidence=adj_confidence)

    print(f"\n--- Bonferroni Significance Test ---")
    n_active = len(_active_candidates(reg))
    print(f"Active candidates: {n_active}")
    print(f"Family α: {ALPHA}  →  Adjusted α: {adj_alpha:.4f}  →  CI confidence: {adj_confidence*100:.2f}%")
    if ci:
        print(f"OOS return CI ({adj_confidence*100:.1f}%): [{ci['low']:+.2f}%, {ci['high']:+.2f}%]")
        passes = ci["low"] > 0.0
        print(f"CI excludes zero: {'YES ✓' if passes else 'NO ✗'}")
    else:
        passes = False
        print("Insufficient data for bootstrap CI")

    print(f"\nOOS Sharpe: {overall_sc['sharpe']:.3f}  Sortino: {overall_sc['sortino']:.3f}  "
          f"Max DD: {overall_sc['max_dd_pct']:.1f}%")

    # Save results to registry
    candidate["status"] = "validated"
    candidate["validation"] = {
        "run_at": datetime.now().isoformat(),
        "period": f"{start:%Y-%m-%d} → {train_end:%Y-%m-%d}",
        "folds": len(folds),
        "fold_returns": fold_returns,
        "fold_trades": fold_trades,
        "avg_return_pct": avg_ret,
        "total_trades": total_trades,
        "sharpe": overall_sc["sharpe"],
        "sortino": overall_sc["sortino"],
        "max_dd_pct": overall_sc["max_dd_pct"],
        "bonferroni_alpha": adj_alpha,
        "bonferroni_ci": ci,
        "passes_bonferroni": passes,
    }
    _save_registry(reg)

    if passes:
        print(f"\n→ Candidate #{candidate['id']} PASSES Bonferroni-corrected validation.")
        print(f"  Eligible for hold-out test: python3 -m scripts.validated_backtest holdout --candidate {candidate['id']}")
    else:
        print(f"\n→ Candidate #{candidate['id']} FAILS Bonferroni-corrected validation.")
        print(f"  OOS performance not significantly > 0 at adjusted α={adj_alpha:.4f}")
        print(f"  Do NOT run hold-out. Tune the strategy and register a new candidate.")

    # --- Detailed trade analysis ---
    if all_trade_details:
        _print_trade_analysis(all_trade_details, fold_returns)
        # Persist to JSON for deeper analysis
        analysis_path = REGISTRY_PATH.parent / f"backtest_trades_c{candidate['id']}.json"
        try:
            with open(analysis_path, "w") as f:
                json.dump(all_trade_details, f, default=str)
            print(f"\nTrade details saved to {analysis_path}")
        except Exception as exc:
            logger.warning("Failed to save trade details: %s", exc)

    # --- Per-strategy signal analysis ---
    if all_signal_logs:
        _print_signal_analysis(all_signal_logs, all_trade_details)
        sig_path = REGISTRY_PATH.parent / f"backtest_signals_c{candidate['id']}.json"
        try:
            with open(sig_path, "w") as f:
                json.dump(all_signal_logs, f, default=str)
            print(f"Signal log saved to {sig_path}")
        except Exception as exc:
            logger.warning("Failed to save signal log: %s", exc)


def _print_trade_analysis(trades: list[dict], fold_returns: list[float]) -> None:
    """Print per-symbol and per-fold trade analysis."""
    sells = [t for t in trades if t.get("side") == "sell" and "pnl_pct" in t]
    if not sells:
        print("\nNo completed trades to analyze.")
        return

    print(f"\n{'='*70}")
    print("TRADE ANALYSIS")
    print(f"{'='*70}")

    # Per-fold breakdown
    print(f"\n{'Fold':>4}  {'Buys':>5}  {'Sells':>5}  {'Wins':>4}  {'Losses':>6}  {'WinRate':>7}  {'AvgPnL%':>8}  {'TotalPnL%':>10}")
    print("-" * 70)
    for fi in sorted(set(t["fold"] for t in trades)):
        fold_sells = [t for t in sells if t["fold"] == fi]
        fold_buys = [t for t in trades if t["fold"] == fi and t["side"] == "buy"]
        wins = [t for t in fold_sells if t["pnl_pct"] > 0]
        losses = [t for t in fold_sells if t["pnl_pct"] <= 0]
        wr = len(wins) / len(fold_sells) * 100 if fold_sells else 0
        avg_pnl = sum(t["pnl_pct"] for t in fold_sells) / len(fold_sells) if fold_sells else 0
        total_pnl = sum(t["pnl_pct"] for t in fold_sells)
        print(f"{fi:>4}  {len(fold_buys):>5}  {len(fold_sells):>5}  {len(wins):>4}  {len(losses):>6}  {wr:>6.1f}%  {avg_pnl:>+7.3f}%  {total_pnl:>+9.3f}%")

    # Per-symbol breakdown
    print(f"\n{'Symbol':>12}  {'Sells':>5}  {'Wins':>4}  {'WinRate':>7}  {'AvgPnL%':>8}  {'TotalPnL%':>10}")
    print("-" * 60)
    sym_data = {}
    for t in sells:
        sym_data.setdefault(t["symbol"], []).append(t)
    for sym in sorted(sym_data, key=lambda s: sum(t["pnl_pct"] for t in sym_data[s])):
        st = sym_data[sym]
        wins = [t for t in st if t["pnl_pct"] > 0]
        wr = len(wins) / len(st) * 100 if st else 0
        avg_pnl = sum(t["pnl_pct"] for t in st) / len(st) if st else 0
        total_pnl = sum(t["pnl_pct"] for t in st)
        print(f"{sym:>12}  {len(st):>5}  {len(wins):>4}  {wr:>6.1f}%  {avg_pnl:>+7.3f}%  {total_pnl:>+9.3f}%")

    # Overall summary
    wins = [t for t in sells if t["pnl_pct"] > 0]
    losses = [t for t in sells if t["pnl_pct"] <= 0]
    avg_win = sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0
    print(f"\nOverall: {len(sells)} sells, {len(wins)} wins ({len(wins)/len(sells)*100:.1f}%), {len(losses)} losses")
    print(f"Avg win: {avg_win:+.3f}%  Avg loss: {avg_loss:+.3f}%  Ratio: {abs(avg_win/avg_loss) if avg_loss else 0:.2f}")


def _print_signal_analysis(signals: list[dict], trades: list[dict]) -> None:
    """Print per-strategy signal accuracy vs final execution outcome."""
    if not signals:
        return

    print(f"\n{'='*70}")
    print("STRATEGY SIGNAL ANALYSIS")
    print(f"{'='*70}")

    # Aggregate per-strategy vote counts and agreement with final action
    strat_stats: dict[str, dict] = {}
    for rec in signals:
        final = rec.get("final_action", "hold")
        for sig in rec.get("signals", []):
            name = sig.get("name", "?")
            action = sig.get("action", "hold")
            conf = float(sig.get("confidence", 0) or 0)
            s = strat_stats.setdefault(name, {
                "buy": 0, "sell": 0, "hold": 0,
                "agree_buy": 0, "agree_sell": 0, "total_conf": 0.0, "n_nonhold": 0,
            })
            s[action] = s.get(action, 0) + 1
            if action != "hold":
                s["total_conf"] += conf
                s["n_nonhold"] += 1
            if action == final and action != "hold":
                s[f"agree_{action}"] += 1

    print(f"\n{'Strategy':<25} {'Buy':>5} {'Sell':>5} {'Hold':>6} {'AvgConf':>8} {'BuyAgree':>9} {'SellAgree':>10}")
    print("-" * 75)
    for name in sorted(strat_stats):
        s = strat_stats[name]
        avg_conf = s["total_conf"] / s["n_nonhold"] if s["n_nonhold"] > 0 else 0
        buy_agree = f"{s['agree_buy']}/{s['buy']}" if s["buy"] else "-"
        sell_agree = f"{s['agree_sell']}/{s['sell']}" if s["sell"] else "-"
        print(f"{name:<25} {s['buy']:>5} {s['sell']:>5} {s['hold']:>6} {avg_conf:>7.3f} {buy_agree:>9} {sell_agree:>10}")

    # Signal-to-outcome: when a strategy said "buy", what happened to the trade?
    # Build a lookup: symbol+fold → list of sell P&Ls
    sell_pnl: dict[str, list[float]] = {}
    for t in trades:
        if t.get("side") == "sell" and "pnl_pct" in t:
            key = f"{t['symbol']}_{t.get('fold', 0)}"
            sell_pnl.setdefault(key, []).append(t["pnl_pct"])

    # For each strategy's buy signal, find if the resulting trade was profitable
    print(f"\n{'Strategy':<25} {'BuySignals':>10} {'Executed':>8} {'Profitable':>10} {'AvgPnL':>8}")
    print("-" * 65)
    for name in sorted(strat_stats):
        buy_signals = 0
        executed = 0
        profitable = 0
        pnl_sum = 0.0
        for rec in signals:
            for sig in rec.get("signals", []):
                if sig.get("name") == name and sig.get("action") == "buy":
                    buy_signals += 1
                    if rec.get("final_action") == "buy":
                        key = f"{rec['symbol']}_{rec.get('fold', 0)}"
                        pnls = sell_pnl.get(key, [])
                        if pnls:
                            executed += 1
                            avg_pnl = sum(pnls) / len(pnls)
                            pnl_sum += avg_pnl
                            if avg_pnl > 0:
                                profitable += 1
        avg = pnl_sum / executed if executed > 0 else 0
        print(f"{name:<25} {buy_signals:>10} {executed:>8} {profitable:>10} {avg:>+7.3f}%")

    # Disagreements: when strategies voted differently, which was right?
    disagree_count = 0
    for rec in signals:
        sigs = rec.get("signals", [])
        actions = set(s.get("action") for s in sigs)
        if len(actions) > 1 and "hold" in actions:
            actions.discard("hold")
        if len(actions) > 1:
            disagree_count += 1
    total = len(signals)
    print(f"\nDisagreements: {disagree_count}/{total} bars ({disagree_count/total*100:.1f}%) where strategies voted differently")


def cmd_holdout(cfg: dict, candidate_id: int | None) -> None:
    """Run on the sacred hold-out set. ONE SHOT — results are final."""
    reg = _load_registry()
    active = _active_candidates(reg)

    candidate = None
    if candidate_id:
        candidate = next((c for c in reg["candidates"] if c["id"] == candidate_id), None)
    else:
        # Pick the best validated candidate
        validated = [c for c in active if c.get("validation") and c["validation"].get("passes_bonferroni")]
        if validated:
            candidate = max(validated, key=lambda c: c["validation"]["sharpe"])
        elif active:
            candidate = active[-1]
    if not candidate:
        print(f"Candidate not found.")
        return

    if candidate.get("holdout"):
        print(f"Candidate #{candidate['id']} already has hold-out results!")
        print(f"Hold-out return: {candidate['holdout']['return_pct']:+.2f}%")
        print("Hold-out is a one-shot test. No re-runs allowed.")
        return

    val = candidate.get("validation")
    if not val or not val.get("passes_bonferroni"):
        print(f"WARNING: Candidate #{candidate['id']} did NOT pass Bonferroni validation.")
        print("Running hold-out on a non-validated candidate contaminates the test.")
        print("Proceed anyway? This is not recommended.")
        resp = input("Type 'yes' to proceed: ")
        if resp.strip().lower() != "yes":
            print("Aborted.")
            return

    start, holdout_start, end = _compute_split(cfg)

    print(f"\n{'='*70}")
    print(f"HOLD-OUT TEST — Candidate #{candidate['id']} '{candidate['name']}'")
    print(f"{'='*70}")
    print(f"Hold-out period: {holdout_start:%Y-%m-%d} → {end:%Y-%m-%d} ({(end-holdout_start).days}d)")
    print(f"THIS IS A ONE-SHOT TEST. Results are final.\n")

    holdout_cfg = _apply_updates(cfg, {
        "backtest.start": holdout_start.strftime("%Y-%m-%d"),
        "backtest.end": end.strftime("%Y-%m-%d"),
    })

    try:
        result = run_agent_backtest(holdout_cfg)
        ret_pct = float(result.return_pct) if hasattr(result, "return_pct") else float(result.average_return_pct)
        trades = int(result.trades) if hasattr(result, "trades") else int(result.total_trades)
    except Exception as exc:
        logger.error("Hold-out backtest failed: %s", exc)
        return

    sc = _scorecard([ret_pct])

    print(f"Hold-out return:  {ret_pct:+.2f}%")
    print(f"Hold-out trades:  {trades}")
    print(f"Hold-out Sharpe:  {sc['sharpe']:.3f}")

    # Compare to validation
    if val:
        val_ret = val.get("avg_return_pct", 0)
        decay = val_ret - ret_pct if val_ret else 0
        print(f"\nValidation avg return: {val_ret:+.2f}%")
        print(f"Hold-out return:       {ret_pct:+.2f}%")
        print(f"OOS decay:             {decay:+.2f}%")
        if ret_pct > 0 and (val_ret <= 0 or ret_pct >= val_ret * 0.5):
            print("\n→ PASS: Hold-out confirms strategy has real edge.")
        elif ret_pct > 0:
            print("\n→ MARGINAL: Positive but significant decay from validation.")
        else:
            print("\n→ FAIL: Strategy lost money on unseen data. Likely overfit.")

    candidate["holdout"] = {
        "run_at": datetime.now().isoformat(),
        "period": f"{holdout_start:%Y-%m-%d} → {end:%Y-%m-%d}",
        "return_pct": ret_pct,
        "trades": trades,
        "sharpe": sc["sharpe"],
    }
    _save_registry(reg)


def cmd_status(cfg: dict) -> None:
    """Print status of all candidates."""
    reg = _load_registry()
    active = _active_candidates(reg)
    n_active = len(active)
    adj_alpha = _bonferroni_threshold(reg)

    start, holdout_start, end = _compute_split(cfg)

    print(f"\n{'='*70}")
    print(f"BONFERRONI BACKTEST REGISTRY")
    print(f"{'='*70}")
    print(f"Family α: {ALPHA}   Active candidates: {n_active}   Adjusted α: {adj_alpha:.4f}")
    print(f"Data: {start:%Y-%m-%d} → {end:%Y-%m-%d} ({(end-start).days}d)")
    print(f"Train/Validate: {start:%Y-%m-%d} → {holdout_start:%Y-%m-%d} ({(holdout_start-start).days}d)")
    print(f"Hold-out: {holdout_start:%Y-%m-%d} → {end:%Y-%m-%d} ({(end-holdout_start).days}d)")
    print()

    if not reg["candidates"]:
        print("No candidates registered yet.")
        print("  Register: python3 -m scripts.validated_backtest register --name 'my_config'")
        return

    header = f"{'#':>3}  {'Name':<20}  {'Status':<12}  {'FP':>8}  {'OOS Ret%':>8}  {'Sharpe':>7}  {'Bonf?':>5}  {'HO Ret%':>8}"
    print(header)
    print("-" * len(header))

    for c in reg["candidates"]:
        val = c.get("validation") or {}
        ho = c.get("holdout") or {}
        status = c.get("status", "?")
        oos_ret = f"{val['avg_return_pct']:+.2f}" if val.get("avg_return_pct") is not None else "—"
        sharpe = f"{val['sharpe']:.3f}" if val.get("sharpe") is not None else "—"
        bonf = "YES" if val.get("passes_bonferroni") else ("NO" if val else "—")
        ho_ret = f"{ho['return_pct']:+.2f}" if ho.get("return_pct") is not None else "—"
        fp = c.get("fingerprint", "")[:8]

        print(f"{c['id']:>3}  {c['name']:<20}  {status:<12}  {fp:>8}  {oos_ret:>8}  {sharpe:>7}  {bonf:>5}  {ho_ret:>8}")

    print()
    print("Commands:")
    print("  register  — Register current config as new candidate")
    print("  validate  — Run walk-forward on train period")
    print("  holdout   — Run on sacred hold-out (one shot, requires passing validation)")
    print("  reject    — Mark a candidate as rejected (excluded from Bonferroni count)")


def cmd_reject(candidate_id: int | None) -> None:
    """Mark a candidate as rejected (removes from active count)."""
    reg = _load_registry()
    if not candidate_id:
        print("Specify --candidate N")
        return
    candidate = next((c for c in reg["candidates"] if c["id"] == candidate_id), None)
    if not candidate:
        print(f"Candidate #{candidate_id} not found.")
        return
    candidate["status"] = "rejected"
    _save_registry(reg)
    n_active = len(_active_candidates(reg))
    print(f"Candidate #{candidate_id} rejected. Active: {n_active} → Bonferroni α = {ALPHA}/{max(n_active,1)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Bonferroni-corrected backtesting")
    parser.add_argument("command", choices=["register", "validate", "holdout", "status", "reject"],
                        help="Command to run")
    parser.add_argument("--config", default="config/config.yaml", help="Config file")
    parser.add_argument("--name", help="Candidate name (for register)")
    parser.add_argument("--candidate", type=int, help="Candidate ID")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.command == "register":
        cmd_register(cfg, args.name)
    elif args.command == "validate":
        cmd_validate(cfg, args.candidate)
    elif args.command == "holdout":
        cmd_holdout(cfg, args.candidate)
    elif args.command == "status":
        cmd_status(cfg)
    elif args.command == "reject":
        cmd_reject(args.candidate)


if __name__ == "__main__":
    main()
