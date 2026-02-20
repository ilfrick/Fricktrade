#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required (pip install pyyaml).") from exc


@dataclass
class SymbolDiagnostics:
    rows: int = 0
    decisions: Counter = field(default_factory=Counter)
    actions: Counter = field(default_factory=Counter)
    reasons: Counter = field(default_factory=Counter)
    stages: Counter = field(default_factory=Counter)
    action_strategies: Counter = field(default_factory=Counter)
    signal_actions: Counter = field(default_factory=Counter)
    signal_name_actions: Counter = field(default_factory=Counter)
    bias_blocks: Counter = field(default_factory=Counter)
    buy_signal_rows: int = 0
    sell_signal_rows: int = 0
    buy_only_rows: int = 0
    sell_only_rows: int = 0
    conflicting_rows: int = 0
    under_conviction_rows: int = 0


def _parse_args():
    p = argparse.ArgumentParser(description="Explain why daily top movers were not traded.")
    p.add_argument("--date", required=True, help="Date in YYYY-MM-DD.")
    p.add_argument("--repo-root", default="/home/nicola/Fricktrade", help="Repo root path.")
    p.add_argument(
        "--decision-trace",
        default="",
        help="Decision trace JSONL path. Defaults to data/reports/decision_trace/<date>.jsonl",
    )
    p.add_argument(
        "--report-dir",
        default="",
        help="Daily top movers report dir. Defaults to data/reports/daily_top_movers/<date>/alpaca",
    )
    p.add_argument(
        "--decision-summary",
        default="",
        help="Optional monitoring decision summary JSONL. Defaults to data/monitoring/<date>/decision_summary.jsonl",
    )
    p.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to data/reports/modeling",
    )
    p.add_argument(
        "--allow-shorts",
        choices=["true", "false"],
        default="",
        help="Optional override for allow_shorts when diagnosing historical runs.",
    )
    p.add_argument(
        "--min-conviction",
        type=float,
        default=None,
        help="Optional override for strategy.min_conviction when diagnosing historical runs.",
    )
    p.add_argument(
        "--single-sided-conviction-multiplier",
        type=float,
        default=None,
        help="Optional override for strategy.single_sided_conviction_multiplier (0-1).",
    )
    return p.parse_args()


def _load_config(repo_root: Path) -> dict:
    cfg_path = repo_root / "config" / "config.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def _top_symbols(report_dir: Path) -> tuple[list[str], list[tuple[str, str]]]:
    entries: list[tuple[str, str]] = []
    for csv_path in sorted(report_dir.rglob("*_1m.csv")):
        symbol = csv_path.name.split("_1m.csv")[0]
        venue = csv_path.parent.name
        entries.append((symbol, venue))
    unique = sorted({s for s, _ in entries})
    return unique, entries


def _score_sides(signals: list[dict], weights: dict[str, float]) -> tuple[float, float]:
    buy = 0.0
    sell = 0.0
    for sig in signals:
        name = str(sig.get("name", ""))
        action = str(sig.get("action", "hold")).lower()
        conf = float(sig.get("confidence", 1.0) or 1.0)
        w = float(weights.get(name, 1.0))
        score = conf * w
        if action == "buy":
            buy += score
        elif action == "sell":
            sell += score
    return buy, sell


def _load_latest_decision_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    last = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last or {}


def _classify_root_cause(
    d: SymbolDiagnostics,
    allow_shorts: bool,
) -> str:
    if d.rows == 0:
        return "not_in_decision_loop"
    non_hold_actions = sum(v for k, v in d.actions.items() if str(k).lower() not in {"", "hold"})
    if non_hold_actions > 0:
        return "has_directional_actions"
    if d.buy_only_rows > 0 and d.under_conviction_rows >= d.buy_only_rows:
        return "buy_under_min_conviction"
    if d.sell_only_rows > 0 and not allow_shorts:
        return "sell_signals_filtered_no_short"
    if d.conflicting_rows > 0:
        return "conflicting_signals_net_hold"
    if sum(d.bias_blocks.values()) > 0:
        return "bias_guard_contributed"
    if d.buy_signal_rows == 0 and d.sell_signal_rows == 0:
        return "all_strategies_hold"
    return "strategies_hold_other"


def main() -> None:
    args = _parse_args()
    run_date = date.fromisoformat(args.date)
    repo_root = Path(args.repo_root)

    trace_path = (
        Path(args.decision_trace)
        if args.decision_trace
        else repo_root / "data" / "reports" / "decision_trace" / f"{run_date.isoformat()}.jsonl"
    )
    report_dir = (
        Path(args.report_dir)
        if args.report_dir
        else repo_root / "data" / "reports" / "daily_top_movers" / run_date.isoformat() / "alpaca"
    )
    summary_path = (
        Path(args.decision_summary)
        if args.decision_summary
        else repo_root / "data" / "monitoring" / run_date.isoformat() / "decision_summary.jsonl"
    )
    output_dir = Path(args.output_dir) if args.output_dir else (repo_root / "data" / "reports" / "modeling")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(repo_root)
    allow_shorts = bool(cfg.get("trading_limits", {}).get("allow_shorts", False))
    if args.allow_shorts:
        allow_shorts = args.allow_shorts.lower() == "true"
    min_conviction = float(cfg.get("strategy", {}).get("min_conviction", 0.0) or 0.0)
    if args.min_conviction is not None:
        min_conviction = float(args.min_conviction)
    single_side_mult = float(cfg.get("strategy", {}).get("single_sided_conviction_multiplier", 1.0) or 1.0)
    if args.single_sided_conviction_multiplier is not None:
        single_side_mult = float(args.single_sided_conviction_multiplier)
    single_side_mult = min(max(single_side_mult, 0.0), 1.0)

    symbols, entries = _top_symbols(report_dir)
    if not symbols:
        raise SystemExit(f"No *_1m.csv top-mover files found in {report_dir}")

    diag: dict[str, SymbolDiagnostics] = {s: SymbolDiagnostics() for s in symbols}

    if not trace_path.exists():
        raise SystemExit(f"Decision trace file not found: {trace_path}")

    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sym = rec.get("symbol")
            if sym not in diag:
                continue
            d = diag[sym]
            d.rows += 1
            decision = str(rec.get("decision", ""))
            action = str(rec.get("action", decision or ""))
            d.decisions[decision] += 1
            d.actions[action] += 1
            d.reasons[str(rec.get("reason", ""))] += 1
            d.stages[str(rec.get("stage", ""))] += 1
            d.action_strategies[str(rec.get("action_strategy", ""))] += 1

            signals = rec.get("signals") or []
            weights = rec.get("effective_weights")
            if not isinstance(weights, dict):
                weights = rec.get("orchestrator_weights")
            if not isinstance(weights, dict):
                weights = {}

            buy_score, sell_score = _score_sides(signals, weights)
            if buy_score > 0:
                d.buy_signal_rows += 1
            if sell_score > 0:
                d.sell_signal_rows += 1
            if buy_score > 0 and sell_score == 0:
                d.buy_only_rows += 1
            elif sell_score > 0 and buy_score == 0:
                d.sell_only_rows += 1
            elif sell_score > 0 and buy_score > 0:
                d.conflicting_rows += 1

            winning = max(buy_score, sell_score)
            eff_min_conv = min_conviction
            if min_conviction > 0 and ((buy_score > 0 and sell_score == 0) or (sell_score > 0 and buy_score == 0)):
                eff_min_conv = min_conviction * single_side_mult
            if winning > 0 and eff_min_conv > 0 and winning < eff_min_conv:
                d.under_conviction_rows += 1

            for sig in signals:
                name = str(sig.get("name", ""))
                sig_action = str(sig.get("action", ""))
                d.signal_actions[sig_action] += 1
                d.signal_name_actions[(name, sig_action)] += 1
                block = sig.get("signal_bias_block")
                if block:
                    d.bias_blocks[str(block)] += 1

    seen = sorted([s for s, d in diag.items() if d.rows > 0])
    missing = sorted([s for s, d in diag.items() if d.rows == 0])
    latest_summary = _load_latest_decision_summary(summary_path)

    rows = []
    for sym in symbols:
        d = diag[sym]
        root_cause = _classify_root_cause(d, allow_shorts=allow_shorts)
        top_reason = d.reasons.most_common(1)[0][0] if d.reasons else ""
        top_action_strategy = d.action_strategies.most_common(1)[0][0] if d.action_strategies else ""
        rows.append(
            {
                "symbol": sym,
                "rows": d.rows,
                "in_decision_loop": int(d.rows > 0),
                "final_hold_pct": (
                    float(d.actions.get("hold", 0)) / float(max(d.rows, 1)) * 100.0
                ),
                "buy_signal_rows": d.buy_signal_rows,
                "sell_signal_rows": d.sell_signal_rows,
                "buy_only_rows": d.buy_only_rows,
                "sell_only_rows": d.sell_only_rows,
                "conflicting_rows": d.conflicting_rows,
                "under_conviction_rows": d.under_conviction_rows,
                "bias_block_rows": int(sum(d.bias_blocks.values())),
                "top_reason": top_reason,
                "top_action_strategy": top_action_strategy,
                "root_cause": root_cause,
            }
        )

    csv_path = output_dir / f"top_movers_trade_gap_{run_date.isoformat()}.csv"
    json_path = output_dir / f"top_movers_trade_gap_{run_date.isoformat()}.json"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "date": run_date.isoformat(),
        "paths": {
            "decision_trace": str(trace_path),
            "report_dir": str(report_dir),
            "decision_summary": str(summary_path),
            "output_csv": str(csv_path),
        },
        "config_snapshot": {
            "allow_shorts": allow_shorts,
            "strategy_min_conviction": min_conviction,
            "single_sided_conviction_multiplier": single_side_mult,
        },
        "top_mover_entries": [{"symbol": s, "venue": v} for s, v in entries],
        "top_mover_unique_symbols": symbols,
        "symbols_seen_in_trace": seen,
        "symbols_missing_in_trace": missing,
        "rows_by_symbol": rows,
        "global_decision_summary_latest": latest_summary,
        "aggregate": {
            "n_top_unique_symbols": len(symbols),
            "n_seen": len(seen),
            "n_missing": len(missing),
            "n_hold_only_symbols": int(
                sum(
                    1
                    for s in symbols
                    if diag[s].rows > 0 and sum(v for k, v in diag[s].actions.items() if str(k).lower() not in {"", "hold"}) == 0
                )
            ),
            "root_cause_counts": dict(Counter(r["root_cause"] for r in rows)),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Top mover symbols: {len(symbols)}")
    print(f"Seen in decision trace: {len(seen)}")
    print(f"Missing from decision trace: {len(missing)}")
    print(f"CSV report: {csv_path}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()
