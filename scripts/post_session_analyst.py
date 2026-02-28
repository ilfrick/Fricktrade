#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""
Post-session LLM analyst — standalone script.

Reads today's (or a specified) session data from:
  - decision_monitor session report  → strategy signals, skip reasons
  - performance tracker JSON log     → trade PnL by strategy
  - decision trace (sampled)         → orchestrator decisions, risk events

Calls Gemini (or Claude) to produce a structured JSON analysis report
saved to data/session_reports/report_YYYY-MM-DD.json.

Run daily after market close (alongside decision_monitor.py):
  cron: 30 22 * * 1-5    (22:30 CET = 16:30 ET)

Usage:
  python3 scripts/post_session_analyst.py [--date YYYY-MM-DD] [--backend gemini|claude]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("post_session_analyst")

ET = ZoneInfo("US/Eastern")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MONITORING_DIR = _PROJECT_ROOT / "data" / "monitoring"
TRACE_DIR = _PROJECT_ROOT / "data" / "reports" / "decision_trace"
OUTPUT_DIR = _PROJECT_ROOT / "data" / "session_reports"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Session date YYYY-MM-DD (ET). Default: today.")
    parser.add_argument("--backend", default="gemini", choices=["gemini", "claude"])
    args = parser.parse_args()

    et_now = datetime.now(ET)
    session_date = args.date or et_now.strftime("%Y-%m-%d")

    log.info("Post-session analysis for %s using backend=%s", session_date, args.backend)

    # --- Load session report from decision_monitor ---
    monitor_dir = MONITORING_DIR / session_date
    report_path = monitor_dir / "session_report.txt"
    if not report_path.exists():
        log.error("Session report not found: %s", report_path)
        sys.exit(1)

    session_text = report_path.read_text(encoding="utf-8")

    # Parse key metrics from text report
    regime = _extract(session_text, r"(\w+_\w+): \d+", default="unknown")
    buys = int(_extract(session_text, r"Buys:\s+(\d+)", default="0"))
    sells = int(_extract(session_text, r"Sells:\s+(\d+)", default="0"))
    holds = int(_extract(session_text, r"Holds:\s+[\d,]+", default="0").replace(",", ""))
    skips = int(_extract(session_text, r"Skips:\s+[\d,]+", default="0").replace(",", ""))

    # Load strategy signal distribution from snapshot (last JSONL entry)
    strategy_stats = _load_strategy_stats(monitor_dir)

    # Sample orchestrator decisions from trace
    target_dt = datetime.strptime(session_date, "%Y-%m-%d").replace(tzinfo=ET)
    utc_date = target_dt.astimezone(timezone.utc).strftime("%Y-%m-%d")
    trace_path = TRACE_DIR / f"{utc_date}.jsonl"
    orchestrator_decisions, risk_events = _sample_trace(trace_path)

    # Dummy equity (will be filled from performance tracker when available)
    start_equity = 0.0
    end_equity = 0.0
    vix = 20.0
    spy_change = 0.0

    # Build trade list from strategy_stats
    trades = _build_synthetic_trades(strategy_stats)

    # --- Call LLM ---
    _check_env(args.backend)

    from app.llm.client import LLMClient
    from app.llm.post_session import PostSessionAnalyst

    client = LLMClient()
    analyst = PostSessionAnalyst(client, backend=args.backend, output_dir=str(OUTPUT_DIR))

    result = analyst.analyze(
        date=session_date,
        regime=regime,
        vix=vix,
        spy_change=spy_change,
        start_equity=start_equity,
        end_equity=end_equity,
        trades=trades,
        strategy_stats=strategy_stats,
        orchestrator_decisions=orchestrator_decisions,
        risk_events=risk_events,
        fees=0.0,
    )

    if result:
        log.info(
            "Analysis complete: grade=%s, findings=%d, cost=$%.4f",
            result.overall_grade,
            len(result.key_findings),
            result.cost_usd,
        )
        # Print summary to stdout
        print(f"\n=== POST-SESSION ANALYSIS — {session_date} ===")
        print(f"Grade: {result.overall_grade}")
        print(f"Regime analysis: {result.regime_analysis}")
        print("\nKey findings:")
        for f in result.key_findings:
            print(f"  [{f.get('severity', '?').upper()}] {f.get('finding', '')} "
                  f"(${f.get('estimated_pnl_impact', 0):+.2f} est. PnL impact)")
        print("\nTomorrow recommendations:")
        for r in result.tomorrow_recommendations:
            print(f"  - {r}")
        print(f"\nReport saved: {OUTPUT_DIR}/report_{session_date}.json")
        print(f"Cost: ${result.cost_usd:.4f}")
    else:
        log.error("Analysis failed")
        sys.exit(1)


def _extract(text: str, pattern: str, default: str = "") -> str:
    m = re.search(pattern, text)
    return m.group(1) if m else default


def _load_strategy_stats(monitor_dir: Path) -> dict:
    """Load per-strategy signal counts from decision_summary.jsonl (last snapshot)."""
    summary_path = monitor_dir / "decision_summary.jsonl"
    if not summary_path.exists():
        return {}
    last_line = ""
    with summary_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last_line = line.strip()
    if not last_line:
        return {}
    try:
        snap = json.loads(last_line)
        raw = snap.get("strategy_signals", {})
        result = {}
        for name, counts in raw.items():
            buy = counts.get("buy", 0)
            sell = counts.get("sell", 0)
            total = buy + sell
            result[name] = {
                "count": total,
                "buy_signals": buy,
                "sell_signals": sell,
                "win_rate": 50.0,  # placeholder
                "pnl": 0.0,
                "avg_pnl": 0.0,
                "sharpe": 0.0,
            }
        return result
    except Exception:
        return {}


def _sample_trace(trace_path: Path) -> tuple[list[dict], list[dict]]:
    """Sample orchestrator decisions and risk events from trace file."""
    if not trace_path.exists():
        return [], []
    decisions: list[dict] = []
    risk_events: list[dict] = []
    size = trace_path.stat().st_size
    # Sample 1MB near market close (last ~20% of file)
    sample_start = max(0, int(size * 0.8))
    with trace_path.open("rb") as f:
        f.seek(sample_start)
        f.readline()  # skip partial
        for raw in f:
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            if rec.get("decision") == "order_enqueued":
                decisions.append({
                    "timestamp": rec.get("ts", ""),
                    "symbol": rec.get("symbol", ""),
                    "action": rec.get("action", ""),
                    "confidence": rec.get("conviction", 0),
                    "weights": rec.get("effective_weights", {}),
                })
            if rec.get("decision") == "skip" and "risk_block" in str(rec.get("reason", "")):
                detail = rec.get("risk_block_detail", {})
                if detail:
                    risk_events.append({
                        "severity": "high",
                        "timestamp": rec.get("ts", ""),
                        "type": "risk_block",
                        "description": json.dumps(detail),
                    })
            if len(decisions) >= 50 and len(risk_events) >= 20:
                break
    return decisions, risk_events


def _build_synthetic_trades(strategy_stats: dict) -> list[dict]:
    """Build a synthetic trade list from strategy signal counts (no PnL data yet)."""
    trades = []
    for name, stats in strategy_stats.items():
        for i in range(min(stats.get("count", 0), 5)):
            trades.append({
                "id": f"{name}_{i}",
                "timestamp": "",
                "action": "buy",
                "symbol": "?",
                "quantity": 0,
                "price": 0,
                "strategy": name,
                "conviction": 0.5,
                "pnl": 0.0,
            })
    return trades


def _check_env(backend: str) -> None:
    if backend == "claude" and not os.getenv("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)
    if backend == "gemini" and not os.getenv("GOOGLE_GEMINI_API_KEY"):
        log.error("GOOGLE_GEMINI_API_KEY not set")
        sys.exit(1)


if __name__ == "__main__":
    main()
