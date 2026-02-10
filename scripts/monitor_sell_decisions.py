#!/usr/bin/env python3
"""Monitor sell/exit decisions from trader logs and decision traces.

Runs from a start time until US market close, capturing:
- All sell/exit signals (placed, skipped, blocked)
- All hold decisions on symbols with open positions
- Strategy signal distribution (buy/sell/hold/exit counts)
- Skip/block reasons breakdown

Outputs a summary report at the end.

Usage:
    python3 scripts/monitor_sell_decisions.py \
        --start "2026-02-11 15:30" \
        --tz Europe/Rome \
        --output data/monitoring/sell_decisions_report.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


US_EASTERN = ZoneInfo("US/Eastern")
US_CLOSE = (16, 0)  # 4:00 PM ET regular close


def _now_utc() -> datetime:
    return datetime.now(ZoneInfo("UTC"))


def _us_market_close_today(now: datetime) -> datetime:
    et_now = now.astimezone(US_EASTERN)
    close = et_now.replace(hour=US_CLOSE[0], minute=US_CLOSE[1], second=0, microsecond=0)
    if close < et_now:
        close += timedelta(days=1)
    return close.astimezone(ZoneInfo("UTC"))


def _collect_logs(since: str) -> str:
    result = subprocess.run(
        ["docker", "compose", "logs", "trader", "--since", since, "--no-log-prefix"],
        capture_output=True, text=True, timeout=30,
    )
    return result.stdout


def _parse_structured_events(log_text: str) -> list[dict]:
    events = []
    for line in log_text.splitlines():
        if "trading_agent:" not in line and "risk_blocked" not in line:
            continue
        idx = line.find("{")
        if idx < 0:
            continue
        try:
            evt = json.loads(line[idx:])
            # extract timestamp from log prefix
            ts_part = line[:idx].strip().rstrip("INFO").rstrip("WARNING").strip()
            parts = ts_part.split()
            if len(parts) >= 2:
                evt["_log_ts"] = parts[0] + " " + parts[1].rstrip(",")
            events.append(evt)
        except (json.JSONDecodeError, ValueError):
            continue
    return events


def _parse_skip_lines(log_text: str) -> list[dict]:
    skips = []
    for line in log_text.splitlines():
        if "Skipping " not in line:
            continue
        try:
            parts = line.split("Skipping ", 1)[1]
            action_rest = parts.split(" for ", 1)
            if len(action_rest) < 2:
                continue
            action = action_rest[0].strip()
            sym_reason = action_rest[1].split(": ", 1)
            symbol = sym_reason[0].strip()
            reason = sym_reason[1].strip() if len(sym_reason) > 1 else "unknown"
            ts_part = line.split(" INFO ")[0].strip() if " INFO " in line else ""
            skips.append({
                "action": action, "symbol": symbol, "reason": reason,
                "_log_ts": ts_part,
            })
        except Exception:
            continue
    return skips


def _load_traces(trace_dir: Path, date_str: str) -> list[dict]:
    path = trace_dir / f"{date_str}.jsonl"
    if not path.exists():
        return []
    traces = []
    for line in path.read_text().splitlines():
        try:
            traces.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return traces


def _analyze(
    events: list[dict],
    skips: list[dict],
    traces: list[dict],
    start_utc: datetime,
) -> dict:
    start_iso = start_utc.isoformat()

    # Filter traces to our window
    window_traces = [
        t for t in traces
        if t.get("ts", "") >= start_iso
    ]

    # Signal action distribution from traces
    signal_actions: Counter = Counter()
    outcome_actions: Counter = Counter()
    sell_traces = []
    hold_with_position = []
    rl_value_timeline: list[dict] = []  # value estimates over time

    for t in window_traces:
        outcome = t.get("outcome", "unknown")
        outcome_actions[outcome] += 1

        signals = t.get("signals", [])
        for s in signals:
            sig_action = s.get("action", "unknown")
            signal_actions[sig_action] += 1

        # Collect RL value estimates and action probabilities
        for s in signals:
            if s.get("value_estimate") is not None or s.get("action_probs"):
                rl_value_timeline.append({
                    "symbol": t.get("symbol"),
                    "ts": t.get("ts"),
                    "strategy": s.get("name"),
                    "action": s.get("action"),
                    "value_estimate": s.get("value_estimate"),
                    "action_probs": s.get("action_probs"),
                    "broker": t.get("broker_hint"),
                })

        # Collect sell/exit outcomes
        action_field = None
        for s in signals:
            action_field = s.get("action")
        final_action = t.get("final_action") or action_field

        if final_action in ("sell", "exit"):
            sell_traces.append({
                "symbol": t.get("symbol"),
                "ts": t.get("ts"),
                "outcome": outcome,
                "reason": t.get("reason"),
                "stage": t.get("stage"),
                "broker": t.get("broker_hint"),
                "signals": [
                    {k: v for k, v in s.items() if k != "features"}
                    for s in signals
                ],
            })

        # Hold decisions where a position exists
        portfolio = t.get("portfolio", {})
        positions = portfolio.get("positions", {}) if isinstance(portfolio, dict) else {}
        sym = t.get("symbol", "")
        if sym in positions and float(positions[sym].get("qty", 0) or 0) > 0:
            if final_action in ("hold", None):
                hold_with_position.append({
                    "symbol": sym,
                    "ts": t.get("ts"),
                    "qty": positions[sym].get("qty"),
                    "signals": [
                        {k: v for k, v in s.items() if k != "features"}
                        for s in signals
                    ],
                    "outcome": outcome,
                })

    # Skip/block reasons
    sell_skip_reasons: Counter = Counter()
    buy_skip_reasons: Counter = Counter()
    for s in skips:
        if s.get("action") == "sell":
            sell_skip_reasons[s.get("reason", "unknown")] += 1
        elif s.get("action") == "buy":
            buy_skip_reasons[s.get("reason", "unknown")] += 1

    # Structured event counts
    event_types: Counter = Counter()
    risk_reasons: Counter = Counter()
    for e in events:
        event_types[e.get("event", "unknown")] += 1
        if e.get("event") == "risk_blocked":
            key = f"{e.get('action', '?')}:{e.get('reason', '?')}"
            risk_reasons[key] += 1

    # RL value estimate statistics
    rl_values = [r["value_estimate"] for r in rl_value_timeline if r.get("value_estimate") is not None]
    rl_sell_probs = [
        r["action_probs"]["sell"] for r in rl_value_timeline
        if r.get("action_probs") and "sell" in r["action_probs"]
    ]
    rl_buy_probs = [
        r["action_probs"]["buy"] for r in rl_value_timeline
        if r.get("action_probs") and "buy" in r["action_probs"]
    ]

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {}
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        return {
            "count": n,
            "min": vals_sorted[0],
            "max": vals_sorted[-1],
            "mean": sum(vals_sorted) / n,
            "median": vals_sorted[n // 2],
            "p10": vals_sorted[max(0, n // 10)],
            "p90": vals_sorted[min(n - 1, n * 9 // 10)],
        }

    return {
        "window_start": start_utc.isoformat(),
        "window_end": _now_utc().isoformat(),
        "total_traces_in_window": len(window_traces),
        "signal_action_distribution": dict(signal_actions.most_common()),
        "outcome_distribution": dict(outcome_actions.most_common()),
        "sell_exit_traces": sell_traces[:50],
        "sell_exit_count": len(sell_traces),
        "hold_with_position": hold_with_position[:50],
        "hold_with_position_count": len(hold_with_position),
        "sell_skip_reasons": dict(sell_skip_reasons.most_common()),
        "buy_skip_reasons": dict(buy_skip_reasons.most_common()),
        "structured_event_types": dict(event_types.most_common()),
        "risk_block_reasons": dict(risk_reasons.most_common(20)),
        "rl_value_estimate_stats": _stats(rl_values),
        "rl_sell_prob_stats": _stats(rl_sell_probs),
        "rl_buy_prob_stats": _stats(rl_buy_probs),
        "rl_value_timeline_sample": rl_value_timeline[:100],
        "rl_value_timeline_count": len(rl_value_timeline),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor sell/exit decisions")
    parser.add_argument("--start", required=True, help="Start time, e.g. '2026-02-11 15:30'")
    parser.add_argument("--tz", default="Europe/Rome", help="Timezone for --start")
    parser.add_argument("--output", default="data/monitoring/sell_decisions_report.json")
    parser.add_argument("--trace-dir", default="data/reports/decision_trace")
    parser.add_argument("--interval", type=int, default=300, help="Poll interval seconds")
    args = parser.parse_args()

    local_tz = ZoneInfo(args.tz)
    start_local = datetime.strptime(args.start, "%Y-%m-%d %H:%M").replace(tzinfo=local_tz)
    start_utc = start_local.astimezone(ZoneInfo("UTC"))
    output_path = Path(args.output)
    trace_dir = Path(args.trace_dir)

    print(f"Sell decision monitor")
    print(f"  Start:    {start_local.isoformat()}")
    print(f"  Start (UTC): {start_utc.isoformat()}")
    print(f"  Output:   {output_path}")
    print(f"  Interval: {args.interval}s")

    # Wait until start time
    now = _now_utc()
    if now < start_utc:
        wait = (start_utc - now).total_seconds()
        print(f"  Waiting {wait:.0f}s until start time...")
        time.sleep(wait)

    market_close = _us_market_close_today(_now_utc())
    print(f"  US close: {market_close.astimezone(local_tz).isoformat()}")

    all_events = []
    all_skips = []
    cycle = 0

    while _now_utc() < market_close:
        cycle += 1
        now = _now_utc()
        elapsed = (now - start_utc).total_seconds()
        since_arg = f"{int(elapsed)}s"

        try:
            log_text = _collect_logs(since_arg)
            events = _parse_structured_events(log_text)
            skips = _parse_skip_lines(log_text)

            date_str = now.strftime("%Y-%m-%d")
            traces = _load_traces(trace_dir, date_str)

            report = _analyze(events, skips, traces, start_utc)
            report["cycle"] = cycle
            report["market_close_utc"] = market_close.isoformat()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2, default=str))

            sell_ct = report["sell_exit_count"]
            hold_pos_ct = report["hold_with_position_count"]
            total = report["total_traces_in_window"]
            sig_dist = report["signal_action_distribution"]
            print(
                f"  [{now.astimezone(local_tz).strftime('%H:%M:%S')}] "
                f"cycle={cycle} traces={total} "
                f"sell/exit={sell_ct} hold_w_pos={hold_pos_ct} "
                f"signals={dict(sig_dist)}"
            )

        except Exception as exc:
            print(f"  [{_now_utc().isoformat()}] Error: {exc}", file=sys.stderr)

        time.sleep(args.interval)

    # Final collection
    print("\nMarket closed. Final collection...")
    try:
        elapsed = (_now_utc() - start_utc).total_seconds()
        log_text = _collect_logs(f"{int(elapsed)}s")
        events = _parse_structured_events(log_text)
        skips = _parse_skip_lines(log_text)
        date_str = _now_utc().strftime("%Y-%m-%d")
        traces = _load_traces(trace_dir, date_str)
        report = _analyze(events, skips, traces, start_utc)
        report["cycle"] = "final"
        report["market_close_utc"] = market_close.isoformat()
        output_path.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nFinal report written to {output_path}")
        print(f"  Total traces: {report['total_traces_in_window']}")
        print(f"  Signal distribution: {report['signal_action_distribution']}")
        print(f"  Outcome distribution: {report['outcome_distribution']}")
        print(f"  Sell/exit decisions: {report['sell_exit_count']}")
        print(f"  Hold with position: {report['hold_with_position_count']}")
        print(f"  Sell skip reasons: {report['sell_skip_reasons']}")
        print(f"  Risk block reasons: {report['risk_block_reasons']}")
    except Exception as exc:
        print(f"Final collection error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
