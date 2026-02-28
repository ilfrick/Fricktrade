#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""End-of-day session analysis.

Reads the monitoring output collected during the trading session and produces
a structured markdown report covering:
  - Decision summary (counts, rates)
  - Strategy signal distribution and win/hold/sell breakdown
  - Skip reasons ranked by frequency
  - Confidence → outcome correlations
  - Regime distribution and stability
  - Latency profile
  - Memory trend
  - Anomalies and improvement suggestions

Run after market close (22:15 CET = 16:15 ET).

Usage:
  python3 scripts/analyze_session.py [--date YYYY-MM-DD] [--monitoring-dir data/monitoring]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("analyze_session")

ET = ZoneInfo("US/Eastern")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Session date YYYY-MM-DD (ET)")
    _project_root = str(Path(__file__).resolve().parent.parent)
    parser.add_argument("--monitoring-dir", default=f"{_project_root}/data/monitoring")
    parser.add_argument("--trace-dir", default=f"{_project_root}/data/reports/decision_trace")
    args = parser.parse_args()

    date_str = args.date or datetime.now(ET).strftime("%Y-%m-%d")
    mon_dir = Path(args.monitoring_dir) / date_str

    if not mon_dir.exists():
        log.error("Monitoring directory not found: %s", mon_dir)
        sys.exit(1)

    log.info("Analyzing session: %s", date_str)

    # --- Load data ---
    decisions = _load_jsonl(mon_dir / "decisions_full.jsonl")
    signals = _load_jsonl(mon_dir / "strategy_signals.jsonl")
    risk_blocks = _load_jsonl(mon_dir / "risk_blocks.jsonl")
    pnl_rows = _load_csv(mon_dir / "pnl_snapshots.csv")
    metrics_files = sorted((mon_dir / "metrics").glob("*.txt")) if (mon_dir / "metrics").exists() else []

    # If decisions_full not available yet, fall back to raw trace
    if not decisions:
        trace_path = Path(args.trace_dir) / f"{date_str}.jsonl"
        if trace_path.exists():
            log.info("decisions_full.jsonl empty — falling back to raw trace")
            decisions = _load_jsonl_filtered(trace_path, decision_types={"order_enqueued", "skip", "exit"})

    log.info("Loaded: %d decisions, %d signals, %d risk blocks, %d pnl rows",
             len(decisions), len(signals), len(risk_blocks), len(pnl_rows))

    # --- Analyse ---
    report_lines: list[str] = []
    _header(report_lines, date_str)
    _section_decisions(report_lines, decisions)
    _section_strategies(report_lines, signals, decisions)
    _section_skip_reasons(report_lines, risk_blocks)
    _section_confidence_correlation(report_lines, decisions)
    _section_regime(report_lines, decisions)
    _section_leverage_deep_dive(report_lines, risk_blocks, decisions)
    _section_latency(report_lines, decisions)
    _section_pnl(report_lines, pnl_rows)
    _section_memory(report_lines, metrics_files)
    _section_anomalies(report_lines, decisions, signals, risk_blocks, pnl_rows)
    _section_improvements(report_lines, decisions, signals, risk_blocks)

    report_path = mon_dir / "analysis_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    log.info("Report written to %s", report_path)

    # Also print to stdout
    print("\n".join(report_lines))


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _header(lines: list, date_str: str) -> None:
    lines += [
        f"# Session Analysis — {date_str}",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
    ]


def _section_decisions(lines: list, decisions: list[dict]) -> None:
    lines.append("## Decision Summary")
    if not decisions:
        lines += ["_No decision data available._", ""]
        return

    total = len(decisions)
    by_decision: Counter = Counter()
    by_action: Counter = Counter()
    for d in decisions:
        by_decision[d.get("decision", "unknown")] += 1
        by_action[d.get("action", "unknown")] += 1

    buys = by_action["buy"]
    sells = by_action.get("sell", 0) + by_decision.get("exit", 0)
    skips = by_decision["skip"]

    lines += [
        f"| Metric | Count | Rate |",
        f"|--------|-------|------|",
        f"| Total records | {total:,} | — |",
        f"| Buys executed | {buys:,} | {buys/total*100:.2f}% |",
        f"| Sells / exits | {sells:,} | {sells/total*100:.2f}% |",
        f"| Skips | {skips:,} | {skips/total*100:.2f}% |",
        "",
    ]

    # Top 15 most-active symbols
    sym_counts: Counter = Counter()
    for d in decisions:
        if d.get("symbol"):
            sym_counts[d["symbol"]] += 1
    lines.append("**Top 15 most-active symbols:**")
    lines.append("")
    lines.append("| Symbol | Decisions |")
    lines.append("|--------|-----------|")
    for sym, cnt in sym_counts.most_common(15):
        lines.append(f"| {sym} | {cnt:,} |")
    lines.append("")


def _section_strategies(lines: list, signals: list[dict], decisions: list[dict]) -> None:
    lines.append("## Strategy Signal Distribution")

    # From strategy_signals.jsonl (trade executions)
    if signals:
        strat_buy: Counter = Counter()
        strat_sell: Counter = Counter()
        for s in signals:
            strat = s.get("action_strategy", "unknown")
            action = s.get("action", "unknown")
            if action == "buy":
                strat_buy[strat] += 1
            elif action == "sell":
                strat_sell[strat] += 1

        all_strats = sorted(set(strat_buy) | set(strat_sell))
        if all_strats:
            lines.append("")
            lines.append("**Strategies driving executed trades:**")
            lines.append("")
            lines.append("| Strategy | Buys | Sells |")
            lines.append("|----------|------|-------|")
            for s in all_strats:
                lines.append(f"| {s} | {strat_buy[s]} | {strat_sell[s]} |")
            lines.append("")

    # From decisions, individual strategy signal votes
    sig_buy: Counter = Counter()
    sig_sell: Counter = Counter()
    sig_hold: Counter = Counter()
    for d in decisions:
        for sig in (d.get("signals") or []):
            name = sig.get("name", sig.get("strategy", "?"))
            action = str(sig.get("action", "hold")).lower()
            if action == "buy":
                sig_buy[name] += 1
            elif action == "sell":
                sig_sell[name] += 1
            else:
                sig_hold[name] += 1

    if sig_buy or sig_sell or sig_hold:
        all_names = sorted(set(sig_buy) | set(sig_sell) | set(sig_hold))
        lines.append("**Per-strategy vote distribution (all decisions):**")
        lines.append("")
        lines.append("| Strategy | Buy | Sell | Hold | Buy% |")
        lines.append("|----------|-----|------|------|------|")
        for name in all_names:
            b, s, h = sig_buy[name], sig_sell[name], sig_hold[name]
            tot = b + s + h
            pct = f"{b/tot*100:.1f}%" if tot else "—"
            lines.append(f"| {name} | {b:,} | {s:,} | {h:,} | {pct} |")
        lines.append("")


def _section_skip_reasons(lines: list, risk_blocks: list[dict]) -> None:
    lines.append("## Skip / Risk Block Reasons")
    if not risk_blocks:
        lines += ["_No skip data available._", ""]
        return

    reasons: Counter = Counter()
    stage_reasons: Counter = Counter()
    for r in risk_blocks:
        reason = r.get("reason", "unknown")
        stage = r.get("stage", "")
        reasons[reason] += 1
        stage_reasons[f"{stage}:{reason}" if stage else reason] += 1

    lines.append(f"Total skips: **{len(risk_blocks):,}**")
    lines.append("")
    lines.append("| Reason | Count | % |")
    lines.append("|--------|-------|---|")
    total_skips = len(risk_blocks)
    for reason, cnt in reasons.most_common(20):
        lines.append(f"| {reason} | {cnt:,} | {cnt/total_skips*100:.1f}% |")
    lines.append("")


def _section_confidence_correlation(lines: list, decisions: list[dict]) -> None:
    lines.append("## Confidence → Execution Correlation")

    buckets: dict[str, list[float]] = defaultdict(list)
    for d in decisions:
        conf = d.get("confidence")
        # Fallback: extract max confidence from individual strategy signals
        if conf is None:
            sigs = d.get("signals") or []
            for sig in sigs:
                c = sig.get("confidence")
                if c is not None:
                    try:
                        c = float(c)
                        conf = max(conf or 0.0, c)
                    except (TypeError, ValueError):
                        pass
        action = d.get("action", "")
        if conf is None:
            continue
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            continue
        # Bucket by 0.1 increments
        bucket = f"{int(conf * 10) / 10:.1f}"
        buckets[bucket].append(1.0 if action == "buy" else 0.0)

    if not buckets:
        lines += ["_No confidence data._", ""]
        return

    lines.append("")
    lines.append("| Confidence bucket | Records | Buy rate |")
    lines.append("|------------------|---------|----------|")
    for bucket in sorted(buckets):
        vals = buckets[bucket]
        buy_rate = sum(vals) / len(vals) if vals else 0.0
        lines.append(f"| {bucket} | {len(vals):,} | {buy_rate*100:.1f}% |")
    lines.append("")
    lines.append(
        "_Expected: buy rate should increase monotonically with confidence bucket._"
    )
    lines.append("")


def _section_regime(lines: list, decisions: list[dict]) -> None:
    lines.append("## Regime Distribution")

    regime_counts: Counter = Counter()
    probs: list[float] = []
    for d in decisions:
        r = d.get("regime_name")
        if r:
            regime_counts[r] += 1
        p = d.get("regime_probability")
        if p is not None:
            try:
                probs.append(float(p))
            except (TypeError, ValueError):
                pass

    if not regime_counts:
        lines += ["_No regime data._", ""]
        return

    total = sum(regime_counts.values())
    lines.append("")
    lines.append("| Regime | Count | % |")
    lines.append("|--------|-------|---|")
    for regime, cnt in regime_counts.most_common():
        lines.append(f"| {regime} | {cnt:,} | {cnt/total*100:.1f}% |")
    lines.append("")

    if probs:
        mean_p = sum(probs) / len(probs)
        lines.append(f"Mean regime probability: **{mean_p:.4f}**")
        if mean_p > 0.95:
            lines.append("⚠️ **HMM may be stuck** — probability persistently near 1.0")
        lines.append("")


def _section_pnl(lines: list, pnl_rows: list[dict]) -> None:
    lines.append("## PnL Trend")
    if not pnl_rows:
        lines += ["_No PnL snapshots._", ""]
        return

    lines.append("")
    lines.append(f"Snapshots collected: **{len(pnl_rows)}**")
    if pnl_rows:
        first = pnl_rows[0]
        last = pnl_rows[-1]
        lines.append("")
        lines.append("| Metric | Open | Close |")
        lines.append("|--------|------|-------|")
        for key in ("pnl_pct", "drawdown_pct", "equity", "cash", "leverage", "gross_exposure", "open_positions", "trades_total", "rss_mb"):
            lines.append(f"| {key} | {first.get(key, '—')} | {last.get(key, '—')} |")
    lines.append("")


def _section_memory(lines: list, metrics_files: list[Path]) -> None:
    lines.append("## Memory Trend")
    if not metrics_files:
        lines += ["_No metrics snapshots._", ""]
        return

    import re
    rss_pattern = re.compile(r"^process_resident_memory_bytes\s+([\d.e+]+)", re.MULTILINE)
    snapshots: list[float] = []
    for fpath in metrics_files:
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
            m = rss_pattern.search(content)
            if m:
                snapshots.append(float(m.group(1)) / (1024 * 1024))
        except Exception:
            pass

    if not snapshots:
        lines += ["_No RSS data in metrics._", ""]
        return

    growth = (snapshots[-1] - snapshots[0]) / max(len(snapshots) - 1, 1)
    lines += [
        f"- Start: **{snapshots[0]:.0f} MB**",
        f"- End: **{snapshots[-1]:.0f} MB**",
        f"- Peak: **{max(snapshots):.0f} MB**",
        f"- Growth rate: **{growth:+.1f} MB/min**",
    ]
    if growth > 5:
        lines.append(f"⚠️ **ALERT: growth {growth:.1f} MB/min — possible memory leak**")
    if max(snapshots) > 4000:
        lines.append(f"⚠️ **ALERT: peak {max(snapshots):.0f} MB — OOM risk**")
    lines.append("")


def _section_leverage_deep_dive(lines: list, risk_blocks: list[dict], decisions: list[dict]) -> None:
    """Investigate pending_leverage_cap — the dominant skip reason from Feb 23."""
    lines.append("## Leverage Cap Deep Dive")

    lev_blocks = [r for r in risk_blocks if r.get("reason") == "pending_leverage_cap"]
    if not lev_blocks:
        lines += ["_No `pending_leverage_cap` skips this session._", ""]
        return

    total_skips = len(risk_blocks)
    n = len(lev_blocks)
    lines.append(f"`pending_leverage_cap` blocked **{n:,}** of **{total_skips:,}** skips ({n/total_skips*100:.1f}%)")
    lines.append("")

    # Time distribution — are they clustered or spread throughout the session?
    hourly: Counter = Counter()
    for r in lev_blocks:
        ts_str = r.get("ts", "")
        try:
            ts = datetime.fromisoformat(ts_str)
            hourly[ts.strftime("%H:00")] += 1
        except (TypeError, ValueError):
            pass
    if hourly:
        lines.append("**Hourly distribution:**")
        lines.append("")
        lines.append("| Hour (UTC) | Blocks |")
        lines.append("|------------|--------|")
        for hour in sorted(hourly):
            lines.append(f"| {hour} | {hourly[hour]:,} |")
        lines.append("")

    # Which symbols are most affected?
    sym_blocks: Counter = Counter(r.get("symbol", "?") for r in lev_blocks)
    lines.append("**Top 10 symbols blocked by leverage cap:**")
    lines.append("")
    lines.append("| Symbol | Blocks |")
    lines.append("|--------|--------|")
    for sym, cnt in sym_blocks.most_common(10):
        lines.append(f"| {sym} | {cnt:,} |")
    lines.append("")

    # How many unique symbols wanted to buy but were blocked?
    blocked_syms = set(r.get("symbol") for r in lev_blocks if r.get("symbol"))
    bought_syms = set(d.get("symbol") for d in decisions if d.get("action") == "buy")
    only_blocked = blocked_syms - bought_syms
    lines.append(f"- Unique symbols blocked: **{len(blocked_syms)}**")
    lines.append(f"- Of those, never got to buy: **{len(only_blocked)}**")
    lines.append(f"- Symbols that were both blocked and bought: **{len(blocked_syms & bought_syms)}**")
    lines.append("")


def _section_latency(lines: list, decisions: list[dict]) -> None:
    lines.append("## Decision Latency")

    latencies = []
    for d in decisions:
        lat = d.get("decision_latency_seconds")
        if lat is not None:
            try:
                latencies.append(float(lat))
            except (TypeError, ValueError):
                pass

    if not latencies:
        lines += ["_No latency data available._", ""]
        return

    latencies.sort()
    n = len(latencies)
    p50 = latencies[int(n * 0.50)]
    p90 = latencies[int(n * 0.90)]
    p95 = latencies[int(n * 0.95)]
    p99 = latencies[min(int(n * 0.99), n - 1)]
    mean = sum(latencies) / n

    lines += [
        "",
        f"| Percentile | Seconds |",
        f"|------------|---------|",
        f"| Mean | {mean:.3f} |",
        f"| p50 | {p50:.3f} |",
        f"| p90 | {p90:.3f} |",
        f"| p95 | {p95:.3f} |",
        f"| p99 | {p99:.3f} |",
        f"| Max | {latencies[-1]:.3f} |",
        "",
    ]

    if p95 > 2.0:
        lines.append(f"⚠️ **High latency**: p95 = {p95:.2f}s — may miss 5m bar windows")
    elif p95 > 1.0:
        lines.append(f"Note: p95 = {p95:.2f}s — acceptable but monitor for growth")
    else:
        lines.append(f"Latency OK: p95 = {p95:.2f}s")
    lines.append("")


def _section_anomalies(
    lines: list,
    decisions: list[dict],
    signals: list[dict],
    risk_blocks: list[dict],
    pnl_rows: list[dict],
) -> None:
    lines.append("## Anomalies Detected")
    anomalies: list[str] = []

    # Zero buys
    buys = sum(1 for d in decisions if d.get("action") == "buy")
    if buys == 0 and decisions:
        anomalies.append("**Zero buys** executed during the session — check strategy min_conviction thresholds and AI filter universe size.")

    # Single dominant skip reason (>70% of skips)
    if risk_blocks:
        reason_counts: Counter = Counter(r.get("reason", "unknown") for r in risk_blocks)
        top_reason, top_count = reason_counts.most_common(1)[0]
        if top_count / len(risk_blocks) > 0.70:
            anomalies.append(f"**Dominant skip reason**: `{top_reason}` accounts for {top_count/len(risk_blocks)*100:.0f}% of all skips — may indicate a misconfigured guard.")

    # Confidence not monotone with buy rate
    buckets: dict[float, list[int]] = defaultdict(list)
    for d in decisions:
        conf = d.get("confidence")
        if conf is None:
            continue
        try:
            bucket = round(float(conf), 1)
            buckets[bucket].append(1 if d.get("action") == "buy" else 0)
        except (TypeError, ValueError):
            pass
    if len(buckets) >= 3:
        keys = sorted(buckets)
        rates = [sum(buckets[k]) / len(buckets[k]) for k in keys]
        inversions = sum(1 for i in range(1, len(rates)) if rates[i] < rates[i - 1] - 0.05)
        if inversions >= 2:
            anomalies.append(f"**Confidence non-monotone**: {inversions} inversions in confidence→buy-rate — calibrator or conviction threshold may need tuning.")

    # Regime stuck
    regime_counts: Counter = Counter(d.get("regime_name") for d in decisions if d.get("regime_name"))
    if regime_counts:
        top_r, top_rc = regime_counts.most_common(1)[0]
        if top_rc / max(sum(regime_counts.values()), 1) > 0.95:
            anomalies.append(f"**Regime stuck at `{top_r}`** ({top_rc/sum(regime_counts.values())*100:.0f}% of decisions) — HMM may not be transitioning.")

    # High latency
    latencies = []
    for d in decisions:
        lat = d.get("decision_latency_seconds")
        if lat is not None:
            try:
                latencies.append(float(lat))
            except (TypeError, ValueError):
                pass
    if latencies:
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        if p95 > 2.0:
            anomalies.append(f"**High latency**: p95 decision latency = {p95:.2f}s — symbol loop may be too slow for 5m bars.")

    if not anomalies:
        lines += ["_No anomalies detected._", ""]
    else:
        for a in anomalies:
            lines.append(f"- {a}")
        lines.append("")


def _section_improvements(
    lines: list,
    decisions: list[dict],
    signals: list[dict],
    risk_blocks: list[dict],
) -> None:
    lines.append("## Suggested Improvements")

    suggestions: list[str] = []

    # --- Data collection ---
    suggestions.append(
        "**Data for further collection**: Record `portfolio.gross_exposure`, "
        "`portfolio.cash`, and `portfolio.equity` in each trace record — these "
        "are needed to build position-sizing features for the return ranker."
    )

    # --- Model ---
    suggestions.append(
        "**Return ranker**: After 5+ sessions of training data, enable "
        "`return_ranker.enabled: true` in config and disable keras overlay. "
        "Monitor holdout MAE and correlation vs Keras scores for one week before "
        "making it the sole scorer."
    )

    # Strategy-specific
    strat_buy: Counter = Counter()
    for d in decisions:
        for sig in (d.get("signals") or []):
            if str(sig.get("action", "")).lower() == "buy":
                strat_buy[sig.get("name", "?")] += 1
    if strat_buy:
        least_active = strat_buy.most_common()[-1]
        suggestions.append(
            f"**Strategy `{least_active[0]}`** produced the fewest buy signals ({least_active[1]}) — "
            "investigate whether its feature inputs are being computed correctly."
        )

    # Skip analysis
    if risk_blocks:
        reason_counts: Counter = Counter(r.get("reason", "?") for r in risk_blocks)
        top_reason, top_count = reason_counts.most_common(1)[0]
        if "exposure" in top_reason or "leverage" in top_reason:
            suggestions.append(
                f"**Top skip reason `{top_reason}`**: consider raising `max_portfolio_leverage` "
                "or adding more brokers/accounts to increase buying power."
            )
        elif "cooldown" in top_reason:
            suggestions.append(
                f"**Top skip reason `{top_reason}`**: shorten `cooldown_seconds` or "
                "track per-strategy cooldowns instead of per-symbol."
            )

    # top_movers_rf check
    tm_buy = sum(1 for d in decisions for sig in (d.get("signals") or [])
                 if sig.get("name") == "top_movers_rf" and str(sig.get("action", "")).lower() == "buy")
    if tm_buy < 50:
        suggestions.append(
            f"**`top_movers_rf`** generated only {tm_buy} buy signals — investigate "
            "whether its RF model is loaded and feature inputs are being computed correctly."
        )

    # stat_arb_pairs check
    sa_buy = sum(1 for d in decisions for sig in (d.get("signals") or [])
                 if sig.get("name") == "stat_arb_pairs" and str(sig.get("action", "")).lower() == "buy")
    if sa_buy == 0 and decisions:
        suggestions.append(
            "**`stat_arb_pairs`** produced zero buy signals — check ADF cointegration "
            "passing rate and active pair count in logs."
        )

    suggestions.append(
        "**Investigation**: cross-join `decisions_full.jsonl` with actual trade P&L "
        "(once available from broker) to validate that high-confidence decisions "
        "yield better returns than low-confidence ones."
    )

    for s in suggestions:
        lines.append(f"- {s}")
    lines.append("")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _load_jsonl_filtered(path: Path, decision_types: set[str]) -> list[dict]:
    records = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("decision") in decision_types or r.get("action") == "exit":
                    records.append(r)
            except json.JSONDecodeError:
                pass
    return records


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    main()
