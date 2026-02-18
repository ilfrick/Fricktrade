#!/usr/bin/env python3
"""Decision Monitor — Market-hours aggregated trace analysis.

Incrementally reads decision trace JSONL files via byte-offset seeking,
aggregates running statistics, writes periodic JSONL summaries and a
human-readable session report at market close.

Designed to run via cron at 15:25 CET Mon-Fri (09:25 ET).
Stdlib only — no external dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# TraceFileReader — byte-offset incremental JSONL reader
# ---------------------------------------------------------------------------

MAX_CHUNK = 64 * 1024 * 1024  # 64 MB per read


class TraceFileReader:
    """Incrementally read new JSON lines from a growing JSONL file."""

    def __init__(self, path: Path, skip_existing: bool = False):
        self._path = path
        self._offset: int = 0
        self._partial: str = ""
        if skip_existing and path.exists():
            self._offset = path.stat().st_size

    def read_new(self) -> list[dict]:
        """Seek to offset, read new bytes, return parsed JSON records."""
        if not self._path.exists():
            return []
        try:
            size = self._path.stat().st_size
        except OSError:
            return []
        if size <= self._offset:
            return []

        records: list[dict] = []
        try:
            with self._path.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(self._offset)
                chunk = fh.read(MAX_CHUNK)
                self._offset = fh.tell()
        except OSError:
            return []

        text = self._partial + chunk
        lines = text.split("\n")
        # Last element is either empty (complete line) or a partial
        self._partial = lines[-1]
        for line in lines[:-1]:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return records


# ---------------------------------------------------------------------------
# DecisionAggregator — running statistics
# ---------------------------------------------------------------------------


class DecisionAggregator:
    """Aggregate decision trace records into session statistics."""

    def __init__(self):
        # Global
        self.total_records: int = 0
        self.decision_counts: Counter = Counter()  # decision field values
        self.buys: int = 0
        self.sells: int = 0
        self.holds: int = 0
        self.skips: int = 0

        # Per-symbol
        self.symbol_buys: Counter = Counter()
        self.symbol_sells: Counter = Counter()
        self.symbol_holds: Counter = Counter()
        self.symbol_skips: Counter = Counter()

        # Per-strategy signal distribution
        self.strategy_signal_buy: Counter = Counter()
        self.strategy_signal_sell: Counter = Counter()
        self.strategy_signal_hold: Counter = Counter()

        # Strategies driving trades
        self.buy_drivers: Counter = Counter()
        self.sell_drivers: Counter = Counter()

        # Skip reasons
        self.skip_reasons: Counter = Counter()

        # P0 fix verification
        self.config_weights_applied: int = 0
        self.uniform_weights: int = 0
        self.phantom_sells_filtered: int = 0
        self.pending_sell_blocks: int = 0

        # Regime
        self.regime_counts: Counter = Counter()

        # Latency
        self.latency_sum: float = 0.0
        self.latency_max: float = 0.0
        self.latency_count: int = 0

    def ingest(self, rec: dict) -> None:
        self.total_records += 1

        symbol = rec.get("symbol", "?")
        decision = str(rec.get("decision", "")).lower()
        action = str(rec.get("action", "")).lower()
        reason = rec.get("reason", "")
        stage = rec.get("stage", "")

        self.decision_counts[decision] += 1

        # Classify into buy/sell/hold/skip
        if decision == "order_enqueued":
            if action == "buy":
                self.buys += 1
                self.symbol_buys[symbol] += 1
                strat = rec.get("action_strategy", "")
                if strat:
                    self.buy_drivers[strat] += 1
            elif action == "sell":
                self.sells += 1
                self.symbol_sells[symbol] += 1
                strat = rec.get("action_strategy", "")
                if strat:
                    self.sell_drivers[strat] += 1
        elif decision == "skip":
            self.skips += 1
            self.symbol_skips[symbol] += 1
            if reason:
                self.skip_reasons[f"{stage}:{reason}" if stage else reason] += 1
        elif decision in ("hold", "exit"):
            if decision == "hold":
                self.holds += 1
                self.symbol_holds[symbol] += 1
            elif decision == "exit":
                self.sells += 1
                self.symbol_sells[symbol] += 1
        else:
            # Other decision types → hold bucket
            self.holds += 1
            self.symbol_holds[symbol] += 1

        # Strategy signal distribution (from signals array)
        signals = rec.get("signals")
        if isinstance(signals, list):
            for sig in signals:
                name = sig.get("name", sig.get("strategy", ""))
                sig_action = str(sig.get("action", "hold")).lower()
                if sig_action == "buy":
                    self.strategy_signal_buy[name] += 1
                elif sig_action == "sell":
                    self.strategy_signal_sell[name] += 1
                else:
                    self.strategy_signal_hold[name] += 1

        # P0 fix: orchestrator weights
        ow = rec.get("orchestrator_weights")
        if ow is not None:
            if isinstance(ow, dict):
                vals = list(ow.values())
            elif isinstance(ow, list):
                vals = ow
            else:
                vals = []
            if vals and all(v == 1.0 for v in vals):
                self.uniform_weights += 1
            else:
                self.config_weights_applied += 1

        # P0 fix: phantom sell detection
        # If signals contain sell entries but final action is hold/buy and
        # position qty <= 0, a phantom sell was filtered
        if isinstance(signals, list) and action != "sell":
            has_sell_signal = any(
                str(s.get("action", "")).lower() == "sell" for s in signals
            )
            pos_qty = 0.0
            portfolio = rec.get("portfolio", {})
            if isinstance(portfolio, dict):
                pos_qty = float(portfolio.get("position_qty", 0) or 0)
            if has_sell_signal and pos_qty <= 0:
                self.phantom_sells_filtered += 1

        # P0 fix: pending sell blocks
        if reason == "pending_sell_covers_position":
            self.pending_sell_blocks += 1

        # Regime
        regime_name = rec.get("regime_name")
        if regime_name:
            self.regime_counts[regime_name] += 1

        # Latency
        lat = rec.get("decision_latency_seconds")
        if lat is not None:
            try:
                lat = float(lat)
                self.latency_sum += lat
                self.latency_count += 1
                if lat > self.latency_max:
                    self.latency_max = lat
            except (TypeError, ValueError):
                pass

    def snapshot(self) -> dict:
        """Return current aggregated stats as a JSON-serializable dict."""
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "total_records": self.total_records,
            "buys": self.buys,
            "sells": self.sells,
            "holds": self.holds,
            "skips": self.skips,
            "buy_rate_pct": round(self.buys / self.total_records * 100, 2) if self.total_records else 0,
            "config_weights_applied": self.config_weights_applied,
            "uniform_weights": self.uniform_weights,
            "phantom_sells_filtered": self.phantom_sells_filtered,
            "pending_sell_blocks": self.pending_sell_blocks,
            "top_skip_reasons": dict(self.skip_reasons.most_common(15)),
            "regime": dict(self.regime_counts),
            "latency_mean": round(self.latency_sum / self.latency_count, 4) if self.latency_count else 0,
            "latency_max": round(self.latency_max, 4),
        }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_periodic_summary(agg: DecisionAggregator, path: Path) -> None:
    """Append one JSONL snapshot line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(agg.snapshot()) + "\n")


def write_session_report(agg: DecisionAggregator, path: Path, date_str: str, window: str) -> None:
    """Write human-readable session report."""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    sep = "=" * 60

    lines.append(sep)
    lines.append("  Decision Monitor — Session Report")
    lines.append(f"  Date: {date_str}  Window: {window}")
    lines.append(sep)
    lines.append("")

    # Global
    lines.append("--- Global Summary ---")
    lines.append(f"  Total records:  {agg.total_records:,}")
    lines.append(f"  Buys:           {agg.buys:,}   ({agg.buys / agg.total_records * 100:.2f}%)" if agg.total_records else f"  Buys:           {agg.buys}")
    lines.append(f"  Sells:          {agg.sells:,}")
    lines.append(f"  Holds:          {agg.holds:,}")
    lines.append(f"  Skips:          {agg.skips:,}")
    lines.append("")

    # Decision breakdown
    lines.append("--- Decision Types ---")
    for dec, cnt in agg.decision_counts.most_common():
        lines.append(f"  {dec}: {cnt:,}")
    lines.append("")

    # P0 verification
    lines.append("--- P0 Fix Verification ---")
    total_weight_checks = agg.config_weights_applied + agg.uniform_weights
    lines.append(f"  Config weights applied:      {agg.config_weights_applied:,}" + (f" ({agg.config_weights_applied / total_weight_checks * 100:.0f}%)" if total_weight_checks else ""))
    lines.append(f"  Uniform 1.0 (unfixed):       {agg.uniform_weights:,}")
    lines.append(f"  Phantom sells filtered:      {agg.phantom_sells_filtered:,}")
    lines.append(f"  Pending sell blocks:          {agg.pending_sell_blocks:,}")
    if total_weight_checks > 0 and agg.uniform_weights == 0 and agg.buys > 0:
        lines.append("  VERDICT: FIXES WORKING")
    elif agg.total_records == 0:
        lines.append("  VERDICT: NO DATA")
    else:
        issues = []
        if agg.uniform_weights > 0:
            issues.append("uniform weights detected")
        if agg.buys == 0:
            issues.append("zero buys")
        lines.append(f"  VERDICT: CHECK NEEDED — {', '.join(issues)}")
    lines.append("")

    # Strategy signal distribution
    all_strats = sorted(
        set(agg.strategy_signal_buy) | set(agg.strategy_signal_sell) | set(agg.strategy_signal_hold)
    )
    if all_strats:
        lines.append("--- Strategy Signal Distribution ---")
        for s in all_strats:
            b = agg.strategy_signal_buy.get(s, 0)
            sl = agg.strategy_signal_sell.get(s, 0)
            h = agg.strategy_signal_hold.get(s, 0)
            lines.append(f"  {s:25s} buy={b:<6} sell={sl:<6} hold={h}")
        lines.append("")

    # Strategies driving trades
    if agg.buy_drivers or agg.sell_drivers:
        lines.append("--- Strategies Driving Trades ---")
        lines.append(f"  Buys:  {dict(agg.buy_drivers.most_common())}")
        lines.append(f"  Sells: {dict(agg.sell_drivers.most_common())}")
        lines.append("")

    # Skip reasons
    if agg.skip_reasons:
        lines.append("--- Skip Reasons (top 15) ---")
        for reason, cnt in agg.skip_reasons.most_common(15):
            lines.append(f"  {reason}: {cnt:,}")
        lines.append("")

    # Regime
    if agg.regime_counts:
        lines.append("--- Regime Distribution ---")
        for regime, cnt in agg.regime_counts.most_common():
            lines.append(f"  {regime}: {cnt:,}")
        lines.append("")

    # Latency
    if agg.latency_count:
        lines.append("--- Decision Latency ---")
        lines.append(f"  Mean: {agg.latency_sum / agg.latency_count:.4f}s")
        lines.append(f"  Max:  {agg.latency_max:.4f}s")
        lines.append(f"  Count: {agg.latency_count:,}")
        lines.append("")

    # Top 20 symbols
    all_symbols = sorted(
        set(agg.symbol_buys) | set(agg.symbol_sells) | set(agg.symbol_holds) | set(agg.symbol_skips),
        key=lambda s: agg.symbol_buys[s] + agg.symbol_sells[s],
        reverse=True,
    )[:20]
    if all_symbols:
        lines.append("--- Top 20 Symbols ---")
        for sym in all_symbols:
            parts = []
            if agg.symbol_buys[sym]:
                parts.append(f"buy: {agg.symbol_buys[sym]}")
            if agg.symbol_sells[sym]:
                parts.append(f"sell: {agg.symbol_sells[sym]}")
            if agg.symbol_holds[sym]:
                parts.append(f"hold: {agg.symbol_holds[sym]}")
            if agg.symbol_skips[sym]:
                parts.append(f"skip: {agg.symbol_skips[sym]}")
            lines.append(f"  {sym}: {{{', '.join(parts)}}}")
        lines.append("")

    lines.append(sep)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_time(s: str) -> tuple[int, int]:
    """Parse 'HH:MM' into (hour, minute)."""
    parts = s.split(":")
    return int(parts[0]), int(parts[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Decision Monitor — aggregated trace analysis")
    parser.add_argument("--date", default=None, help="Date YYYY-MM-DD (default: today in --tz)")
    parser.add_argument("--trace-dir", default="data/reports/decision_trace", help="Trace JSONL directory")
    parser.add_argument("--output-dir", default="data/monitoring", help="Output directory")
    parser.add_argument("--summary-interval", type=int, default=300, help="Periodic summary interval (seconds)")
    parser.add_argument("--poll-interval", type=int, default=30, help="File poll interval (seconds)")
    parser.add_argument("--tz", default="US/Eastern", help="Timezone for market hours")
    parser.add_argument("--open-hour", default="9:30", help="Market open HH:MM")
    parser.add_argument("--close-hour", default="16:00", help="Market close HH:MM")
    parser.add_argument("--skip-existing", action="store_true", help="Start from end of file")
    args = parser.parse_args()

    tz = ZoneInfo(args.tz)
    now = datetime.now(tz)
    open_h, open_m = parse_time(args.open_hour)
    close_h, close_m = parse_time(args.close_hour)

    if args.date:
        date_str = args.date
    else:
        date_str = now.strftime("%Y-%m-%d")

    # Trace file uses UTC date in filename
    # Parse the target date and figure out UTC date
    target_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=tz)
    market_open = target_date.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
    market_close = target_date.replace(hour=close_h, minute=close_m, second=0, microsecond=0)

    # Trace files are named by UTC date
    utc_date_str = market_open.astimezone(timezone.utc).strftime("%Y-%m-%d")

    trace_path = Path(args.trace_dir) / f"{utc_date_str}.jsonl"
    output_dir = Path(args.output_dir) / date_str
    summary_path = output_dir / "decision_summary.jsonl"
    report_path = output_dir / "session_report.txt"
    output_dir.mkdir(parents=True, exist_ok=True)

    window_str = f"{args.open_hour}-{args.close_hour} {args.tz}"

    print(f"[decision_monitor] date={date_str} trace={trace_path}")
    print(f"[decision_monitor] window={window_str}")
    print(f"[decision_monitor] output={output_dir}")
    sys.stdout.flush()

    # Wait for market open
    now = datetime.now(tz)
    if now < market_open:
        wait = (market_open - now).total_seconds()
        print(f"[decision_monitor] waiting {wait:.0f}s for market open at {args.open_hour} {args.tz}")
        sys.stdout.flush()
        time.sleep(wait)

    reader = TraceFileReader(trace_path, skip_existing=args.skip_existing)
    agg = DecisionAggregator()
    last_summary = time.monotonic()

    print(f"[decision_monitor] monitoring started — polling every {args.poll_interval}s")
    sys.stdout.flush()

    while True:
        now = datetime.now(tz)
        if now >= market_close:
            # Final read
            for rec in reader.read_new():
                agg.ingest(rec)
            break

        records = reader.read_new()
        for rec in records:
            agg.ingest(rec)

        if records:
            print(f"[decision_monitor] +{len(records)} records — total={agg.total_records} buys={agg.buys} sells={agg.sells}")
            sys.stdout.flush()

        # Periodic summary
        elapsed = time.monotonic() - last_summary
        if elapsed >= args.summary_interval and agg.total_records > 0:
            write_periodic_summary(agg, summary_path)
            last_summary = time.monotonic()

        time.sleep(args.poll_interval)

    # Session end
    print(f"[decision_monitor] market closed — writing final report")
    sys.stdout.flush()

    if agg.total_records > 0:
        write_periodic_summary(agg, summary_path)
    write_session_report(agg, report_path, date_str, window_str)

    print(f"[decision_monitor] done — {agg.total_records:,} records, {agg.buys} buys, {agg.sells} sells")
    print(f"[decision_monitor] report: {report_path}")


if __name__ == "__main__":
    main()
