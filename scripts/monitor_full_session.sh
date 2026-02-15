#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
#
# Full US Market Session Monitor
# Captures complete decision flow for every symbol choice from NYSE open to close.
# Collects: decision traces, docker logs, Prometheus metrics snapshots,
# order flow, strategy signals, risk blocks, position changes, and PnL.
#
# Usage: ./scripts/monitor_full_session.sh [--start HH:MM] [--end HH:MM]
# Default: US market 09:30-16:00 ET (15:30-22:00 CET)
#
# Output: data/monitoring/YYYY-MM-DD/
#   decision_trace.jsonl    — every per-symbol decision (signal, skip, trade)
#   docker_logs.txt         — trader + healthwatch + api + learner logs
#   metrics_*.txt           — Prometheus snapshots every 60s
#   order_flow.jsonl        — extracted order submissions/completions/rejects
#   strategy_signals.jsonl  — extracted strategy signal events
#   risk_blocks.jsonl       — extracted risk/skip events
#   position_changes.jsonl  — position open/close events
#   pnl_snapshots.csv       — periodic PnL/drawdown/equity/leverage readings
#   session_summary.txt     — end-of-session summary report

set -euo pipefail

# --- Configuration ---
TZ_LOCAL="Europe/Rome"
# US market open/close in local timezone
DEFAULT_START="15:30"
DEFAULT_END="22:00"
METRICS_URL="http://localhost:8001/metrics"
PROM_URL="http://localhost:9090"
SAMPLE_INTERVAL=60    # seconds between metric samples
PNL_INTERVAL=120      # seconds between PnL snapshots

# Parse arguments
START_TIME="$DEFAULT_START"
END_TIME="$DEFAULT_END"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start) START_TIME="$2"; shift 2 ;;
        --end)   END_TIME="$2"; shift 2 ;;
        *)       echo "Unknown arg: $1"; exit 1 ;;
    esac
done

today=$(TZ="$TZ_LOCAL" date +%F)
now_ts=$(TZ="$TZ_LOCAL" date +%s)
start_ts=$(TZ="$TZ_LOCAL" date -d "$today $START_TIME" +%s)
end_ts=$(TZ="$TZ_LOCAL" date -d "$today $END_TIME" +%s)

if [ "$end_ts" -le "$start_ts" ]; then
    echo "End time must be after start time" >&2
    exit 1
fi

run_dir="/home/nicola/Fricktrade/data/monitoring/$today"
mkdir -p "$run_dir"

# --- Helpers ---
prom_query() {
    local expr="$1"
    local encoded
    encoded=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$expr'))")
    curl -sf "${PROM_URL}/api/v1/query?query=${encoded}" 2>/dev/null || echo "{}"
}

prom_value() {
    local expr="$1"
    prom_query "$expr" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    results = data.get('data', {}).get('result', [])
    vals = []
    for r in results:
        labels = r.get('metric', {})
        v = r['value'][1]
        lbl = ','.join(f'{k}={v2}' for k,v2 in labels.items() if k != '__name__')
        vals.append(f'{lbl}:{v}' if lbl else v)
    print('|'.join(vals) if vals else 'N/A')
except Exception:
    print('N/A')
" 2>/dev/null
}

# --- Wait for start ---
sleep_until=$(( start_ts - now_ts ))
if [ "$sleep_until" -gt 0 ]; then
    echo "[$today] Monitoring $START_TIME-$END_TIME ($TZ_LOCAL). Sleeping ${sleep_until}s until start." | tee "$run_dir/session.log"
    sleep "$sleep_until"
else
    echo "[$today] Monitoring $START_TIME-$END_TIME ($TZ_LOCAL). Starting now (past start time)." | tee "$run_dir/session.log"
fi

echo "$(TZ="$TZ_LOCAL" date): Session monitoring started" | tee -a "$run_dir/session.log"

# --- Background collectors ---
PIDS=()

# 1. Decision trace tail (the core decision flow)
trace_src="/home/nicola/Fricktrade/data/reports/decision_trace/${today}.jsonl"
trace_out="$run_dir/decision_trace.jsonl"
touch "$trace_out"
(
    # Wait for trace file to appear, then tail it
    while [ ! -f "$trace_src" ] && [ $(TZ="$TZ_LOCAL" date +%s) -lt "$end_ts" ]; do
        sleep 5
    done
    if [ -f "$trace_src" ]; then
        tail -F "$trace_src" >> "$trace_out" 2>/dev/null
    fi
) &
PIDS+=($!)

# 2. Docker logs (all trading-relevant containers)
logs_out="$run_dir/docker_logs.txt"
(
    while [ $(TZ="$TZ_LOCAL" date +%s) -lt "$end_ts" ]; do
        docker compose logs -f --tail=0 trader healthwatch api learner market-cache >> "$logs_out" 2>&1 || true
        sleep 5
    done
) &
PIDS+=($!)

# 3. Prometheus metrics snapshots (full scrape every SAMPLE_INTERVAL)
metrics_dir="$run_dir/metrics"
mkdir -p "$metrics_dir"
(
    while [ $(TZ="$TZ_LOCAL" date +%s) -lt "$end_ts" ]; do
        ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        outf="$metrics_dir/$(date -u +%H%M%S).txt"
        {
            echo "### $ts"
            curl -sf "$METRICS_URL" 2>/dev/null || echo "# metrics_unavailable"
        } > "$outf"
        sleep "$SAMPLE_INTERVAL"
    done
) &
PIDS+=($!)

# 4. PnL/equity/leverage snapshots (CSV for easy analysis)
pnl_out="$run_dir/pnl_snapshots.csv"
echo "timestamp,pnl_pct,drawdown_pct,equity,cash,leverage,open_positions,trades_total" > "$pnl_out"
(
    while [ $(TZ="$TZ_LOCAL" date +%s) -lt "$end_ts" ]; do
        ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        pnl=$(prom_value 'pnl_percent')
        dd=$(prom_value 'drawdown_percent')
        equity=$(prom_value 'account_total')
        cash=$(prom_value 'account_cash')
        leverage=$(prom_value 'portfolio_leverage')
        positions=$(prom_value 'count(position_qty > 0)')
        trades=$(prom_value 'sum(trades_total)')
        echo "${ts},${pnl},${dd},${equity},${cash},${leverage},${positions},${trades}" >> "$pnl_out"
        sleep "$PNL_INTERVAL"
    done
) &
PIDS+=($!)

# 5. Order flow extractor (parse docker logs in real-time for order events)
order_flow_out="$run_dir/order_flow.jsonl"
(
    while [ $(TZ="$TZ_LOCAL" date +%s) -lt "$end_ts" ]; do
        sleep 30
        # Extract order-related log lines from docker logs
        if [ -f "$logs_out" ]; then
            grep -E "(order_submitted|order_completed|order_rejected|order_cancelled|enqueue|pending_leverage_cap|position_exit)" "$logs_out" 2>/dev/null \
                | tail -n +1 > "$order_flow_out.tmp" 2>/dev/null || true
            if [ -f "$order_flow_out.tmp" ]; then
                mv "$order_flow_out.tmp" "$order_flow_out"
            fi
        fi
    done
) &
PIDS+=($!)

# 6. Strategy signal + risk block extractor (from decision traces)
(
    signals_out="$run_dir/strategy_signals.jsonl"
    risk_out="$run_dir/risk_blocks.jsonl"
    positions_out="$run_dir/position_changes.jsonl"
    while [ $(TZ="$TZ_LOCAL" date +%s) -lt "$end_ts" ]; do
        sleep 30
        if [ -f "$trace_out" ] && [ -s "$trace_out" ]; then
            # Extract buy/sell signals
            python3 -c "
import json, sys
seen = set()
with open('$trace_out') as f:
    for line in f:
        try:
            d = json.loads(line)
            key = (d.get('symbol',''), d.get('ts',''), d.get('outcome',''))
            if key in seen:
                continue
            seen.add(key)
            outcome = d.get('outcome','')
            if outcome in ('buy', 'sell', 'exit'):
                print(json.dumps({
                    'ts': d.get('ts'),
                    'symbol': d.get('symbol'),
                    'outcome': outcome,
                    'stage': d.get('stage'),
                    'strategy': d.get('strategy'),
                    'confidence': d.get('confidence'),
                    'signal': d.get('signal'),
                }, default=str), flush=True)
        except Exception:
            pass
" > "$signals_out.tmp" 2>/dev/null || true
            [ -f "$signals_out.tmp" ] && mv "$signals_out.tmp" "$signals_out"

            # Extract risk blocks / skips
            python3 -c "
import json
seen = set()
with open('$trace_out') as f:
    for line in f:
        try:
            d = json.loads(line)
            key = (d.get('symbol',''), d.get('ts',''), d.get('outcome',''))
            if key in seen:
                continue
            seen.add(key)
            outcome = d.get('outcome','')
            if outcome == 'skip':
                print(json.dumps({
                    'ts': d.get('ts'),
                    'symbol': d.get('symbol'),
                    'reason': d.get('reason'),
                    'stage': d.get('stage'),
                }, default=str), flush=True)
        except Exception:
            pass
" > "$risk_out.tmp" 2>/dev/null || true
            [ -f "$risk_out.tmp" ] && mv "$risk_out.tmp" "$risk_out"

            # Extract position changes
            python3 -c "
import json
seen = set()
with open('$trace_out') as f:
    for line in f:
        try:
            d = json.loads(line)
            key = (d.get('symbol',''), d.get('ts',''), d.get('outcome',''))
            if key in seen:
                continue
            seen.add(key)
            outcome = d.get('outcome','')
            if outcome in ('buy', 'sell', 'exit'):
                print(json.dumps({
                    'ts': d.get('ts'),
                    'symbol': d.get('symbol'),
                    'outcome': outcome,
                    'price': d.get('price'),
                    'qty': d.get('qty'),
                    'broker': d.get('broker'),
                    'strategy': d.get('strategy'),
                }, default=str), flush=True)
        except Exception:
            pass
" > "$positions_out.tmp" 2>/dev/null || true
            [ -f "$positions_out.tmp" ] && mv "$positions_out.tmp" "$positions_out"
        fi
    done
) &
PIDS+=($!)

# --- Main wait loop ---
echo "$(TZ="$TZ_LOCAL" date): Collectors started (${#PIDS[@]} background processes)" | tee -a "$run_dir/session.log"
echo "Output directory: $run_dir" | tee -a "$run_dir/session.log"

while [ $(TZ="$TZ_LOCAL" date +%s) -lt "$end_ts" ]; do
    sleep 60
    # Heartbeat
    alive=0
    for pid in "${PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null && alive=$((alive + 1))
    done
    echo "$(TZ="$TZ_LOCAL" date): heartbeat — $alive/${#PIDS[@]} collectors alive" >> "$run_dir/session.log"
done

# --- Cleanup ---
for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
done
wait 2>/dev/null || true

echo "$(TZ="$TZ_LOCAL" date): Session monitoring ended" | tee -a "$run_dir/session.log"

# --- Generate summary report ---
summary="$run_dir/session_summary.txt"
{
    echo "========================================"
    echo "  Session Summary: $today"
    echo "  Window: $START_TIME - $END_TIME ($TZ_LOCAL)"
    echo "  Generated: $(TZ="$TZ_LOCAL" date)"
    echo "========================================"
    echo ""

    echo "--- Decision Trace Stats ---"
    if [ -f "$trace_out" ] && [ -s "$trace_out" ]; then
        total=$(wc -l < "$trace_out")
        buys=$(grep -c '"outcome".*"buy"' "$trace_out" 2>/dev/null || echo 0)
        sells=$(grep -c '"outcome".*"sell"' "$trace_out" 2>/dev/null || echo 0)
        exits=$(grep -c '"outcome".*"exit"' "$trace_out" 2>/dev/null || echo 0)
        skips=$(grep -c '"outcome".*"skip"' "$trace_out" 2>/dev/null || echo 0)
        holds=$(grep -c '"outcome".*"hold"' "$trace_out" 2>/dev/null || echo 0)
        echo "  Total decisions: $total"
        echo "  Buys: $buys"
        echo "  Sells: $sells"
        echo "  Exits: $exits"
        echo "  Holds: $holds"
        echo "  Skips: $skips"
    else
        echo "  (no decision traces captured)"
    fi
    echo ""

    echo "--- Skip Reasons ---"
    if [ -f "$run_dir/risk_blocks.jsonl" ] && [ -s "$run_dir/risk_blocks.jsonl" ]; then
        python3 -c "
import json
from collections import Counter
reasons = Counter()
with open('$run_dir/risk_blocks.jsonl') as f:
    for line in f:
        try:
            d = json.loads(line)
            reasons[d.get('reason','unknown')] += 1
        except Exception:
            pass
for r, c in reasons.most_common(20):
    print(f'  {r}: {c}')
" 2>/dev/null || echo "  (parse error)"
    else
        echo "  (no risk blocks captured)"
    fi
    echo ""

    echo "--- Strategy Signals ---"
    if [ -f "$run_dir/strategy_signals.jsonl" ] && [ -s "$run_dir/strategy_signals.jsonl" ]; then
        python3 -c "
import json
from collections import Counter
strategies = Counter()
actions = Counter()
with open('$run_dir/strategy_signals.jsonl') as f:
    for line in f:
        try:
            d = json.loads(line)
            s = d.get('strategy','unknown')
            a = d.get('outcome','unknown')
            strategies[s] += 1
            actions[a] += 1
        except Exception:
            pass
print('  By strategy:')
for s, c in strategies.most_common():
    print(f'    {s}: {c}')
print('  By action:')
for a, c in actions.most_common():
    print(f'    {a}: {c}')
" 2>/dev/null || echo "  (parse error)"
    else
        echo "  (no signals captured)"
    fi
    echo ""

    echo "--- PnL Summary ---"
    if [ -f "$pnl_out" ] && [ $(wc -l < "$pnl_out") -gt 1 ]; then
        python3 -c "
import csv
rows = []
with open('$pnl_out') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)
if rows:
    last = rows[-1]
    print(f\"  Final PnL: {last.get('pnl_pct', 'N/A')}%\")
    print(f\"  Final Drawdown: {last.get('drawdown_pct', 'N/A')}%\")
    print(f\"  Final Equity: {last.get('equity', 'N/A')}\")
    print(f\"  Final Leverage: {last.get('leverage', 'N/A')}\")
    print(f\"  Open Positions: {last.get('open_positions', 'N/A')}\")
    print(f\"  Total Trades: {last.get('trades_total', 'N/A')}\")
    print(f\"  Snapshots: {len(rows)}\")
else:
    print('  (no data)')
" 2>/dev/null || echo "  (parse error)"
    else
        echo "  (no PnL snapshots)"
    fi
    echo ""

    echo "--- Files ---"
    for f in "$run_dir"/*; do
        if [ -f "$f" ]; then
            sz=$(du -sh "$f" | cut -f1)
            lines=$(wc -l < "$f" 2>/dev/null || echo 0)
            echo "  $(basename "$f"): $sz ($lines lines)"
        fi
    done
    if [ -d "$run_dir/metrics" ]; then
        mc=$(ls "$run_dir/metrics/" 2>/dev/null | wc -l)
        echo "  metrics/: $mc snapshots"
    fi
    echo ""

    echo "========================================"
} | tee "$summary"

echo ""
echo "Full session data saved to: $run_dir"
