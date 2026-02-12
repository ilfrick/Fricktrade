#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
#
# Monitor sell/exit performance during market hours.
# Captures exit triggers, success/failure rates, retry budget, and pending notional.
# Schedule: crontab -e → 25 15 * * 1-5 /path/to/scripts/monitor_sell_analysis.sh
#
set -euo pipefail

LOG_DIR="${LOG_DIR:-/data/logs}"
METRICS_URL="${METRICS_URL:-http://localhost:8001/metrics}"
REPORT_DIR="${REPORT_DIR:-/data/reports}"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
REPORT_FILE="${REPORT_DIR}/sell_analysis_$(date -u +%Y%m%d).txt"

mkdir -p "$REPORT_DIR"

{
  echo "=== Sell/Exit Analysis Report ==="
  echo "Generated: $TIMESTAMP"
  echo ""

  # --- Position exit triggers from trader log ---
  echo "--- Position Exit Triggers ---"
  if [ -f "$LOG_DIR/trader.log" ]; then
    grep -c "Position exit for" "$LOG_DIR/trader.log" 2>/dev/null | xargs -I{} echo "Total exit triggers: {}"
    for reason in hard_stop time_exit take_profit partial_take_profit trailing_stop; do
      count=$(grep -c "position_exit.*reason=$reason" "$LOG_DIR/trader.log" 2>/dev/null || echo 0)
      echo "  $reason: $count"
    done
  else
    echo "  (trader.log not found)"
  fi
  echo ""

  # --- Exit backoff events ---
  echo "--- Exit Backoff ---"
  if [ -f "$LOG_DIR/trader.log" ]; then
    grep -c "Exit backoff for" "$LOG_DIR/trader.log" 2>/dev/null | xargs -I{} echo "Backoff activations: {}"
    grep -c "exit_backoff_active" "$LOG_DIR/trader.log" 2>/dev/null | xargs -I{} echo "Skipped due to backoff: {}"
  else
    echo "  (trader.log not found)"
  fi
  echo ""

  # --- Pending leverage cap ---
  echo "--- Pending Leverage Cap ---"
  if [ -f "$LOG_DIR/trader.log" ]; then
    grep -c "pending_leverage_cap" "$LOG_DIR/trader.log" 2>/dev/null | xargs -I{} echo "Orders blocked by pending leverage: {}"
  else
    echo "  (trader.log not found)"
  fi
  echo ""

  # --- Order submission vs rejection by side ---
  echo "--- Order Submission Rates ---"
  if curl -sf "$METRICS_URL" > /tmp/metrics_snapshot.txt 2>/dev/null; then
    echo "Buy trades:"
    grep 'trades_total.*side="buy"' /tmp/metrics_snapshot.txt 2>/dev/null || echo "  (no data)"
    echo "Sell trades:"
    grep 'trades_total.*side="sell"' /tmp/metrics_snapshot.txt 2>/dev/null || echo "  (no data)"
    echo ""
    echo "Order rejects:"
    grep 'order_rejects_total' /tmp/metrics_snapshot.txt 2>/dev/null || echo "  (no data)"
    echo ""
    echo "Skipped orders:"
    grep 'skipped_orders_total' /tmp/metrics_snapshot.txt 2>/dev/null || echo "  (no data)"
    rm -f /tmp/metrics_snapshot.txt
  else
    echo "  (metrics endpoint unreachable)"
  fi
  echo ""

  # --- Retry budget usage ---
  echo "--- Retry Budget ---"
  if [ -f "$LOG_DIR/trader.log" ]; then
    grep -c '"status":"retrying"' "$LOG_DIR/trader.log" 2>/dev/null | xargs -I{} echo "Retry attempts: {}"
    grep -c "Queued order failed" "$LOG_DIR/trader.log" 2>/dev/null | xargs -I{} echo "Final rejections: {}"
  else
    echo "  (trader.log not found)"
  fi
  echo ""

  echo "=== End Report ==="
} | tee "$REPORT_FILE"
