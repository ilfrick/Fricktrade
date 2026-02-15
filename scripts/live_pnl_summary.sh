#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
#
# Live PnL Summary — query Prometheus for key trading metrics.
# Usage: ./scripts/live_pnl_summary.sh [PROMETHEUS_URL]

set -euo pipefail

PROM_URL="${1:-http://localhost:9090}"

query() {
    local expr="$1"
    curl -s --fail-with-body "${PROM_URL}/api/v1/query?query=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$expr'))")" 2>/dev/null \
        | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    results = data.get('data', {}).get('result', [])
    if not results:
        print('N/A')
    else:
        for r in results:
            labels = r.get('metric', {})
            value = r['value'][1]
            label_str = ', '.join(f'{k}={v}' for k, v in labels.items() if k != '__name__')
            if label_str:
                print(f'  {label_str}: {value}')
            else:
                print(f'  {value}')
except Exception:
    print('  error')
" 2>/dev/null || echo "  query failed"
}

echo "======================================"
echo "  Fricktrade Live PnL Summary"
echo "  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "======================================"
echo ""

echo "--- Account ---"
echo "PnL %:"
query 'pnl_percent'
echo "Drawdown %:"
query 'drawdown_percent'
echo "Equity:"
query 'account_equity'
echo ""

echo "--- Positions ---"
echo "Open positions:"
query 'open_positions'
echo "Gross exposure:"
query 'gross_exposure'
echo ""

echo "--- Trades Today ---"
echo "Total trades:"
query 'increase(trades_total[1d])'
echo "Trades by strategy:"
query 'increase(trades_total[1d]) by (strategy)'
echo ""

echo "--- Win Rate (rolling 1d) ---"
echo "Wins:"
query 'increase(trade_wins_total[1d])'
echo "Losses:"
query 'increase(trade_losses_total[1d])'
echo ""

echo "--- Orders ---"
echo "Orders submitted:"
query 'increase(orders_submitted_total[1d])'
echo "Orders skipped:"
query 'sum(increase(orders_skipped_total[1d])) by (reason)'
echo "Order rejects:"
query 'increase(order_rejects_total[1d])'
echo ""

echo "--- Risk ---"
echo "Current leverage:"
query 'portfolio_leverage'
echo "VaR %:"
query 'var_percent'
echo "Circuit breaker blocks:"
query 'increase(circuit_breaker_blocks_total[1d])'
echo ""
echo "======================================"
