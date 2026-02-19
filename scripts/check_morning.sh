#!/usr/bin/env bash
set -euo pipefail

# One-shot check: did short-cover orders fill + is monitoring running?
# Scheduled via cron for 15:35 CET (5 min after market open)

LOG="/home/nicola/Fricktrade/data/monitoring/morning_check.log"
TODAY=$(TZ="Europe/Rome" date +%F)

{
  echo "=== Morning check $TODAY $(TZ='Europe/Rome' date) ==="

  # 1. Check if short-cover orders filled
  set -a; source /home/nicola/Fricktrade/.env; set +a
  KEYS=(${ALPACA_API_KEYS//,/ })
  SECRETS=(${ALPACA_API_SECRETS//,/ })
  # Account 2 (index 1)
  K="${KEYS[1]}"
  S="${SECRETS[1]}"

  echo ""
  echo "--- Account 2: Short-cover orders ---"
  python3 -c "
import os, json, urllib.request
k, s = '$K', '$S'

# Check open orders
req = urllib.request.Request('https://paper-api.alpaca.markets/v2/orders?status=open',
    headers={'APCA-API-KEY-ID': k, 'APCA-API-SECRET-KEY': s})
orders = json.loads(urllib.request.urlopen(req).read())
pending = [o for o in orders if o['symbol'] in ('MCY','SBCF','SIGI')]
if pending:
    print('WARNING: Close orders still pending!')
    for o in pending:
        print(f'  {o[\"symbol\"]} status={o[\"status\"]}')
else:
    print('OK: No pending close orders (filled or cancelled)')

# Check positions
req2 = urllib.request.Request('https://paper-api.alpaca.markets/v2/positions',
    headers={'APCA-API-KEY-ID': k, 'APCA-API-SECRET-KEY': s})
positions = json.loads(urllib.request.urlopen(req2).read())
shorts = [p for p in positions if float(p.get('market_value',0)) < 0]
if shorts:
    print(f'WARNING: {len(shorts)} short positions still open!')
    for p in shorts:
        print(f'  {p[\"symbol\"]} qty={p[\"qty\"]} mv={p[\"market_value\"]}')
else:
    print('OK: Zero short positions')

# Account summary
req3 = urllib.request.Request('https://paper-api.alpaca.markets/v2/account',
    headers={'APCA-API-KEY-ID': k, 'APCA-API-SECRET-KEY': s})
a = json.loads(urllib.request.urlopen(req3).read())
eq = float(a.get('equity',0))
smv = abs(float(a.get('short_market_value',0)))
pct = (smv/eq*100) if eq else 0
print(f'Equity: {eq:.2f}  Short MV: {smv:.2f}  Short exposure: {pct:.1f}%')
"

  # 2. Check if monitoring script is running
  echo ""
  echo "--- Monitoring script ---"
  if pgrep -f "monitor_full_session.sh" > /dev/null 2>&1; then
    echo "OK: monitor_full_session.sh is running"
  else
    echo "WARNING: monitor_full_session.sh is NOT running"
  fi

  MONITOR_DIR="/home/nicola/Fricktrade/data/monitoring/$TODAY"
  if [ -d "$MONITOR_DIR" ]; then
    echo "OK: Today's monitoring dir exists: $MONITOR_DIR"
    ls -lh "$MONITOR_DIR"
  else
    echo "WARNING: No monitoring dir for today"
  fi


  # 3. Fix-verification checks (runs at 15:35, 5 min into session)
  echo ""
  echo "--- Fix verification ---"

  TRACE_FILE="/home/nicola/Fricktrade/data/reports/decision_trace/${TODAY}.jsonl"
  DOCKER_LOG="$MONITOR_DIR/docker_logs.txt"

  # Fix 2: regime_probability emitted in trace
  echo ""
  echo "Fix 2 — regime_probability in trace:"
  if [ -f "$TRACE_FILE" ]; then
    count=$(grep -c '"regime_probability"' "$TRACE_FILE" 2>/dev/null || echo 0)
    if [ "$count" -gt 0 ]; then
      echo "  OK: $count records contain regime_probability"
    else
      echo "  WARNING: No regime_probability found in trace (may be too early)"
    fi
  else
    echo "  INFO: Trace file not yet created ($TRACE_FILE)"
  fi

  # Fix 1: stat_arb ADF logging in docker logs
  echo ""
  echo "Fix 1 — stat_arb ADF logging in docker logs:"
  if [ -f "$DOCKER_LOG" ]; then
    count=$(grep -c 'stat_arb:' "$DOCKER_LOG" 2>/dev/null || echo 0)
    if [ "$count" -gt 0 ]; then
      echo "  OK: $count stat_arb log lines found"
      grep 'stat_arb:' "$DOCKER_LOG" 2>/dev/null | tail -3 | sed 's/^/    /'
    else
      echo "  INFO: No stat_arb: lines yet (pairs refresh every ~30 min)"
    fi
  else
    echo "  INFO: Docker log not yet created ($DOCKER_LOG)"
  fi

  echo ""
  echo "=== Check complete ==="
} >> "$LOG" 2>&1
