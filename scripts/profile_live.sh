#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

set -euo pipefail

PROFILE_DIR="/data/profiles"
mkdir -p "${PROFILE_DIR}"

if ! command -v py-spy >/dev/null 2>&1; then
  python3 -m pip install --no-cache-dir py-spy
fi

python3 - <<'PY'
import time
import yaml

from app.utils.market import is_market_open, next_market_open

with open("/app/config/config.yaml", "r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)

while True:
    if is_market_open(cfg):
        print("Market open; starting profiling.")
        break
    nxt = next_market_open(cfg)
    print(f"Market closed; next open: {nxt}")
    time.sleep(60)
PY

ts="$(date -u +%Y%m%dT%H%M%SZ)"
CPROF="${PROFILE_DIR}/live-${ts}.cprof"
FLAME="${PROFILE_DIR}/live-${ts}.svg"

python3 -m cProfile -o "${CPROF}" -m app.main --config /app/config/config.yaml trade &
TRADER_PID=$!

sleep 5
py-spy record --duration 5400 --pid "${TRADER_PID}" --output "${FLAME}"

kill -INT "${TRADER_PID}" || true
for _ in $(seq 1 30); do
  if ! kill -0 "${TRADER_PID}" 2>/dev/null; then
    exit 0
  fi
  sleep 1
done
kill -TERM "${TRADER_PID}" || true
