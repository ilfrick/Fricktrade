#!/usr/bin/env bash
set -euo pipefail

log_file="/home/nicola/Fricktrade/monitoring_checks.log"
repo_dir="/home/nicola/Fricktrade"

cd "$repo_dir"

{
  echo "=== $(date -u +"%Y-%m-%dT%H:%M:%SZ") ==="
  echo "docker compose ps"
  docker compose ps

  echo ""
  echo "market-cache logs (last 5 lines)"
  docker compose logs --since 10m market-cache | tail -n 5 || true

  echo ""
  echo "redis ping"
  docker compose exec -T redis redis-cli ping || true

  echo ""
  echo "trader metrics check"
  if metrics=$(curl -sf http://localhost:8001/metrics); then
    echo "metrics ok"
    active_total=$(printf "%s" "$metrics" | awk '/^symbol_active\{/{sum+=$NF} END{printf "%d", sum+0}')
    active_by_broker=$(printf "%s" "$metrics" | awk '/^symbol_active_by_broker\{/{sum+=$NF} END{printf "%d", sum+0}')
    echo "active_symbols_total=${active_total}"
    echo "active_symbols_by_broker_total=${active_by_broker}"
  else
    echo "metrics fetch failed"
  fi
  echo ""
} >> "$log_file" 2>&1
