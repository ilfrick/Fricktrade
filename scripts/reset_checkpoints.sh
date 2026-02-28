#!/usr/bin/env bash
# Reset trader checkpoints and stale position state for a clean start.
# Run with: sudo bash scripts/reset_checkpoints.sh
set -euo pipefail

data_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"

files=(
  "${data_dir}/checkpoints/trader.json"
  "${data_dir}/learning/live_positions.json"
)

removed=0
for f in "${files[@]}"; do
  if [[ -f "$f" ]]; then
    rm -f "$f"
    echo "Removed: $f"
    ((removed++))
  else
    echo "Already gone: $f"
  fi
done

echo "Done. ${removed} file(s) removed."
