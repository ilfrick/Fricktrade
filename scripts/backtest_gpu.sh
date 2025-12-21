#!/usr/bin/env bash
set -euo pipefail

docker compose --profile gpu run --rm backtest-gpu
