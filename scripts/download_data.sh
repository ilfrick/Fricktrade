#!/usr/bin/env bash
set -euo pipefail

docker compose run --rm trader python -m app.main download --config /app/config/config.yaml
