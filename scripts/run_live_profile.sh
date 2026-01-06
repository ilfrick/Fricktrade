#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/data/profiles"
LOG_FILE="${LOG_DIR}/profile_host.log"

mkdir -p "${LOG_DIR}"
{
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting live profiling run"
  cd "${ROOT_DIR}"
  docker compose stop trader || true
  docker compose stop healthwatch || true
  docker compose run --rm trader bash -lc "/app/scripts/profile_live.sh"
  docker compose up -d healthwatch trader
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Profiling run complete"
} >> "${LOG_FILE}" 2>&1
