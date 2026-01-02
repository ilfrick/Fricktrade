#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

set -euo pipefail

docker compose run --rm trader python -m app.main download --config /app/config/config.yaml \
  --symbols AAPL GOOGL TSLA MSFT
