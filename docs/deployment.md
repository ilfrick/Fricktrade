<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Deployment

## Docker Compose (Default)
```bash
docker compose up -d --build
```
Ensure your Docker build environment is non-interactive by setting `DEBIAN_FRONTEND=noninteractive` where necessary in Dockerfiles, and `pip` is up-to-date.

## GPU vs CPU
- Default trader image is GPU-enabled.
- Force CPU: set `learning.device: cpu`.
- Disable GPU in backtests: `backtest.use_gpu: false`.
- The system automatically falls back to CPU for GPU-reliant components upon `CUDA out of memory` or similar errors. This fallback is latched until the next restart to prevent continuous failures.


## Live vs Dev
- Use a separate project name for dev:
```bash
docker compose -p fricktrade-dev up -d --build
```
- Keep dev credentials empty for no live trading.

## Ports
- API: `18081`
- Grafana: `3002`
- Prometheus: `9090`
- Ollama: `11434`

## Restart / Redeploy
```bash
docker compose up -d --build
```

## Rollbacks
- Use git tags or branch pins for known-good versions.
