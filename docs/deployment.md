<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Deployment

## Docker Compose (Default)
```bash
./scripts/compose_up.sh --build
```
This auto-enables GPU if available and falls back to CPU otherwise. Use `docker compose up -d --build` for a CPU-only start.
Ensure your Docker build environment is non-interactive by setting `DEBIAN_FRONTEND=noninteractive` where necessary in Dockerfiles, and `pip` is up-to-date.

## GPU vs CPU
- Default trader image is GPU-enabled.
- Force CPU: set `learning.device: cpu`.
- Disable GPU in backtests: `backtest.use_gpu: false`.
- The system automatically falls back to CPU for GPU-reliant components upon `CUDA out of memory` or similar errors. This fallback is latched until the next restart to prevent continuous failures.
- GPU state is stored in `/data/gpu_state.json`. To manually re-enable GPU after an error:
  ```bash
  sudo ./scripts/enable_gpu.sh
  ```
  Then restart containers with `./scripts/compose_up.sh --build`.

### CPU Fallback Behavior
When GPU is unavailable or disabled:
1. **RL Training/Inference**: Falls back to CPU via `_resolve_device()` in `train_rl.py`
2. **AI Symbol Filter**: Retries with CPU on CUDA OOM errors (see `ai_filter.py`)
3. **Keras Models**: Force CPU via `_force_tf_cpu()` when GPU disabled

### Multithreading on CPU
All processing remains multithreaded regardless of GPU availability:
- Risk manager: Thread-safe with `threading.Lock()`
- Order queue: Thread-safe execution
- Reporting loop: Runs in dedicated daemon thread
- Healthwatch: Parallel health checks
- Parallel symbol routing: `ThreadPoolExecutor` for multi-broker dispatch


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
./scripts/compose_up.sh --build
```

## Rollbacks
- Use git tags or branch pins for known-good versions.
