<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Troubleshooting

## Backtest fails: "No symbols configured"
- Ensure `backtest.symbols_source` is `data_dir` and CSVs exist in `backtest.data_dir`.
- Or set explicit `data.symbols` for the backtest.

## Orders rejected by Alpaca
- Verify account permissions (shorting, margin, PDT).
- Check symbol venue gating and market hours.
- Confirm quantity and notional limits.
- Alertmanager now includes broker/code/symbol/side/reason for rejected orders.

## AI filter not running
- Ensure `data.dynamic_symbols.ai_filter.enabled: true`.
- Confirm Alpaca data credentials are set.
- Check logs for model load and heartbeat messages.

## GPU not used
- Confirm NVIDIA runtime is available.
- Start the stack with `./scripts/compose_up.sh --build` so GPU devices are mapped into containers.
- Set `learning.device: auto` and use GPU profile if needed.
- If a CUDA error occurred, GPU usage is disabled until the next restart.

## Grafana shows no data
- Confirm Prometheus is scraping `trader`.
- Check metrics endpoint on `:8001`.

## Cached bars look stale
- Check `market_cache.max_age_multiplier` and `market_cache.ignore_staleness`.
- Confirm market-cache service is running and `data.provider: yfinance` or cache-only mode is enabled.

## Containers restarting
- Check logs for stack traces.
- Validate config YAML for invalid values.

## Healthwatch alerts firing
- Confirm `healthwatch` container is running.
- Check `healthwatch_service_up` metrics in Prometheus.
- Verify service URLs in `healthwatch.targets`.

## Broker Connectivity Issues
If you are receiving `BrokerApiErrorRate` or `BrokerApiNoSuccess` alerts, it indicates a problem with the connection or API calls to your configured broker.
- Check `trader` logs for specific error messages or exceptions related to broker API calls.
- Verify that your broker API keys (`APP_BROKER_API_KEY`, `APP_BROKER_SECRET_KEY`) and base URL (`APP_BROKER_BASE_URL`) are correctly configured and have the necessary permissions.
- Ensure network connectivity from the `trader` container to the broker's API endpoints.
- Review the broker's status page for any outages or announced issues.

## LLM/Ollama Timeouts
If you see `LLM catalyst request failed: Read timed out` errors:
- Ollama runs on CPU by default to reserve GPU for the trader's AI filter and RL model.
- CPU inference for LLMs is slow (~55-60s per request for 3B models).
- Increase `news.llm.timeout_seconds` to 120 or higher in `config/config.yaml`.
- Use a smaller model like `llama3.2:3b` instead of `llama3.1:8b`.
- Alternatively, disable LLM filtering: `news.llm.enabled: false`.

## Keras Model Loading Errors
If you see `Failed to load Keras model` or `quantization_config` errors:
- This typically occurs with models saved in different Keras versions.
- The system automatically tries `safe_mode=False` for cross-version compatibility.
- Ensure you're using Keras 3.x (installed via `keras>=3.0`).
- Re-save models if compatibility issues persist.

## Symbol Universe Contains Bad Symbols
If you see data errors for preferred shares, warrants, or ADRs:
- The scanner automatically excludes symbols ending in `.PR*`, `.W`, `.U`, `.UN`, etc.
- ADRs ending in `Y` (except whitelisted ones like SONY) are also excluded.
- SPAC warrants (5+ char symbols ending in W/U) are filtered out.
- Check `app/data/scanner.py` `_WHITELIST` to add legitimate tickers that match exclusion patterns.

## GPU Memory Issues
If you see CUDA out-of-memory errors:
- The trader container reserves GPU for AI filter (~5GB) and RL model inference.
- Ollama is configured to use CPU only (`CUDA_VISIBLE_DEVICES=` in docker-compose.yml).
- If OOM persists, reduce `data.dynamic_symbols.ai_filter.online.max_symbols`.
