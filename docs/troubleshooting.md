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
