# Troubleshooting

## Backtest fails: "No symbols configured"
- Ensure `backtest.symbols_source` is `data_dir` and CSVs exist in `backtest.data_dir`.
- Or set explicit `data.symbols` for the backtest.

## Orders rejected by Alpaca
- Verify account permissions (shorting, margin, PDT).
- Check symbol venue gating and market hours.
- Confirm quantity and notional limits.
- Alertmanager now includes broker/code/symbol/side for rejected orders.

## AI filter not running
- Ensure `data.dynamic_symbols.ai_filter.enabled: true`.
- Confirm Alpaca data credentials are set.
- Check logs for model load and heartbeat messages.

## GPU not used
- Confirm NVIDIA runtime is available.
- Set `learning.device: auto` and use GPU profile if needed.

## Grafana shows no data
- Confirm Prometheus is scraping `trader`.
- Check metrics endpoint on `:8001`.

## Containers restarting
- Check logs for stack traces.
- Validate config YAML for invalid values.

## Healthwatch alerts firing
- Confirm `healthwatch` container is running.
- Check `healthwatch_service_up` metrics in Prometheus.
- Verify service URLs in `healthwatch.targets`.
