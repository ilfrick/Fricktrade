# Monitoring

## Purpose
Expose Prometheus metrics and Grafana dashboards.

## Implementation
- Metrics: `app/monitoring/metrics.py`.
- Prometheus config: `prometheus/prometheus.yml`.
- Grafana dashboards: `grafana/provisioning/dashboards/*.json`.

## Key Metrics
- `trades_total`
- `orders_skipped_total`
- `pnl_percent`
- `drawdown_percent`
- `account_total`, `account_cash`, `account_invested`
- `position_qty`, `position_value`
- `open_orders`

## Configuration
`config/config.yaml`:
- `monitoring.prometheus_port`

## Access
- Grafana: `http://localhost:3002`
- Prometheus: `http://localhost:9090`
