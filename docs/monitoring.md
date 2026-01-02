# Monitoring

## Purpose
Expose Prometheus metrics and Grafana dashboards.

## Implementation
- Metrics: `app/monitoring/metrics.py`.
- Prometheus config: `prometheus/prometheus.yml`.
- Grafana dashboards: `grafana/provisioning/dashboards/*.json`.
- Latency dashboard: `grafana/provisioning/dashboards/autotrader_latency.json`.
- Audit/compliance exports: `app/monitoring/audit.py`.

## Key Metrics
- `trades_total`
- `orders_skipped_total`
- `pnl_percent`
- `drawdown_percent`
- `account_total`, `account_cash`, `account_buying_power`, `account_invested`
- `position_qty`, `position_value`
- `open_orders`
- `decision_latency_seconds`
- `order_enqueue_latency_seconds`
- `signal_return_30m_pct`, `signal_return_60m_pct`, `signal_early_volume_pct`
- `signal_runup_pct`, `signal_drawdown_pct`, `signal_abs_move`, `signal_runup_abs`, `signal_drawdown_abs`

## Audit & Compliance Logs
- Audit logs capture full decision traces in JSONL for later replay.
- Compliance exports can write JSONL and CSV summaries per day.
- Enable via `monitoring.audit.enabled` and `monitoring.compliance.enabled`.
- Retention and reason-code enforcement are controlled via `monitoring.audit.*` and `monitoring.compliance.*`.
- Compliance exports write `.sha256` sidecar digests (and optional signatures).

## Configuration
`config/config.yaml`:
- `monitoring.prometheus_port`
- `monitoring.audit.*` (JSONL audit logs for decision paths)
- `monitoring.compliance.*` (daily JSONL/CSV compliance exports)

## Access
- Grafana: `http://localhost:3002`
- Prometheus: `http://localhost:9090`
