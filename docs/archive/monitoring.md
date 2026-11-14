<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Monitoring

## Purpose
Expose Prometheus metrics and Grafana dashboards.

## Implementation
- Metrics: `app/monitoring/metrics.py`.
- Prometheus config: `prometheus/prometheus.yml`.
- Alertmanager config: `prometheus/alerts.yml` and `alertmanager/alertmanager.yml`.
- Grafana dashboards: `grafana/provisioning/dashboards/*.json`.
- Per-account dashboards: auto-generated from `.env` by `scripts/generate_grafana_dashboards.py` (template: `_template_account.json.template`).
- Latency dashboard: `grafana/provisioning/dashboards/fricktrade_latency.json`.
- Audit/compliance exports: `app/monitoring/audit.py`.

## Key Metrics
- `trades_total`
- `orders_skipped_total`
- `orders_skipped_by_broker_total`
- `pnl_percent`
- `drawdown_percent`
- `account_total`, `account_cash`, `account_buying_power`, `account_invested`
- `position_qty`, `position_value`
- `open_orders`
- `decision_latency_seconds`
- `order_enqueue_latency_seconds`
- `signal_return_30m_pct`, `signal_return_60m_pct`, `signal_early_volume_pct`
- `signal_runup_pct`, `signal_drawdown_pct`, `signal_abs_move`, `signal_runup_abs`, `signal_drawdown_abs`
- `portfolio_scale_fallback_total` — portfolio position scale fell back to 1.0 on error (labels: `symbol`)
- `take_profit_exits_total` — take-profit exit events (labels: `symbol`, `reason`: `take_profit` or `partial_take_profit`)
- `broker_requests_total`: Total number of API calls made to brokers, labeled by `broker`, `method`, and `status` (e.g., `success`, `error`).
- `broker_last_success_timestamp_seconds`: Unix timestamp of the last successful API call to a specific broker and method.

## Broker API Monitoring
Critical for ensuring continuous operation, broker API calls are monitored using the following:
-   **`broker_requests_total`**: A counter for every API interaction, categorized by broker, method, and the outcome (`success` or `error`). This metric helps identify frequent failures for specific API endpoints.
-   **`broker_last_success_timestamp_seconds`**: A gauge that records the Unix timestamp of the last successful call for each broker API method. This is crucial for detecting prolonged periods of API unavailability.

Alerts based on these metrics:
-   **`BrokerApiErrorRate`**: Triggered when the rate of `error` statuses for a broker API exceeds a defined threshold (e.g., 20% over 10 minutes). This indicates a high frequency of failed API calls.
-   **`BrokerApiNoSuccess`**: A critical alert fired if no successful API calls have been recorded for a specific broker method over an extended period (e.g., 10 minutes). This often signifies a complete loss of connectivity or a severe issue with the broker's API.

## Configuration
`config/config.yaml`:
- `monitoring.prometheus_port`
- `monitoring.audit.*` (JSONL audit logs for decision paths)
- `monitoring.compliance.*` (daily JSONL/CSV compliance exports)

## Scripts

- `scripts/live_pnl_summary.sh [PROMETHEUS_URL]` — queries Prometheus for PnL, drawdown, equity, positions, trades by strategy, win rate, leverage, VaR, and circuit breaker blocks. Quick terminal dashboard.
- `scripts/monitor_full_session.sh [--start HH:MM] [--end HH:MM]` — full US market session monitor. Captures decision traces, docker logs, Prometheus snapshots (every 60s), PnL/equity CSV (every 2 min), order flow, strategy signals, risk blocks, position changes, and generates an end-of-session summary report. Output: `data/monitoring/YYYY-MM-DD/`.
- `scripts/monitor_agent_session.sh` — lighter session capture (decision traces + docker logs + metrics samples).
- `scripts/monitor_sell_analysis.sh` — sell/exit performance report (triggers, backoff, leverage caps, retry budget).

## Access
- Grafana: `http://localhost:3002`
- Prometheus: `http://localhost:9090`
