<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Testing

## Unit Tests
Run the test suite:
```bash
docker compose run --rm tests-when-closed python -m pytest
```

### Test Files
- `tests/test_risk_manager.py` — RiskManager limits, cooldown, exposure caps, order limits, circuit breaker, RiskConfig dataclass (23 tests).
- `tests/test_risk_governance.py` — VaR/CVaR, group exposure, realized volatility (4 tests, requires `prometheus_client`).
- `tests/test_trader_sizing.py` — `_size_order` buy/sell paths, edge cases (11 tests, requires `tensorflow`).
- `tests/test_trader_signals.py` — `_combine_signals` priority/weighted modes (11 tests, requires `tensorflow`).
- `tests/test_symbol_manager.py` — symbol resolution, venue/sector mapping, position merging, cash capping (18 tests).
- `tests/test_market_state.py` — MarketState dataclass roundtrip and defaults.
- `tests/test_performance_tracker.py` — PerformanceTracker trade recording and stats.
- `tests/test_open_order_manager.py` — OpenOrderManager cache and pending checks.
- `tests/test_account_metrics.py` — AccountMetricsUpdater equity/VaR tracking.
- Additional test files cover API auth, market cache, IBKR, and order queue.

## Automated Tests When Markets Are Closed
The `tests-when-closed` service runs when markets are closed and can trigger backtests.
Key config:
- `backtest.run_when_closed: true`

## Backtests
Agent-aligned backtest:
```bash
docker compose run --rm trader python3 -m app.main backtest --config /app/config/config.yaml
```

Notes:
- Uses CSVs from `backtest.data_dir`.
- If `data.symbols` is empty, backtests can fall back to CSV symbols.

## Benchmarks
Benchmarks run when markets are closed if `benchmarking.run_when_closed: true`.
They use `scripts/benchmark_runner.py` and write reports/plots/PDFs under
`/data/reports/`.

## Smoke Checks
- `/health` endpoint returns `ok`.
- Grafana shows active broker, symbols, and orders.
