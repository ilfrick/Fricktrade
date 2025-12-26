# Testing

## Unit Tests
Run the test suite:
```bash
docker compose run --rm tests-when-closed python -m pytest
```

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

## Smoke Checks
- `/health` endpoint returns `ok`.
- Grafana shows active broker, symbols, and orders.
