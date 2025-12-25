# Backtesting

## Purpose
Run historical simulations with the real trading loop or legacy SMA engine.

## Implementation
- Agent-aligned engine: `app/backtest/agent_engine.py`.
- Legacy SMA engine: `app/backtest/engine.py`.
- Uses CSV data from `backtest.data_dir`.

## Configuration
`config/config.yaml`:
- `backtest.mode` (agent or legacy)
- `backtest.data_dir`
- `backtest.start`
- `backtest.end`
- `backtest.initial_cash`
- `backtest.commission_pct`
- `backtest.use_gpu`
- `backtest.symbols_source` (data_dir, data, dynamic)
- `backtest.dynamic_symbols_enabled`
- If enabled and `data.symbols` is empty, the backtest falls back to symbols in `backtest.data_dir`.
- `backtest.news_enabled`
- `backtest.news_source` (none, local)
- `backtest.news_path` (JSON: {"YYYY-MM-DD": ["AAPL", ...]})
- `backtest.run_when_closed` (run backtest from tests-when-closed service)
- `backtest.plan.enabled`
- `backtest.plan.window_days`
- `backtest.plan.step_days`
- `backtest.plan.liquidity_tiers`
- `backtest.plan.sample_per_tier`
- `backtest.plan.seed`
