# AI Symbol Filter

## Purpose
Scores a large universe of symbols and returns an ordered list for trading.

## Implementation
- Entry point: `app/data/ai_filter.py` via `TradingAgent._refresh_dynamic_symbols`.
- Loads a linear model from `model_path` if fresh; retrains if missing or stale.
- Builds features from the latest rolling window of returns and volumes.
- Adds a news catalyst flag per symbol to the feature vector.
- Scores each symbol and sorts by descending score.
- Optionally performs lightweight online updates on the latest bars.
- While a refresh is in-flight, the trading loop keeps using the last valid symbol list.
- On startup, the trader seeds the symbol list from the last saved checkpoint while the filter runs.

## Feature Window
- Uses the last `window` returns and volumes for each symbol.
- Feature vector: mean return, std return, momentum sum, last return,
  volume z-score, catalyst flag, plus intraday signal metrics
  (30m/60m returns, early volume %, runup/drawdown %, and absolute moves).
When signal features are added or removed, retrain the model to avoid shape mismatches.
- Keep `lookback_days` low for live runs to reduce scoring latency (current default: 2).

## Configuration
`config/config.yaml`:
- `data.dynamic_symbols.ai_filter.enabled`
- `data.dynamic_symbols.ai_filter.interval`
- `data.dynamic_symbols.ai_filter.lookback_days`
- `data.dynamic_symbols.ai_filter.window`
- `data.dynamic_symbols.ai_filter.retrain_hours`
- `data.dynamic_symbols.ai_filter.model_path`
- `data.dynamic_symbols.ai_filter.train_max_symbols`
- `data.dynamic_symbols.ai_filter.max_samples_per_symbol`
- `data.dynamic_symbols.ai_filter.objective`
- `data.dynamic_symbols.ai_filter.time_penalty_per_bar`
- `data.dynamic_symbols.ai_filter.feed`
- `data.dynamic_symbols.ai_filter.timeout_seconds`
- `data.dynamic_symbols.ai_filter.retries`

Online updates:
- `data.dynamic_symbols.ai_filter.online.enabled`
- `data.dynamic_symbols.ai_filter.online.learning_rate`
- `data.dynamic_symbols.ai_filter.online.steps`
- `data.dynamic_symbols.ai_filter.online.max_symbols`

News features:
- `data.dynamic_symbols.ai_filter.news.enabled`
- `data.dynamic_symbols.ai_filter.news.provider` (use `brokers` to aggregate from enabled brokers; currently Alpaca-backed)
- `data.dynamic_symbols.ai_filter.news.base_url`
- `data.dynamic_symbols.ai_filter.news.lookback_hours`
- `data.dynamic_symbols.ai_filter.news.keywords`
- `data.dynamic_symbols.ai_filter.news.timeout_seconds`
- `data.dynamic_symbols.ai_filter.news.retries`

Universe selection:
- `data.dynamic_symbols.universe` (use `brokers_active` to start from enabled broker universes plus positions/orders)
- `data.dynamic_symbols.universe_price_filter` (filters broker universe by price_min and cash cap before scoring)
- `data.dynamic_symbols.cash_aware` / `data.dynamic_symbols.cash_cap_mode` enforce price <= cash;
  when cash <= 0 or cap < price_min only held/open-order symbols are kept.

## Usage
- Enable dynamic symbols: `data.dynamic_symbols.enabled: true`.
- Enable AI filter: `data.dynamic_symbols.ai_filter.enabled: true`.
- Leave `data.symbols: []` to rely on the filter output.
