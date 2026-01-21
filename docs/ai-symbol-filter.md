<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# AI Symbol Filter

## Purpose
Scores a large universe of symbols and returns an ordered list for trading.

## Implementation
- Entry point: `app/data/ai_filter.py` via `TradingAgent._refresh_dynamic_symbols`.
- Loads a PPO policy (stable-baselines3) from `model_path` if fresh; retrains if missing or stale. Metadata is stored in a `.meta.json` sidecar.
- Builds features from the latest rolling window of returns and volumes.
- Adds a news catalyst flag (optionally processed via Ollama) per symbol to the feature vector.
- Scores each symbol and sorts by descending score.
- Fetches bars from the configured provider (`alpaca` or `yfinance`); yfinance reads from the market-cache service when enabled.
- Optionally performs lightweight online updates on the latest bars.
- While a refresh is in-flight, the trading loop keeps using the last valid symbol list.
- On startup, the trader seeds the symbol list from the last saved checkpoint while the filter runs.

## Feature Window
- Uses the last `window` returns and volumes for each symbol.
- Feature vector: mean return, std return, momentum sum, last return,
  volume z-score, catalyst flag, plus intraday signal metrics
  (30m/60m returns, early volume %, runup/drawdown %, and absolute moves).
When signal features are added or removed, retrain the PPO model to avoid shape mismatches.
- Keep `lookback_days` low for live runs to reduce scoring latency (current default: 2).

## Configuration
`config/config.yaml`:
- `data.dynamic_symbols.ai_filter.enabled`
- `data.dynamic_symbols.ai_filter.interval`
- `data.dynamic_symbols.ai_filter.lookback_days`
- `data.dynamic_symbols.ai_filter.window`
- `data.dynamic_symbols.ai_filter.retrain_hours`
- `data.dynamic_symbols.ai_filter.model_type` (default: `ppo`)
- `data.dynamic_symbols.ai_filter.model_path`
- `data.dynamic_symbols.ai_filter.rl.*` (PPO hyperparameters)
- `data.dynamic_symbols.ai_filter.train_max_symbols`
- `data.dynamic_symbols.ai_filter.max_samples_per_symbol`
- `data.dynamic_symbols.ai_filter.objective`
- `data.dynamic_symbols.ai_filter.time_penalty_per_bar`
- `data.dynamic_symbols.ai_filter.provider` (`alpaca` or `yfinance`)
- `data.dynamic_symbols.ai_filter.coverage_filter` (drop symbols with no bars before scoring)
- `data.dynamic_symbols.ai_filter.use_cached_symbols` (trader consumes cached symbol list when present)
- `market_cache.*` (Redis + file cache for yfinance bars; `cache_only` avoids direct yfinance calls)
- `data.dynamic_symbols.ai_filter.feed`
- `data.dynamic_symbols.ai_filter.timeout_seconds`
- `data.dynamic_symbols.ai_filter.retries`

Online updates:
- `data.dynamic_symbols.ai_filter.online.enabled`
- `data.dynamic_symbols.ai_filter.online.learning_rate`
- `data.dynamic_symbols.ai_filter.online.steps` (legacy alias for timesteps)
- `data.dynamic_symbols.ai_filter.online.timesteps`
- `data.dynamic_symbols.ai_filter.online.max_symbols`

News features:
- `data.dynamic_symbols.ai_filter.news.enabled`
- `data.dynamic_symbols.ai_filter.news.provider` (use `brokers` to aggregate from enabled brokers; currently Alpaca-backed)
- `data.dynamic_symbols.ai_filter.news.base_url`
- `data.dynamic_symbols.ai_filter.news.lookback_hours`
- `data.dynamic_symbols.ai_filter.news.keywords`
- `data.dynamic_symbols.ai_filter.news.timeout_seconds`
- `data.dynamic_symbols.ai_filter.news.retries`

Keras return overlay (optional):
- `data.dynamic_symbols.ai_filter.keras_returns.enabled`
- `data.dynamic_symbols.ai_filter.keras_returns.model_path`
- `data.dynamic_symbols.ai_filter.keras_returns.interval` (expects 5m inputs)
- `data.dynamic_symbols.ai_filter.keras_returns.score_mode` (`expected_return`, `short_term`, `up_prob`, `downside_risk`)
- `data.dynamic_symbols.ai_filter.keras_returns.weight` (score contribution scalar)

Universe selection:
- `data.dynamic_symbols.universe` (use `brokers_active` to start from Alpaca active universe today plus positions/orders)
- `data.dynamic_symbols.universe_price_filter` (filters broker universe by price_min and cash cap before scoring)
- `data.dynamic_symbols.cash_aware` / `data.dynamic_symbols.cash_cap_mode` enforce price <= cash;
  when cash <= 0 or cap < price_min only held/open-order symbols are kept.

## Usage
- Enable dynamic symbols: `data.dynamic_symbols.enabled: true`.
- Enable AI filter: `data.dynamic_symbols.ai_filter.enabled: true`.
- Leave `data.symbols: []` to rely on the filter output.
