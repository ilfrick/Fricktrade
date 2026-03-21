<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

## Fricktrade Agent Guide

This repo contains a Python trading agent for 24/7 crypto (with dormant equity support), with broker adapters (Alpaca, Binance), risk controls, backtesting, data download, and metrics/monitoring. Current state: **v3.0, crypto-only mode, live on Alpaca paper + Binance Spot demo**.

## Quick Orientation

- `app/main.py` — CLI entrypoint: `trade`, `backtest`, `download`, `api`, `train`, `online-train`, `evaluate`, `ingest`.
- `app/agents/trader.py` — core `TradingAgent` class; orchestrates the full trading loop.
- Extracted modules:
  - `app/agents/symbol_manager.py` — symbol selection, AI filter, venue mapping
  - `app/agents/performance.py` — trade recording, stats, kill switch
  - `app/agents/open_orders.py` — open order cache and pending-order checks
  - `app/agents/account_metrics.py` — equity tracking, drawdown, VaR/CVaR
  - `app/agents/orchestrator.py` — `SimpleOrchestrator` (vote/weights combiner)
  - `app/agents/strategy_config.py` — per-strategy enable/disable helpers
  - `app/agents/decision_context.py` — decision trace record builder
- Broker adapters: `app/brokers/alpaca.py`, `app/brokers/binance.py` (Spot + Futures), `app/brokers/ibkr.py` (disabled), `app/brokers/base.py`.
- Risk: `app/risk/manager.py` (`RiskManager` with tz param), `app/risk/config.py`, `app/risk/haircut.py`.
- Execution: `app/execution/executor.py`, `app/execution/order_queue.py`, `app/execution/algos.py`, `app/execution/smart_router.py`, `app/execution/tca.py`.
- LLM: `app/llm/` package — see LLM section below.
- Data: `app/data/` — Alpaca + Binance market data, news RSS, alt-data, AI filter, return ranker.
- Backtesting: `app/backtest/agent_engine.py` (real agent loop on CSVs), `app/backtest/engine.py` (legacy SMA).
- Learning: `app/learning/` — PPO env, training, online updates, registry, drift monitor.
- API: `app/api/server.py` (FastAPI) — `/health`, `/config`, `/config/update`, `/restart`, `/ui`.
- Metrics: `app/monitoring/metrics.py` — all Prometheus metrics.
- Config: `config/config.yaml` (supports `${ENV_VAR}` interpolation).
- Market-hours gating: `app/utils/market.py` — `is_market_open()` with `open_mode: any` returns True 24/7 when Crypto is in `trading_venues`. Use `is_venue_open(cfg,'NYSE')` for equity-only hours.

## Current System State (v3.0)

### Active strategies (3 total, crypto-only, vote mode)

| Strategy | Asset Class | Notes |
|----------|-------------|-------|
| `crypto_momentum` | Crypto only | Multi-timeframe momentum |
| `crypto_mean_reversion` | Crypto only | Bollinger + VWAP + RSI |
| `trend_following` | Both (crypto-only in practice) | EMA crossover + Supertrend + VWAP + RSI + volume |

### Inactive strategies (disabled in Mar 21 strategic reset)

- `factor_model` — Equities only; Hurst-adaptive composite factor score
- `pattern_trading` — Equities only; breakout + ATR stop + partial TP
- `stat_arb_pairs` — Equities; disabled via `enabled: false`
- `top_movers_rf` — RF nowcast + session low-zone entry
- `gap_reversal` — Equities only; 9:35–10:30 ET gap fill
- `earnings_drift` — Equities only; PEAD — requires Alpha Vantage key
- `rl_policy` — PPO policy; disabled pending convergence work
- `intraday_momentum`, `market_maker`, `rl_policy_fees`

### Signal combine mode

`strategy.combine: vote`. All 3 enabled strategies run every cycle; each casts one unweighted vote (buy/sell/hold). Majority wins with configurable thresholds (`buy_vote_threshold: 2`, `exit_vote_threshold: 2`). Weights are completely ignored. Half-Kelly sizing with uncalibrated floor of 0.50.

### Active LLM components

| Component | Cadence | Model |
|-----------|---------|-------|
| Post-Session Analyst | Daily at session end | Gemini 2.5 Flash |
| Macro Regime Analyzer | Every 4h | Gemini 2.5 Flash |
| Risk Interpreter | On risk events | Gemini 2.5 Flash |
| Ollama Aggregate Sentiment | Every 15 min | llama3.2:3b (local) |

### Disabled LLM components (Mar 21 strategic reset)

- `TacticalMetaOrchestrator` — `enabled: false`; was injecting noise via oscillating parameters
- `StrategicMetaOrchestrator` — `enabled: false`; disabled alongside tactical
- `LLMPortfolioOrchestrator` — `llm_orchestrator.mode: vote`; class exists but is not instantiated
- Per-symbol sentiment (`llm.sentiment.enabled: false`) — vote mode skips `_enrich_signals()` entirely
- `symbols_filter` (`llm.symbols_filter.enabled: false`) — manual-only

## Running (Docker-first)

```bash
cp .env.example .env
# Fill in ALPACA_API_KEY, ALPACA_API_SECRET, GOOGLE_GEMINI_API_KEY at minimum
./scripts/compose_up.sh   # auto-detects GPU; generates per-account Grafana dashboards
```

Web UI: `http://localhost:18081/ui`
Grafana: `http://localhost:3002`
Health: `http://localhost:18081/health`

## Useful Commands

```bash
# Live trade (default container command)
docker compose run --rm trader python3 -m app.main trade --config /app/config/config.yaml

# Backtest
docker compose run --rm trader python3 -m app.main backtest --config /app/config/config.yaml

# Walk-forward backtest
python3 scripts/benchmark_runner.py --config config/config.yaml --walk-forward

# Download data
docker compose run --rm trader python3 -m app.main download --config /app/config/config.yaml --symbols AAPL MSFT

# Train RL policy (offline)
docker compose run --rm trader python3 -m app.main train --config /app/config/config.yaml

# Online training updates
docker compose run --rm learner

# Re-enable GPU after OOM
./scripts/enable_gpu.sh

# Ingest Alpaca multi-year bars
docker compose run --rm trader python3 -m app.main ingest --config /app/config/config.yaml
```

## Configuration Notes

- All config in `config/config.yaml`. `${ENV_VAR}` interpolation supported.
- `strategy.combine: vote` — weights under `orchestrator.strategy_weights` are inactive.
- `orchestrator.mode: select`, `top_k: 99` — effectively passes all strategies through; no RL orchestrator active.
- `stat_arb_pairs.enabled: false` — disabled in long-only mode; produces unhedged directional bets.
- `learning.guardrail.enabled: false` — was blocking all buys in sideways markets.
- `llm_orchestrator.mode: vote` — portfolio orchestrator not instantiated; no per-cycle LLM trade decisions.
- `quote_stream.enabled: false` — websockets library incompatible with alpaca-py `extra_headers`.
- `brokers.binance.futures: false` — Spot mode only; demo endpoint.
- `brokers.ibkr.enabled: false` — IBKR adapter present but inactive.
- `trading_limits.crypto_order_margin: 0.97` — 3% haircut on crypto buys to absorb price drift.
- `brokers.alpaca.accounts[*].asset_filter: crypto_only` — equity trading disabled (Mar 21 strategic reset)
- Guard rails disabled (Mar 21): `var.enabled: false`, `vol_targeting.enabled: false`, `exposure_caps.enabled: false`, `signal_bias_guard.enabled: false`, `kill_switch.enabled: false`
- `risk.max_portfolio_leverage: 3.0` (was 1.5); `risk.max_positions: 15` (was 30)
- `data.dynamic_symbols.universe: alpaca_active_all` — equities + crypto from Alpaca; no hardcoded lists.
- `data.dynamic_symbols.max_symbols: 150` — both accounts evaluate full 150-symbol universe; `buying_power_scaling: false`.
- `risk.pdt.force_swing: true` — holds equity positions overnight on PDT-restricted accounts (< $2500).
- `learning.device: auto` — uses CUDA if available, falls back to CPU.

## LLM Integration (`app/llm/`)

All LLM modules use **Gemini 2.5 Flash** by default. Switch any module via its `backend:` config key. Claude backend (`claude-sonnet-4-6`) is available but not default.

| Module | Backend | When Active | Purpose |
|--------|---------|-------------|---------|
| `client.py` | Both | On demand | `LLMClient` — unified API, $5/day budget circuit breaker, retry/backoff. `complete(backend, system_prompt, user_prompt, model=)` → `LLMResponse`. |
| `tactical_meta_orchestrator.py` | Gemini | **DISABLED** (was every 15 min) | Reads live metrics + macro regime + PostSession report. Proposes config changes within declared bounds. Disabled Mar 21 — was injecting noise. |
| `meta_orchestrator.py` (StrategicOrchestrator) | Gemini | **DISABLED** (was weekly) | Reads 7-day PostSession history + tactical change log. Writes `strategic_baseline.json`. Disabled Mar 21 alongside tactical. |
| `post_session.py` | Gemini | After market close | Grades session A–F; key findings with P&L estimates; per-strategy assessment. Writes `report_*.json` consumed by orchestrators. |
| `macro_regime.py` | Gemini | 4h TTL | FRED (VIX/DGS10/DXY) + Gemini → 5-regime classification. Used by tactical orchestrator and strategy weight adjustment. |
| `risk_interpreter.py` | Gemini | On `DriftMonitor` alert | Triages structural break vs noise; may set 1h trading pause or log recommended_action. |
| `portfolio_orchestrator.py` | Gemini | **DISABLED** (`mode: vote`) | Per-cycle cross-symbol quality filter; not instantiated in vote mode. Class preserved for future re-enable. |
| `sentiment.py` | Gemini | **DISABLED** | Per-symbol news sentiment; skipped in vote mode (`_enrich_signals()` not called). |
| `ollama_sentiment.py` | Ollama (local) | Every 15 min | Aggregate market sentiment from recent headlines via llama3.2:3b. |
| `symbols_filter.py` | Gemini | **DISABLED** | Pre-market symbol selection; manual-only. |

**Two-tier orchestrator hierarchy:**
- Strategic (weekly): sets weight baselines + per-strategy corridors → `strategic_baseline.json`
- Tactical (15-min): adjusts within those corridors → `changes.jsonl` + `last_result.json`
- When no strategic baseline exists, tactical self-imposes ±30% corridors from current config values

**Budget:** All LLM costs tracked against `llm.cost.daily_budget_usd: 5.00`. Resets midnight UTC. Budget exceeded → all LLM calls blocked until reset.

## Key Architecture Patterns

### Pending Guards (thread-safe)

| Guard | Dict Key | TTL | Prevents |
|-------|----------|-----|---------|
| `_pending_buy_symbols` | `(broker, symbol)` | 900s | Re-buying same symbol while order in flight |
| `_pending_sell_qty` | `(broker, symbol)` | Until fill/terminal | Duplicate sell order stacking |
| `_pending_notional` | broker | Until fill/terminal | Leverage race during notional reserve → enqueue gap |
| `_stuck_cooldown` | `(broker, symbol)` | 15 min | Re-entry after timed-out buy |
| `_exit_backoff_until` | `(broker, symbol)` | 1/2/4/8/15 min cap | Repeated exit attempt failures |

### Two-Phase Dispatch

`_run_symbol_batch()` splits symbols into holders and non-holders; holders processed first with `wait()` barrier. Guarantees exit orders complete before entry orders start.

### Crypto vs. Equity Branches

- Crypto: `"/" in symbol`
- Always market orders (no TWAP, no limit upgrade)
- Always fractional
- Stops from `risk.crypto.*`
- `factor_model` excluded from crypto
- Alt-data gates: Fear&Greed < 20 or OI change < -5% suppress longs

### Dust Positions

`qty < 1e-6` → all position-exit checks silently skipped (no close_position calls). Exit backoff set to 8h on `floors_to_zero` rejection. `_pending_sell_qty` not released on `floors_to_zero` → permanent sell block until restart.

## Common Pitfalls

- **Sell qty**: use `math.floor()`, never `round()` — float64 broker decimals can cause `round()` to exceed actual held qty
- **`record_pnl()` vs `update_daily_loss()`**: `record_pnl()` accumulates deltas; `update_daily_loss()` sets absolute day P&L — wrong one causes -543% false loss
- **`order_queue.enqueue()` sentinel**: returns `"queued"` (truthy) when order accepted but not started; previously returned `None` causing `order_failed` + skipped `_reserve_pending_buy` (POL/USD Mar 2 bug)
- **`_check_position_exit` crypto stops**: reads from `risk.crypto.*` for `/` symbols; old code used flat values → premature crypto exits
- **`is_market_open()` 24/7**: returns True always with Crypto venue in `trading_venues`; use `is_venue_open('NYSE')` for equity-only check
- **Binance `asset_class`**: check with `"crypto" in asset_class`, not `== "crypto"` (`str(AssetClass.CRYPTO).lower()` = `"assetclass.crypto"`)
- **`/USDT` symbols in Alpaca batches**: must be filtered at all data-path entry points; `build_symbols_by_broker()` applies the filter
- **`_build_symbol_batches()` in parallel mode**: MUST intersect with provided `symbols` set or caller-side filters (e.g. crypto-only) are silently bypassed
- **Meta-orchestrator kwarg names**: use `system_prompt=` and `user_prompt=` for `LLMClient.complete()`; wrong names silently skip the LLM call
- **Pending sell stacking**: `_pending_sell_qty` guard must be applied whenever `_is_closing_position`, not only when `not _can_short_here`
- **Binance demo timeouts**: use persistent `ThreadPoolExecutor` with `future.result(timeout=N)` for wall-clock deadlines; never use `with ThreadPoolExecutor` for timeout enforcement (`__exit__` calls `shutdown(wait=True)`)
- **Deposit-aware P&L**: `BrokerState.day_deposits_baseline` captures today's deposits at session start; each cycle subtracts new deposits from apparent P&L so cash injections don't appear as profit
- **`min_hold_minutes` in both paths**: must be checked both in `_check_position_exit` (ATR/stop/TP exits) AND in the signal path (`_is_closing_position` decision point)

## Extending the Codebase

- New strategies: extend `app/strategies/base.py`; wire in `app/agents/trader.py`; add to `strategy.names` in config; write test.
- New brokers: implement `app/brokers/base.py`; add iterator to `app/brokers/config_utils.py`; wire in `app/main.py:_build_broker()`.
- New metrics: define in `app/monitoring/metrics.py`; record at call site.
- New LLM modules: use `LLMClient` from `app/llm/client.py`; add to `_init_llm()` in trader.py; add config section under `llm:`.

See `docs/DEVELOPMENT.md` for detailed how-to guides.

## Testing

182 tests (17 skip without tensorflow/prometheus). Run: `pytest tests/ -v`.

```bash
docker compose run --rm trader pytest tests/ -v
```

Key test areas: risk manager, order queue, execution algos, strategy signals, broker routing, market hours, data quality, two-phase dispatch, pending notional guards.

## Monitoring

- Prometheus metrics: `http://localhost:8001/metrics`
- Grafana: `http://localhost:3002`
- API health: `http://localhost:18081/health`
- Decision traces: `data/reports/decision_trace/trace_YYYY-MM-DD.jsonl`
- Post-session reports: `data/reports/session/report_YYYY-MM-DD.json`
- Meta-orchestrator changes: `data/reports/meta_orch/changes.jsonl`
- Checkpoints: `data/checkpoints/trader.json` (updated every 60s)

## Documentation

- `README.md` — project overview and quick start
- `docs/ARCHITECTURE.md` — system design, data flow, threading model
- `docs/DEPLOYMENT.md` — full deployment guide, all config keys, broker setup
- `docs/STRATEGIES.md` — all 9 strategies: signals, parameters, limitations
- `docs/DEVELOPMENT.md` — adding strategies/brokers, testing, common pitfalls
- `docs/OPERATIONS.md` — daily monitoring, reading traces, common issues, manual overrides
- `docs/STATUS.md` — current implementation status and roadmap

## History

See `AGENTS.md` history section and git log for full commit-by-commit record. Major milestones (newest first):

- **Strategic reset — crypto-only, 3 strategies (2026-03-21, commit b1bf267):** System was losing money consistently (-5.1% over 3 months). Root causes: guard rail paralysis, tactical orchestrator noise injection, broken capital deployment (Kelly 0.15 floor). Changes: disabled equity trading (`asset_filter: crypto_only`), reduced from 9 strategies to 3 (`crypto_momentum`, `crypto_mean_reversion`, `trend_following`), killed both meta-orchestrators, disabled VaR/vol_targeting/exposure_caps/signal_bias_guard/kill_switch, raised Kelly floor to 0.50, tuned trend_following breakout/exit thresholds. Also: Docker resource limits on all services, stop-exit re-entry cooldown (30 min), per-parameter tactical hysteresis (60 min), portfolio-aware strategy context injection, cross-symbol correlations.

- **Round-7 safety and correctness fixes (2026-03-15, commit c4d74f4):** 14 fixes. `_pending_sell_qty` now released in enqueue() exception path (stuck-position safety). `EarningsDriftStrategy._last_buy` re-armed on restart from `opened_at`. EDGAR hot-path call removed (always returned 0). `twap_slices()` TypeError on fractional qty fixed. `position_state["last_market_state"]` now written per cycle (TMO was getting null indicators). `exit_vote_threshold: [1,3]` added to TMO bounds. MacroRegime FRED I/O moved outside `_refresh_lock`. SEC tickers cached 24h + User-Agent fixed. Yahoo/Google news fetches parallelised (removes 52s serial sleeps). `trend_following` crisis gate uses integer code 2 as primary. `_portfolio_for_broker()` returns safe empty dict on missing broker. CoinGlass upgraded to v3 API. `opportunity_cost_exit` uses current equity. `crypto_momentum` RSI `is not None` guard.

- **Round-6 live-trading readiness fixes (2026-03-15, commit ae4cd0a):** 12 fixes across all layers. Circuit breaker now forces `sell_to_close` on held positions (was returning None — position kept losing). `_enrich_signals` reads flat alt-data keys correctly (was reading `market_state["alt_data"]` nested, injector writes flat). `pattern_trading` infers `took_partial` on restart to prevent duplicate partial exits. `_flush_order_responses` releases `pending_sell_qty` by `filled_qty` not requested qty. `stat_arb_pairs` checks if symbol is held before voting sell (eliminates phantom sell votes). `_ENTRY_ALLOWED_REGIMES` moved to module-level constant. `_config_strategy_weights` converted to `@property` (live view of cfg). RSI uses pre-computed `indicators["rsi"]` in `crypto_momentum` and `trend_following`. Fear/greed fetched once per cycle (60s TTL) not 150×. Single `_now` in `_check_position_exit`. Earnings calendar symbol limit configurable.

- **Round-5 system review fixes (2026-03-15):** Added `risk_off` to `_ENTRY_ALLOWED_REGIMES` (system now trades in defensive-but-not-crisis conditions). MacroRegime name validated against known set (hallucinated names default to `range_bound`). `crypto_mean_reversion` sell gate: skip vote when `avg_entry >= sma` (position not opened by this strategy). `pattern_trading` returns hold for all crypto symbols (equity-only strategy). `pending_sell_qty` guard added to direct-`close_position` path. Pre-register `pending_sell_qty` inside the check-lock to eliminate TOCTOU window. `gross_exposure` decremented on position-close sells. Checkpoint patched to clear `crypto_momentum` from `disabled_strategies` on both Alpaca accounts (performance kill_switch had fired).

- **Multi-source news; Ollama aggregate sentiment; two-tier orchestrator hierarchy; per-broker LLM awareness (2026-03-11):** `news_rss.py` adds 10 RSS sources (CoinDesk, Cointelegraph, Reuters Business, etc.); raw article count 27→46/cycle. Ollama aggregate sentiment (15-min background thread, llama3.2:3b). TacticalMetaOrchestrator and StrategicOrchestrator fully wired with correct LLM kwarg names. Portfolio orchestrator prompt and response format made per-broker-aware (`broker_decisions` dict). `_build_meta_orch_metrics()` includes `brokers` dict.

- **Stale position exit strategies; min-hold in signal path; floors-to-zero handling; sell stacking fix (2026-03-09–10):** Alpha decay exit (entry strategy reversal), regime-conditional time exit (Hurst-scaled), opportunity cost exit (disabled). Min-hold guard added to signal path (not just position-exit path). `floors_to_zero` sets 8h exit backoff + keeps pending_sell permanently set. `_pending_sell_qty` guard fixed to always apply when `_is_closing_position`. `exit` pre-emption removed from `_combine_signals`. `factor_model` excluded from crypto.

- **Crypto stops; deposit-aware P&L; Binance fixes; per-broker exposure cap (2026-03-07–13):** `_check_position_exit` reads from `risk.crypto.*` for crypto. `BrokerState.day_deposits_baseline` subtracts intraday deposits from day P&L. Binance: N+1 ticker → bulk fetch; persistent pool for timeout enforcement; `avg_entry: None` fallback to last price; per-broker `max_crypto_exposure_pct: 95`.

- **LLM portfolio orchestrator; A/B shadow tracking; strategy win rates in prompt; price-move invalidation (2026-03-08):** `LLMPortfolioOrchestrator` wired (now disabled, `mode: vote`). Shadow combine records every cycle. Strategy win rates injected into LLM prompt. `_decision_prices` cache with 1% invalidation. `min_hold_minutes` guard added to `_check_position_exit`.

- **Binance broker; crypto 24/7; Alpaca fix (2026-03-01–06):** `BinanceBroker` (Spot + Futures demo). `load_universe("alpaca_active_all")`. Fractional trading. `_HybridMarketDataProvider`. Crypto always market orders. `asset_class` enum fix. Stablecoin-base filter. Per-symbol circuit breaker. PDT force-swing. Stuck-order timeout.

- **Strategy and execution overhaul (2026-02-16–28):** ~30-indicator injection. EarningsDriftStrategy (PEAD). MacroRegimeAnalyzer (FRED+LLM). AltData (Fear&Greed/CoinGlass/EDGAR). QuoteStream. `adaptive_slices()`. Half-Kelly sizing. ATR stops. Limit orders default. `SmartOrderRouter`. TCA feedback. Walk-forward backtesting. `RebalanceEngine`. `collect_crypto_training_data.py`.

- **v3.0 foundation (2026-02-01–13):** Extracted modules (symbol_manager, performance, open_orders, account_metrics). Concurrency/safety fixes (19 issues). `datetime.utcnow()` fully migrated. PDT suppression. Two-phase dispatch. Pending notional race prevention. Dead code removal.
