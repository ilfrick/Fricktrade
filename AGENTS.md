<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

## Fricktrade Operational Guide

This is the **single authoritative document** for understanding and operating Fricktrade. Files in `docs/` predate the strategic reset (Mar 21) and should be treated as archived planning material.

**System**: v3.0, crypto-only, paper trading. Alpaca paper (2 accounts) + Binance Spot demo.

---

## Current System State

| Parameter | Value |
|-----------|-------|
| Active strategies | `crypto_momentum` (40%), `crypto_mean_reversion` (60%) |
| Combine mode | `weighted` — weighted sum of buy confidences vs min_conviction |
| min_conviction | 0.25 (effective 0.15 when single-sided via SSCM=0.6) |
| Data interval | 1m bars |
| Brokers | Alpaca paper (2 accounts, `crypto_only`), Binance Spot demo |
| Trailing stop (crypto) | 4.5% global; 2.0% for momentum entries |
| Vol targeting | Enabled, target 4%, scale [0.3, 1.5] |
| Macro regime | Enabled; only "crisis" blocks entries |
| Per-symbol sentiment | Enabled (Ollama llama3.2:3b, 900s cache) |
| Aggregate sentiment | Enabled (Ollama, every 15 min) |
| Disabled | Tactical/strategic meta-orchestrators, portfolio orchestrator, RL policy, guardrail, VaR, exposure_caps, signal_bias_guard, kill_switch, symbols_filter |

---

## Quick Orientation (File Map)

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
- Data: `app/data/` — Alpaca + Binance market data, news RSS, alt-data, order book depth.
  - `app/data/order_book.py` — Binance WebSocket depth stream for bid/ask imbalance
  - `app/data/news_rss.py` — RSS from CoinDesk, CoinTelegraph, BitcoinMagazine, TheBlock, Reuters
- Backtesting: `app/backtest/agent_engine.py` (real agent loop on CSVs), `app/backtest/engine.py` (legacy SMA).
- Learning: `app/learning/` — PPO env, training, online updates, registry, drift monitor.
- API: `app/api/server.py` (FastAPI) — `/health`, `/config`, `/config/update`, `/restart`, `/ui`.
- Metrics: `app/monitoring/metrics.py` — all Prometheus metrics.
- Config: `config/config.yaml` (supports `${ENV_VAR}` interpolation).
- Market-hours gating: `app/utils/market.py` — `is_market_open()` with `open_mode: any` returns True 24/7 when Crypto is in `trading_venues`. Use `is_venue_open(cfg,'NYSE')` for equity-only hours.

---

## Active Strategies

### crypto_mean_reversion (primary, 60% weight)

**Buy when ALL three are true:**
1. Price below lower Bollinger Band (`bb_period: 50`, `bb_std: 2.0`)
2. RSI < 30.0 (`rsi_period: 70`)
3. Drop of >1% occurred in < 75 bars (liquidation cascade signature)

**Confidence:** Base 0.4 + depth-below-band * 0.6, boosted by:
- `cascade_score > 0.3` → +0.15 (long liquidations confirm forced selling)
- `funding_extreme > 0.5` → +0.10 (overleveraged longs unwinding)
- `exchange_divergence > 0.05` → +0.05 (Coinbase premium)
- `stablecoin_inflow > 0.3` → +0.05 (money arriving to buy)
- `social_velocity > 0.3` → +0.03 (rising attention)
- Bearish per-symbol sentiment (score < -0.3) → +0.08 (contrarian: supports dip-buy)
- Strong bullish sentiment (score > 0.5) → ×0.95 (may not be a real dip)

**Sell:** Price >= SMA with >0.5% profit AND entry was below SMA (won't sell positions opened by other strategies). Confidence scales with distance above SMA.

**Crash filter:** Disabled (`crash_filter_pct: 0` — hurt Bonferroni performance).

**Hard stop:** 5.0% (per-strategy override).

### crypto_momentum (40% weight)

**Buy when ALL positive:** Per-bar velocity across fast (25), medium (75), slow (300) windows all exceed per-bar threshold. Slow trend must also exceed 20% of per-bar threshold.

**Confidence modifiers:**
- Volume: boost up to 1.5x (soft gate, not hard requirement — thin weekend volume won't block)
- VWAP: overextended above → 0.7x; at VWAP → 1.1x; below in uptrend → 0.85x
- RSI: >80 → 0.5x; >72 → 0.75x
- Per-symbol sentiment: `(1.0 + 0.15 * score)` when confidence >= 0.3

**Sell:** All velocities negative and below threshold. Long-only: signals position exit.

**Trailing stop:** 2.0% (per-strategy override — tighter than global 4.5%).

### trend_following (INACTIVE)

Has config parameters but is **not** in `strategy.names`. Not instantiated. Do not assume it runs.

---

## Signal Combine Logic (Weighted Mode)

```
1. Each enabled strategy calls generate_signal() → {action, confidence, name}
2. _enrich_signals() applies context multipliers (regime, fear/greed, OI, vol)
3. Buy signals: weighted sum = MR_conf × 0.60 + momentum_conf × 0.40
4. If only one side signals (buy with no sell):
   effective_threshold = min_conviction × single_sided_conviction_multiplier
                       = 0.25 × 0.6 = 0.15
5. If both sides signal:
   effective_threshold = min_conviction = 0.25
6. winning_score > effective_threshold → execute
7. Half-Kelly position sizing with uncalibrated floor of 0.50
```

**Entry ranking** (`entry_ranking.enabled: true`): Non-holder symbols ranked by opportunity score before processing. Components: Bollinger %B proximity (40%), MFI oversold (30%), catalyst flag (15%), funding extreme (15%). When aggregate Ollama sentiment is bearish (score < -0.4, conf >= 0.5), all scores penalized 30%.

**Exit logic** (in order of priority):
1. Hard stop (6.0% crypto global, 5.0% MR override)
2. ATR stop (2.5x ATR)
3. Trailing stop (4.5% global, 2.0% momentum override)
4. Take profit (5.0%)
5. Alpha decay exit: if entry strategy now signals sell, exit (min hold 30 min)
6. Regime hold: Hurst-scaled max hold (base 120 min, trending 240, MR 90)
7. Circuit breaker: 8.0% per-symbol drawdown → force sell_to_close

**Vote thresholds** (`buy_vote_threshold`, `exit_vote_threshold`) exist in config but are **inactive** in weighted mode.

---

## Risk Management

### Crypto stops (active values)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `hard_stop_pct` | 6.0% | Global; MR overrides to 5.0% |
| `atr_stop_mult` | 2.5 | ATR multiplier |
| `trailing_stop_pct` | 4.5% | Global; momentum overrides to 2.0% |
| `take_profit_pct` | 5.0% | |
| `circuit_breaker_drawdown_pct` | 8.0% | Per-symbol; forces sell_to_close |
| `max_daily_loss_pct` | 5.0% | Crypto-specific |
| `max_crypto_exposure_pct` | 50.0% | |

### Portfolio-level guards

| Guard | Value | Status |
|-------|-------|--------|
| `max_portfolio_leverage` | 3.0 | Active |
| `max_positions` | 15 | Active |
| `vol_targeting` | target 4%, scale [0.3, 1.5] | **Enabled** |
| VaR | — | Disabled |
| exposure_caps | — | Disabled |
| kill_switch | — | Disabled |
| signal_bias_guard | — | Disabled |

### Execution guards

| Guard | Value |
|-------|-------|
| `min_hold_minutes` | 15 (both signal path and position-exit path) |
| `stop_exit_reentry_cooldown` | 30 min |
| `stuck_blacklist_after` | 3 consecutive timeouts |
| `crypto_order_margin` | 0.97 (3% haircut on crypto buys) |
| `min_notional` | $10 |
| `insufficient_stablecoin_cooldown` | 5 min |

### Exit backoff

Exponential: 1/2/4/8/15 min cap. `floors_to_zero` → 8h backoff. Stale `_pending_sell_qty` entries expire after 5 min (prevents permanent deadlock from lost order responses).

---

## LLM Integration

### Active components

| Component | Backend | Cadence | What it does |
|-----------|---------|---------|-------------|
| Per-symbol sentiment | Ollama llama3.2:3b | Per symbol (900s TTL cache) | Injects `market_state["llm_sentiment"]` with score [-1,+1] and confidence. Crypto-specific prompt. Both strategies apply confidence-gated multipliers. |
| Aggregate sentiment | Ollama llama3.2:3b | Every 15 min (background thread) | Market-wide score from recent headlines. Penalizes entry ranking when bearish. Visible in Grafana via `MARKET_SENTIMENT_SCORE`. |
| Post-session analyst | Gemini 2.5 Flash | Daily at session end | Grades session A-F. Writes `data/reports/session/report_YYYY-MM-DD.json`. |
| Macro regime | Gemini 2.5 Flash | 4h TTL | FRED (VIX/DGS10/DXY) + LLM → 5-regime. Only "crisis" blocks entries. |
| Risk interpreter | Gemini 2.5 Flash | On DriftMonitor alert | Triages structural break vs noise; may set 1h pause. |

### Disabled components

| Component | Reason |
|-----------|--------|
| Tactical meta-orchestrator | Was injecting noise via oscillating parameters (Mar 21) |
| Strategic meta-orchestrator | Disabled alongside tactical (Mar 21) |
| Portfolio orchestrator | Not instantiated in weighted mode |
| Symbols filter | Manual-only |

### Per-symbol sentiment flow

```
RSS feeds (CoinDesk, CoinTelegraph, etc.) → _raw_news_cache
    → NewsSentimentAnalyzer.analyze(symbol, articles)
    → Ollama llama3.2:3b with crypto-specific prompt
    → SentimentResult {score, confidence, bias, risk_flag}
    → inject_into_market_state() → market_state["llm_sentiment"]
    → crypto_momentum reads it: multiplier = 1.0 + 0.15 * score
    → crypto_mean_reversion reads it: bearish boosts, bullish dampens
```

Confidence gate: sentiment ignored when confidence < 0.3. Cache TTL: 900s. The old double-application in `_enrich_signals()` has been removed.

**News executor:** 2 workers (catalyst fetch + sentiment in parallel). Ollama cold-start is ~23s on first call.

**Budget:** `llm.cost.daily_budget_usd: 5.00`. Resets midnight UTC. Ollama is free (local).

---

## Configuration Reference

### Combine mode

```yaml
strategy:
  combine: weighted          # "weighted" or "vote"
  weights:
    crypto_mean_reversion: 0.60
    crypto_momentum: 0.40
  min_conviction: 0.25       # minimum weighted score to trigger buy
  single_sided_conviction_multiplier: 0.6  # reduces threshold when only one side signals
  names:
    - crypto_momentum
    - crypto_mean_reversion
```

### Strategy parameters

```yaml
strategy.params:
  crypto_momentum:
    fast_window: 25            # bars (1m)
    medium_window: 75
    slow_window: 300
    trailing_stop_pct: 2.0
  crypto_mean_reversion:
    bb_period: 50              # Bonferroni-validated optimal
    bb_std: 2.0
    rsi_period: 70
    rsi_oversold: 30.0
    drop_window_bars: 75
    hard_stop_pct: 5.0
    crash_filter_pct: 0        # disabled — hurt Bonferroni
```

### Risk (crypto)

```yaml
risk.crypto:
  hard_stop_pct: 6.0
  atr_stop_mult: 2.5
  trailing_stop_pct: 4.5
  take_profit_pct: 5.0
  circuit_breaker_drawdown_pct: 8.0
  max_daily_loss_pct: 5.0
  vol_targeting:
    target_vol_pct: 4.0           # per-asset override (target only)

risk:                              # global level
  vol_targeting:
    enabled: true                  # master switch — read from risk.vol_targeting, not risk.crypto
    target_vol_pct: 4.0
    min_scale: 0.3
    max_scale: 1.5
```

### Execution

```yaml
execution:
  stop_exit_reentry_cooldown_minutes: 30
  stuck_blacklist_after: 3
  limit_orders.enabled: true    # equities only; crypto always market orders
```

### Data

```yaml
data:
  interval: 1m
  dynamic_symbols:
    max_symbols: 500
    universe: alpaca_active_all
```

---

## Docker Services

| Service | Purpose | Port | Memory |
|---------|---------|------|--------|
| `trader` | Main trading loop | 8001 | 2G |
| `api` | FastAPI config/health/web UI | 18081 | 512M |
| `market-cache` | Bulk symbol data fetch + Redis | — | 4G |
| `redis` | In-memory cache backend | 6379 | 1G |
| `ollama` | Local LLM (llama3.2:3b) | 11434 | — |
| `learner` | RL online training | — | 2G |
| `prometheus` | Metrics scrape + alerting rules | 9090 | — |
| `grafana` | Dashboards | 3002 | — |
| `alertmanager` | Alert routing (email) | 9094 | — |
| `healthwatch` | Service health monitor | 9105 | 512M |
| `tests-when-closed` | pytest + backtest when market closed | — | 2G |
| `daily-report` | Daily top-movers + email | — | 512M |
| `health-reporter` | Health summary reports | — | 256M |
| `data-pruner` | Prunes /data files >7 days | — | 128M |
| `calendar-updater` | Holiday calendar maintenance | — | — |
| `autoheal` | Auto-restarts unhealthy containers | — | — |
| `docker-socket-proxy` | Docker socket proxy for autoheal/healthwatch | — | — |

All services stay running 24/7 (crypto mode). `healthwatch.market_shutdown.mode: partial` with empty `stop_services` list.

---

## Operational Runbook

### Start the system

```bash
./scripts/compose_up.sh   # auto-detects GPU, generates Grafana dashboards
```

### Stop the system

```bash
docker compose down        # stops all containers, preserves volumes
docker compose down -v     # also removes volumes (data loss — use only for full reset)
```

To stop only the trader (positions stay open on broker side):
```bash
docker compose stop trader
```

### Check health

| What | How |
|------|-----|
| API health | `curl http://localhost:18081/health` |
| Prometheus metrics | `http://localhost:8001/metrics` |
| Grafana | `http://localhost:3002` |
| Container resources | `docker stats` |
| Trader logs | `docker logs -f fricktrade-trader-1` |
| Decision traces | `data/reports/decision_trace/trace_YYYY-MM-DD.jsonl` |
| Checkpoint | `data/checkpoints/trader.json` (updated every 60s) |

### Trader healthcheck

The trader container has a Docker healthcheck that verifies:
1. Prometheus metrics endpoint responds on :8001
2. Checkpoint file is less than 900s old

If stale: trading loop is frozen. Check `docker logs trader` for the cause.

### Common issues

**1. No trades happening**

Check decision traces for the most recent reasons:
```bash
docker exec fricktrade-trader-1 python3 -c "
import json
with open('/data/reports/decision_trace/$(date -u +%Y-%m-%d).jsonl', 'rb') as f:
    f.seek(0, 2); f.seek(max(0, f.tell() - 500000)); f.readline()
    reasons = {}
    for line in f:
        r = json.loads(line)
        reasons[r.get('reason','')] = reasons.get(r.get('reason',''), 0) + 1
    for k,v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f'{v:6d} {k}')
"
```

Common causes:
- `strategies_hold` — market conditions don't meet strategy thresholds. Expected in calm markets.
- `pending_sell_covers_position` — deadlocked. Restart trader to clear. If persistent, check broker connectivity.
- `below_min_conviction` — signals too weak. Check if SSCM is set.

**2. `pending_sell_covers_position` deadlock**

Stale pending sells expire after 5 min automatically. If still stuck: restart the trader container. Root cause is usually broker API not responding (order response never arrives).

**3. Binance dust cycling**

~30 sub-LOT_SIZE positions in Binance demo. `floors_to_zero` warnings every cycle. Normal — 8h backoff suppresses them. Dust conversion fails on demo API (`APIError -2008`). Ignore.

**4. Ollama not responding**

```bash
docker logs fricktrade-ollama-1 --since 5m
docker exec fricktrade-ollama-1 curl -s http://localhost:11434/api/tags | python3 -m json.tool
```

Ollama starts AFTER trader is healthy. If trader was restarted, Ollama may not have started. Force: `docker restart fricktrade-ollama-1`.

**5. market-cache using excessive RAM**

Check `docker stats fricktrade-market-cache-1`. If >3.5G: restart the service. `MALLOC_TRIM_THRESHOLD_=65536` helps but can't recover badly fragmented heap.

**6. Kill switch fired (strategy disabled)**

If checkpoint shows `disabled_strategies: ["crypto_momentum"]`:
```bash
docker exec fricktrade-trader-1 python3 -c "
import json
with open('/data/checkpoints/trader.json') as f: cp = json.load(f)
for bname, bs in cp.get('payload',{}).get('broker_states',{}).items():
    ds = bs.get('disabled_strategies', [])
    if ds: print(f'{bname}: {ds}')
"
```
Fix: patch checkpoint and restart. Kill switch is disabled (`enabled: false`) so it shouldn't self-arm.

**7. Rebuild and restart**

```bash
docker compose build trader && docker compose up -d --force-recreate trader
# Also rebuild tests-when-closed (shares same codebase):
docker compose build tests-when-closed && docker compose up -d --force-recreate tests-when-closed
```

Always push to both remotes:
```bash
git push origin v3.0 && git push github v3.0
```

---

## Monitoring

### Endpoints

| Endpoint | URL |
|----------|-----|
| Trader Prometheus metrics | `http://localhost:8001/metrics` |
| Healthwatch metrics | `http://localhost:9105/metrics` |
| API health | `http://localhost:18081/health` |
| Grafana | `http://localhost:3002` |
| Prometheus UI | `http://localhost:9090` |

### Key Prometheus metrics

| Metric | Labels | What it measures |
|--------|--------|-----------------|
| `trades_total` | symbol, side | Total trades executed |
| `pnl_percent` | — | Current P&L percent |
| `drawdown_percent` | — | Current drawdown |
| `entry_ranking_score` | broker, symbol | Per-symbol opportunity ranking (0-1) |
| `market_sentiment_score` | broker | Aggregate Ollama sentiment (-1 to +1) |
| `strategy_win_rate` | strategy | Rolling win rate |
| `strategy_sharpe_ratio` | strategy, asset_class | Rolling 30-day Sharpe |
| `strategy_profit_factor` | strategy, asset_class | Gross win / gross loss |
| `decision_latency_seconds` | symbol | Time per decision |
| `broker_request_latency_seconds` | broker, method | Broker API latency |
| `orders_skipped_total` | symbol, side, reason | Orders blocked by safety checks |

### Grafana dashboards

Three account-specific dashboards (Alpaca Higher, Alpaca Realistic, Binance) generated from `_template_account.json.template`. Each includes:
- Broker status, connectivity, API errors
- Account equity, cash, buying power
- Position quantities and values
- Signal metrics (returns, volume, runup, drawdown)
- **Entry ranking scores** (bar gauge — higher = better opportunity)
- **Market sentiment** (gauge — -1 bearish to +1 bullish)

Plus: `fricktrade.json` (overview), `fricktrade_performance.json`, `fricktrade_latency.json`, `fricktrade_alerts.json`.

### Decision traces

JSON Lines format: `data/reports/decision_trace/trace_YYYY-MM-DD.jsonl`. Each line is a decision record with:
- `symbol`, `ts`, `broker`, `action`, `decision`, `reason`, `stage`
- `signals` — per-strategy signal array with action and confidence
- `effective_weights`, `winning_score`
- `portfolio` — cash, equity, exposure
- `position_exit_reason` — for exits

**Note:** `tests-when-closed` writes to the same trace file. Filter by timestamp to distinguish.

---

## Pending Guards (Thread-Safe)

| Guard | Dict Key | TTL | Prevents |
|-------|----------|-----|---------|
| `_pending_buy_symbols` | `(broker, symbol)` | 900s | Re-buying same symbol while order in flight |
| `_pending_sell_qty` | `(broker, symbol)` | Until fill/terminal; expires after 5 min if no response | Duplicate sell order stacking |
| `_pending_notional` | broker | Until fill/terminal | Leverage race during notional reserve |
| `_stuck_cooldown` | `(broker, symbol)` | 15 min | Re-entry after timed-out buy |
| `_exit_backoff_until` | `(broker, symbol)` | 1/2/4/8/15 min cap | Repeated exit attempt failures |

### Two-Phase Dispatch

`_run_symbol_batch()` splits symbols into holders and non-holders; holders processed first with `wait()` barrier. Guarantees exit orders complete before entry orders start. Non-holders are ranked by entry_ranking before processing.

---

## Common Pitfalls

- **Sell qty**: use `math.floor()`, never `round()` — float64 broker decimals can cause `round()` to exceed actual held qty
- **`record_pnl()` vs `update_daily_loss()`**: `record_pnl()` accumulates deltas; `update_daily_loss()` sets absolute day P&L — wrong one causes -543% false loss
- **`order_queue.enqueue()` sentinel**: returns `"queued"` (truthy) when accepted but not started
- **`_check_position_exit` crypto stops**: reads from `risk.crypto.*` for `/` symbols
- **`is_market_open()` 24/7**: returns True always with Crypto venue; use `is_venue_open('NYSE')` for equity-only
- **Binance `asset_class`**: check with `"crypto" in asset_class`, not `== "crypto"`
- **`/USDT` symbols in Alpaca batches**: must be filtered at all data-path entry points
- **Trailing stop override**: momentum returns `trailing_stop_pct: 2.0` in buy signal dict. Trader uses per-signal value if present, else falls back to config global 4.5%. So momentum entries have a 2.0% stop, not 4.5%.
- **Per-symbol sentiment in strategies**: Both strategies read `market_state["llm_sentiment"]`. When debugging unexpected confidence values, check if Ollama returned a non-neutral score.
- **Vol targeting is live**: `_size_order` applies vol_targeting scale. Positions in high-vol periods will be smaller. Scale range: [0.3, 1.5].
- **`tests-when-closed` shares codebase**: Must be rebuilt when trader code changes. Writes to the same decision trace file.

---

## Extending the Codebase

- New strategies: extend `app/strategies/base.py`; wire in `app/agents/trader.py`; add to `strategy.names` in config; write test.
- New brokers: implement `app/brokers/base.py`; add iterator to `app/brokers/config_utils.py`; wire in `app/main.py:_build_broker()`.
- New metrics: define in `app/monitoring/metrics.py`; record at call site.
- New LLM modules: use `LLMClient` from `app/llm/client.py`; add to `_init_llm()` in trader.py; add config section under `llm:`.

---

## Testing

193 tests (16 skip without tensorflow/prometheus). Run:

```bash
docker compose run --rm trader pytest tests/ -v
# Or locally (requires gymnasium for test_trader_sizing):
python3 -m pytest tests/ --ignore=tests/test_trader_sizing.py -q
```

---

## Backtest

```yaml
backtest:
  symbols: 20 crypto pairs (BTC, ETH, SOL, DOGE, ADA, ... all /USD)
  date_range: 2025-12-21 to 2026-03-21
  costs: commission 0.05%, slippage 3bps, spread 5bps
  initial_cash: $10,000
```

```bash
# Download data first
python3 scripts/download_crypto_backtest_data.py
# Run backtest
docker compose run --rm -T trader python3 -m app.main backtest
# Walk-forward
python3 scripts/benchmark_runner.py --config config/config.yaml --walk-forward
```

Validated: MR bb_period=50 → +1.0% avg return, 96% win rate over 12 days (Bonferroni CI excludes zero).

---

## History

Major milestones (newest first):

- **Per-symbol Ollama sentiment, SSCM=0.6, Grafana ranking (2026-04-05, commit 39c6f2d):** Per-symbol sentiment via Ollama llama3.2:3b with crypto-specific prompt. Both strategies read `llm_sentiment` for confidence modification. Aggregate sentiment gate in entry ranking. `single_sided_conviction_multiplier: 0.6` deployed (effective threshold 0.15 for single-sided signals). Entry ranking + market sentiment Grafana panels added to all account dashboards. News executor bumped to 2 workers.

- **Stale pending_sell_qty expiry (2026-04-05, commit ed2705c):** If order response never arrives (broker API down), `pending_sell_qty` stays set forever, blocking all sells. Added timestamp tracking + 5-min expiry sweep. Also: clean pop on zero (prevents residual float accumulation).

- **Ollama keep_services fix (2026-04-04, commit 9bfbeb7):** Ollama was in `stop_services` list. In crypto-only mode, equity market is always "closed" → Ollama killed every healthwatch cycle. Moved to `keep_services`.

- **pending_sell_qty deadlock fix (2026-04-04, commit ddb9757):** Two bugs caused all 33 symbols to be blocked for a week: (1) `close_position` never cleared `_pending_sell_qty` (only order queue did), (2) `floors_to_zero` kept it permanently set.

- **Strategic reset — crypto-only, 3→2 strategies (2026-03-21, commit b1bf267):** System was losing money consistently. Disabled equity trading, reduced to 2 active strategies (MR + momentum), killed meta-orchestrators, disabled guard rails. Switched to weighted combine mode with min_conviction 0.25.

- **Round 5-7 safety fixes (2026-03-09–15):** 38 fixes across all layers. See git log for details.

- **Binance broker; crypto 24/7 (2026-03-01–06):** `BinanceBroker` (Spot + Futures demo). Fractional trading. Crypto always market orders. Per-symbol circuit breaker.

- **v3.0 foundation (2026-02-01–13):** Extracted modules. 19 concurrency fixes. Two-phase dispatch. Pending notional race prevention.
