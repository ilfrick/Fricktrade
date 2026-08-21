<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

## Fricktrade Operational Guide

This is the **single authoritative document** for understanding and operating Fricktrade. Files in `docs/` predate the strategic reset (Mar 21) and should be treated as archived planning material.

**System**: v3.0, crypto-only, paper trading. Alpaca paper (2 accounts) + Binance Spot demo.

> ## 🛑 STATUS 2026-08-20: SHUT DOWN. TRADING RETIRED.
>
> The full stack was stopped with `docker compose down` on 2026-08-20. This was a
> **deliberate decision, not an outage** — see [Why trading was retired](#why-trading-was-retired-2026-08-20).
>
> Three independent research avenues were measured and all failed **for crypto**:
> price-direction alpha, cross-sectional momentum/reversal, and funding-rate carry.
> Across 3 years and both market regimes the strategy has no out-of-sample edge,
> and its gross edge is smaller than its own transaction costs.
>
> **Equities were tested too and also fail** on a properly sized universe. A 1-day
> reversal signal is genuinely present (t=4.55) and equity costs are 3.5x lower than
> crypto's, but realised expectancy is ~0 after stops and costs — hold-out −0.012%.
> An earlier 122-symbol test that looked promising turned out to be small-sample
> noise, not survivorship bias. See
> [Equities: measured properly](#equities-measured-properly-the-edge-did-not-replicate-2026-08-20).
>
> The repository is retained as **engineering infrastructure and a research
> platform**, not as a live trading system. If you restart it, the
> [alpha gate](#alpha-gate--flat-unless-proven-otherwise) will refuse to open
> positions until a configuration passes `scripts/alpha_validation.py holdout`.
> That default is intentional. Do not disable it to "just see what happens".
>
> Paper accounts as left: Alpaca Realistic $246.34 (−1.5% lifetime),
> Alpaca Higher $2,514.96 (−3.3%), both effectively all-cash plus ~30 untradeable
> dust positions. Nothing was ever funded with real money.

---

## Current System State

| Parameter | Value |
|-----------|-------|
| Active strategies | `crypto_mean_reversion` **only** (weight 1.0) — `crypto_momentum` disabled 2026-04-24 after AAVE/USD churn |
| Combine mode | `weighted` — weighted sum of buy confidences vs min_conviction |
| min_conviction | 0.25 (effective 0.15 when single-sided via SSCM=0.6) |
| Data interval | 1m bars |
| Brokers | Alpaca paper (2 accounts, `crypto_only`), Binance Spot demo |
| Trailing stop (crypto) | 4.5% global; 2.0% for momentum entries (momentum inactive) |
| Vol targeting | Enabled, target 4% **daily**, scale [0.3, 1.5] |
| Macro regime | **Disabled** (`llm.macro_regime.enabled: false`) |
| Per-symbol sentiment | Enabled (Ollama llama3.2:3b, 900s cache) |
| Aggregate sentiment | Enabled (Ollama, every 15 min) |
| Portfolio rebalance | **Disabled** 2026-08-19 — `_maybe_rebalance` called `check_rebalance()` with a mismatched signature; the TypeError was swallowed every cycle so it never did anything |
| Disabled | Tactical/strategic meta-orchestrators, portfolio orchestrator, RL policy, guardrail, VaR, exposure_caps, signal_bias_guard, kill_switch, symbols_filter, rebalance |

---

## Alpha Status (2026-08-19)

A full audit measured whether the system has any edge. It does not. Evidence,
all reproducible via `scripts/alpha_validation.py` against `/data/*_USD_1m.csv`
(20 symbols, 1m bars, 2025-12-22 → 2026-03-22):

| Test | Result |
|------|--------|
| Signal predictive value | Live entry (RSI-70 < 30) is **anti-predictive**: −0.20% edge at 15m (t=−4.0) vs baseline. It buys assets that keep falling. |
| Walk-forward, 6 folds | Mean net expectancy **−0.47%/trade**; only 2/6 folds positive. |
| Train / hold-out | train −0.229%/trade, hold-out **−1.083%/trade**. |
| Parameter grid (36 configs) | **0 of 36** had positive hold-out expectancy. Best train config (+0.241) went to −0.654 out-of-sample. |
| Gross vs net | Best gross edge ≈ **+0.17%/trade**; round-trip cost ≈ **0.21%**. The edge is smaller than the friction. |

### Confirmed across three years and both regimes (2026-08-19, second pass)

The results above came from a single 3-month bear window, so the test was repeated
on **3 years of hourly bars** (2023-08 → 2026-08, `/data/history_1h`) covering a
full bull market (+53%, +61% half-years through 2024) and the 2025–26 bear
(−24%, −42%, −27%). Verdict is unchanged — hold-out fails on every metric:
mean **−1.75%**, median **−2.17%**, 1%-trimmed **−1.93%**, win rate 31.7%, and
only 4 of 20 symbols profitable.

Two traps this second pass exposed, both now guarded against in the harness:

- **Bar-interval units.** The harness originally assumed 1m bars, so on hourly data
  its "480m" horizon silently became 20 days and its 3-day max-hold became 180
  days. That reported a bull market's beta as strategy edge (+34%/trade). Every
  time parameter is now expressed in minutes/hours and converted using the
  detected bar interval. This is the same per-bar-vs-per-day unit class of bug
  that was fixed in `trader.py` — it is the most common way to get a confidently
  wrong answer here.
- **Liquidation-wick outliers.** A single hour — **2025-10-10 21:00**, the October
  2025 liquidation cascade — printed +190% (DOT), +162% (ONDO), +150% (FIL)
  one-hour rebounds. It alone produced an apparent edge: mean forward return
  +9.68% versus a **median of +0.87%**. Those prints are not fillable in size. The
  harness now reports median and 1%-trimmed expectancy alongside the mean, and
  the hold-out gate requires **all three** to be positive.

**Why the system never made money — three compounding causes:**

1. **It barely trades.** `rsi_period: 70` on 1m bars is so smoothed that RSI's 1st
   percentile is ~37 across every symbol, so the `rsi_oversold: 30` gate sits
   *below the 1st percentile* and fires on 0.049% of bars (1,264 of 2.59M). Live,
   that is zero orders in 4 consecutive days.
2. **The exit ladder is inverted.** Partial TP at +1% and the MR SMA-exit at +0.5%
   cap winners at ~+0.55%, while losers run to the −5% stop. An 84% win rate still
   nets −0.26%/trade. In a downtrend the SMA falls to meet the price, so the
   "reversion" exit fires at its floor instead of on a real bounce.
3. **Costs exceed the edge.** 0.05% commission + 3bps slippage + 2.5bps half-spread
   per side ≈ 0.21% per round trip, against a gross edge of at most ~0.17%.

### Funding-rate carry — measured, and also too small (2026-08-20)

Carry was the last avenue with a credible prior: it is delta-neutral (long spot,
short perp), so it sidesteps the long-only-in-a-bear-market problem that dominated
every directional test, and the return is structural rather than predictive.
Measured from Binance perp funding history, last 2 years:

| Symbol | mean / 8h | annualised | % periods positive |
|--------|-----------|-----------|--------------------|
| BTCUSDT | 0.0045% | **5.1%** | 81.8% |
| ETHUSDT | 0.0044% | 5.0% | 80.2% |
| SOLUSDT | 0.0011% | 1.2% | 60.8% |
| XRPUSDT | 0.0030% | 3.4% | 67.3% |
| DOGEUSDT | 0.0045% | 5.0% | 73.4% |
| **average** | 0.0035% | **3.9% gross** | — |

That 3.9% is *before* fees on two legs, spot/perp spread, and margin drag —
realistically 2–3% net. On this account size (~$2,850) that is roughly $70–85 a
year in exchange for liquidation risk and a two-legged position manager running
continuously. Short-term government paper pays more with none of the operational
risk. **Rejected on size, not on validity** — the effect is real and would be worth
revisiting at a materially larger account, or if funding spikes (it exceeded 30%
annualised during the 2024 bull run).

Reproduce: `fapi.binance.com/fapi/v1/fundingRate` is public, no auth required.

**Ranked options, by expected value (none yet validated — validate before deploying):**

1. **Cut transaction costs.** Crypto is hardcoded to market orders in
   `trader.run_once` (the limit-order upgrade is skipped when `"/" in symbol`).
   Binance maker fees are 0.02% vs 0.05% taker, and resting bids avoid crossing the
   spread — plausibly ~0.15%/round-trip saved, which is the same order as the entire
   deficit. Caveat: maker orders may not fill, and non-fills are adversely selected.
2. **Trade less, at higher conviction.** Every sweep showed loosening entries
   increases trade count and *increases* total loss.
3. **Find a different alpha source.** No configuration of this dip-buy family
   cleared costs in any window tested.
4. **Do not allocate capital** to any configuration until
   `scripts/alpha_validation.py holdout` reports a pass.

The bear market matters for context but does not excuse the result: buy-and-hold
across the 20 symbols averaged **−29%** over the same period, so cash beat both the
strategy and the market. Being flat was better than trading.

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
    crypto_mean_reversion: 1.0    # sole active strategy
    # crypto_momentum: disabled 2026-04-24 (AAVE/USD churn)
  min_conviction: 0.25       # minimum weighted score to trigger buy
  single_sided_conviction_multiplier: 0.6  # reduces threshold when only one side signals
  names:
    - crypto_mean_reversion
```

Note on `crypto_momentum`: in weighted mode it was effectively dead even before
being disabled — a 2026-03 backtest logged 137,508 momentum buy signals and **0**
executions, because its typical confidence (~0.268) times its 0.40 weight (0.107)
never cleared the 0.15 single-sided threshold. It contributed churn, not trades.

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

The stack is currently **shut down by choice** ([why](#why-trading-was-retired-2026-08-20)).
Starting it does not resume trading: the alpha gate blocks new entries until a
configuration passes the hold-out. It is safe to bring up for research or to look
at dashboards.

```bash
./scripts/compose_up.sh   # auto-detects GPU, generates Grafana dashboards

# research only, without the trading loop:
docker compose up -d prometheus grafana
```

Note: `compose_up.sh` regenerates `grafana/provisioning/dashboards/account_*.json`
from `.env`. Those files are gitignored but force-added, so a regeneration can show
up as tracked modifications or deletions — check `git status` before committing
after running it.

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
| `_pending_buy_symbols` | `(broker, symbol)` | 900s | Re-buying same symbol while order in flight. Kept on ALL terminal statuses (completed/rejected/canceled); only released on `timed_out`. |
| `recent_position` guard | `broker_state.position_state` | 900s from `opened_at` | Secondary dedup: blocks re-buys if position recently opened |
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
- **Dust position filter**: `_strip_dust_positions()` removes sub-min_notional positions from portfolio before batch dispatch. Two additional notional guards inside `_check_position_exit` and strategy-sell path catch any dust that leaks through (e.g. stale last_prices).
- **Pending-buy dedup release**: Only `timed_out` releases `_pending_buy_symbols`. Releasing on `rejected` caused retry loops — 14 duplicate HYPE/USD buys in 17 min on Binance.
- **Shared-account position dedup**: `BrokerRouter.get_positions()` deduplicates by `api_account_id()` — brokers sharing the same underlying account get split qty. Currently a no-op (Alpaca accounts have separate API keys and account numbers: PA3SW9ZCUYC5 vs PA38SSSDFBVE).
- **Exit backoff escalation**: `_should_skip_exit()` must NOT reset `_exit_fail_counts` on backoff expiry. Only `_clear_exit_backoff()` (on successful exit) should clear it. Resetting on expiry trapped sells in an infinite attempt=1 / 1-min loop.
- **Sell qty cross-check**: Before enqueuing a sell-to-close, qty is validated against `broker_state.position_state`. If portfolio says 34M but position_state says 15M, qty is capped. Guards against transient portfolio snapshot corruption (observed 2026-04-22T00:05: all brokers received Binance's portfolio data for one cycle).
- **Portfolio contamination auto-fix**: `_get_portfolio_snapshot()` cross-validates per-broker positions against `position_state`. If a broker's portfolio qty is >2x its position_state AND matches another broker's position_state (contamination fingerprint), qty is corrected before reaching the trading pipeline. Fixes exposure/sizing/risk — not just sells. (Observed: AAVE/USD on alpaca:Higher showed Binance's 1.16 instead of 0.39, 2026-04-23.)
- **Portfolio cross-contamination diagnostic**: `_run_all_batches_clustered` logs a warning if two broker batches have identical cash values — signals a snapshot corruption event.
- **Cold-restart warmup grace** (`execution.warmup_grace_minutes: 5`): Suppresses non-safety sells (alpha_decay, time_exit, peak_detection, opportunity_cost, strategy-driven sells) for 5 min after cold start. Safety exits (hard_stop, atr_stop, trailing_stop, take_profit) still fire. Prevents the sell storm where all positions trigger exits simultaneously on cycle 1. Market data is also pre-warmed before the first loop iteration so indicators have full bar history from the start.

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

## Equities: measured properly, the edge did not replicate (2026-08-20)

**This section corrects an earlier conclusion in this same document.** An initial
122-symbol test looked promising (+0.213%/trade walk-forward, +0.577% hold-out).
On a larger, survivorship-free universe it did not hold up. Recorded in full
because the way it failed is more instructive than the result.

### What was tested

A 1-day short-term reversal: buy an oversold dip (RSI14 < 30, below the lower
20-day Bollinger band, ≥3% drop in 5 days), hold one day, 8% disaster stop, no
other exits. Daily bars, 2016-08 → 2026-08. Equity round-trip cost is **0.06%**
(commission-free, ~2bps slippage + 2bps spread) versus **0.21%** in crypto.

### Two datasets, two answers

| dataset | symbols | walk-forward | hold-out |
|---------|---------|--------------|----------|
| `data/equity_1d/` (first pass, 400 sampled) | 122 | +0.213% | **+0.577%** |
| `data/equity_1d_sf/` (1,500 sampled, incl. delisted) | 283 | +0.088% | **−0.012%** |

The larger sample is the one to believe: 1,145 hold-out trades versus 546, drawn
from a wider draw of the same pool. **The first result was small-sample noise.**

### Survivorship bias turned out NOT to be the culprit

This was the risk flagged as most serious, so it was measured directly rather than
assumed. Alpaca serves full daily history for delisted names — verified against
SIVB (ends 2023-03-09, the SVB collapse), FRC (ends 2023-04-28 at $3.51), XLNX,
ATVI, CXO — so `--include-delisted` builds a genuinely survivorship-free universe.
Of 283 symbols, 29 (10.2%) delisted during the sample.

Controlled comparison **within the same download**:

| universe | trades | net E% | median% | WR% |
|----------|--------|--------|---------|-----|
| survivors only (biased) | 4,749 | +0.097 | +0.079 | 51.1 |
| delisted only | 270 | +0.024 | +0.134 | 50.7 |
| all (survivorship-free) | 5,019 | +0.093 | +0.080 | 51.1 |

Survivorship inflates expectancy by **+0.004pp/trade** — negligible. The reason is
the liquidity filter: requiring ≥$5M median daily dollar volume keeps established
companies, whose delistings are mostly acquisitions (often at a premium) rather
than bankruptcies. Delisted names underperformed only slightly and are 5% of trades.

So the prior worry was wrong, and worth recording as such: the danger was not
survivorship, it was **sample size**.

### Where that leaves equities

The 1-day signal itself is real — +0.29% edge at one day (t=4.55 on 7,321 signals),
decaying to nothing by day two. But converting it into trades with a stop and
realistic costs yields ~0: walk-forward +0.088% against 0.06% costs, and the
1%-trimmed expectancy is **negative in 4 of 6 folds**, meaning the positive mean
rides on a few large winners. Hold-out is −0.012%. **The gate correctly refuses it.**

Equities remain a better environment than crypto — costs are 3.5x lower and the
signal is genuinely present — but this implementation does not clear the bar. Any
future attempt should start from a universe of several hundred names, not one
hundred, because that is what separated the two answers above.

## Why trading was retired (2026-08-20)

A decision record, so this does not get silently re-litigated.

**The decision:** stop live trading. Keep the repository as engineering
infrastructure and a research platform. Do not fund it with real money.

**Why.** Three independent avenues were measured, and all three failed:

1. **Price-direction alpha** (the shipped strategy). Across 3 years and both
   regimes: hold-out mean −1.75%/trade, median −2.17%, trimmed −1.93%, win rate
   31.7%, 4/20 symbols profitable. 0 of 36 parameter configurations were positive
   out-of-sample.
2. **Cross-sectional selection.** Momentum and reversal at 24/72/168h, and
   trend-following above a moving average, all lost −59% to −98% on train while
   cash returned 0% and buy-and-hold lost −28%. Concentration made it worse.
   Symbol *selection* adds nothing.
3. **Funding-rate carry.** Real, but 3.9% gross / 2–3% net — below the threshold
   that justifies liquidation risk at this account size (see above).

Meanwhile every trade pays ~0.21% round-trip against a gross edge that never
reliably exceeded it, and per-trade expectancy correlates with market direction at
only −0.07 — so this was never a market-timing problem that a regime filter could
fix. The individual decisions carry no information.

**What made this hard to see for so long.** The system looked like it was working.
An 84% win rate is extremely persuasive, and it was real — it just came with
average wins of +0.55% against average losses of −4.5%, which is negative
expectancy. Win rate was the illusion; expectancy was the truth. Compounding that,
several instruments were broken in ways that hid the problem: the ATR stop was dead
code, the peak-detection exit never ran, vol targeting was pinned at 1.5×, and the
risk caps never bound. The dashboards were green because they measured activity,
not edge.

**Two methodological traps found while doing this work** (both now guarded in the
harness, both worth remembering for any future attempt):

- *Unit errors between per-bar and per-day quantities.* This produced a fake
  +34%/trade result and, separately, four of the seven production bugs. Any time a
  parameter is expressed in "bars", confirm what a bar actually is.
- *Rare untradeable prints dominating a mean.* One hour — the 2025-10-10
  liquidation cascade — moved mean forward return to +9.68% against a median of
  +0.87%. Always read the median and trimmed mean next to the mean.

**What would reopen the question.** A hypothesis that passes
`scripts/alpha_validation.py holdout` on multi-regime data — mean, median and
trimmed expectancy all positive out-of-sample. Nothing less. If you are tempted to
relax that bar, re-read the row above about candidate #10, which passed a
Bonferroni test and still lost money live.

**Where an edge would more plausibly live**, if resumed: not in 1-minute bars on
the 20 most liquid pairs, which is the most heavily arbitraged data available.
More likely in structurally awkward places — newly-listed or illiquid pairs, or
market making where you earn the spread instead of predicting direction. Both are
different systems, not parameter changes.

## Alpha gate — flat unless proven otherwise

`strategy.alpha_gate.enabled: true` (default) blocks **new entries** unless the
running configuration has a recorded hold-out pass. **Exits are never blocked** —
an unvalidated system must still be able to close what it already holds.

This inverts the previous default. The system used to treat "find a reason to buy"
as its resting state and traded continuously on a configuration with no measured
edge; cash outperformed every variant tested.

```bash
# from the host — data dir, pattern and config are all auto-detected
python3 scripts/alpha_validation.py holdout --record

# or pick a specific dataset (repo-relative paths work)
python3 scripts/alpha_validation.py holdout --record \
    --data-dir data/history_1h --pattern '*_USD_1h.csv'

# inside the container it resolves /data automatically
docker compose run --rm -T trader python3 scripts/alpha_validation.py holdout --record

# inspect the current decision
docker exec fricktrade-trader-1 python3 -c \
  "from app.utils.config import load_config; from app.risk import alpha_gate; \
   print(alpha_gate.check_entries(load_config('/app/config/config.yaml')))"
```

Behaviour:

| Situation | Entries |
|-----------|---------|
| No record file | blocked (`no_validation_record`) — fails closed |
| Record says `passed: false` | blocked (`validation_failed`) |
| Strategy/risk params changed since validation | blocked (`config_changed_since_validation`) |
| Record passes and fingerprint matches | allowed (`validated`) |
| `alpha_gate.enabled: false` | allowed (`gate_disabled`) — explicit opt-out |

The fingerprint covers only parameters that change trading behaviour (strategy
names/weights/params, stops, take-profits, vol targeting, data interval), so
editing logging or credentials does not invalidate a pass, while changing a stop
does. Passing requires mean, median **and** 1%-trimmed hold-out expectancy all
positive — one metric alone is too easy to satisfy with a few untradeable wicks.

**Current state: blocked** (`validation_failed`; hold-out −1.75%/trade).

## Alpha validation (run before any strategy change)

`scripts/alpha_validation.py` answers whether a configuration has edge, in
increasing order of strictness. It reports **net expectancy per trade** after
realistic costs, because total return is dominated by sizing and market drift and
hides whether each decision adds value.

```bash
# does the entry signal predict anything at all?
docker compose run --rm -T trader python3 scripts/alpha_validation.py signal

# sequential walk-forward — a real edge is positive in most folds, not on average
docker compose run --rm -T trader python3 scripts/alpha_validation.py walk --folds 6

# one-shot hold-out on the untouched last 25%
docker compose run --rm -T trader python3 scripts/alpha_validation.py holdout

# try a candidate configuration
python3 scripts/alpha_validation.py all --data-dir ./data \
    --rsi-period 30 --rsi-oversold 25 --hard-stop 3
```

Rules that keep this honest:
- Select parameters on train/walk-forward only; touch the hold-out **once**.
- An edge below the round-trip cost (~0.21%) is not tradeable no matter its t-stat.
- t-stats printed are optimistic: signals overlap in time and crypto symbols are
  highly correlated, so effective sample size is far below the raw signal count.
- Compare **median and 1%-trimmed** expectancy against the mean. A large gap means
  a few untradeable prints are carrying the result (see the 2025-10-10 wick above).
- Entry windows (`bb_period`, `rsi_period`, `drop_window`) are in **bars**, so their
  meaning changes with the dataset interval; the header line prints the detected
  bar size so you can sanity-check what you actually tested.

Data discovery: with no `--data-dir`/`--pattern` the script searches `/data`, then
`<repo>/data`, then `<repo>/data/history_1h`, and prints exactly which dataset it
used. If nothing matches it lists every dataset it can see with the flags to select
it. `--config` defaults to `<repo>/config/config.yaml`, so `--record` works with no
extra arguments; if the configured record path is not writable (i.e. `/data` from
the host) it falls back to `<repo>/data/reports/`, which is the same volume the
container sees.

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

- **Trading retired; stack shut down (2026-08-20):** `docker compose down`. Funding-rate
  carry measured at 3.9% gross / 2-3% net — the last credible avenue, rejected on size
  rather than validity. Combined with the failed price-direction and cross-sectional
  tests, the decision was taken to stop live trading and keep the repo as engineering
  infrastructure. Full reasoning in [Why trading was retired](#why-trading-was-retired-2026-08-20).

- **Alpha gate + multi-regime validation (2026-08-19):** Added
  `strategy.alpha_gate` (`app/risk/alpha_gate.py`) — new entries require a recorded
  hold-out pass matching the running config's fingerprint; exits are never blocked.
  Validation extended to 3 years of hourly bars covering bull and bear; verdict
  unchanged. Fixed two ways the harness itself could lie: bar-interval unit errors
  and liquidation-wick outliers dominating the mean.

- **Full audit + correctness fixes (2026-08-19):** Measured that the system has no
  out-of-sample edge (see [Alpha Status](#alpha-status-2026-08-19)) and fixed seven
  defects found while reading the full codebase:
  1. `compute_all_indicators()` never emitted `rsi` or `atr`. The ATR stop was dead
     code (it reads `indicators["atr"]`), and the peak-detection exit's RSI leg was
     permanently neutral at 50 — so the "detect peak and sell" behaviour never worked.
  2. ATR stop now scales per-bar ATR to a daily equivalent (`× sqrt(bars/day)`).
     Unscaled, 2.5 × ATR(14 × 1m) is a ~0.25% stop that would exit on noise.
  3. Vol targeting compared per-bar realized vol against a *daily* 4% target, so the
     scale was pinned at `max_scale` 1.5 for every symbol — a permanent 1.5× position
     multiplier that never shrank in volatility. Now daily-equivalent on both sides.
  4. `_enrich_signals` treated `atr_pct` (a percent) as a fraction, pinning `vol_mult`
     at its 0.7 floor for essentially all crypto. Now daily-equivalent.
  5. `RiskManager.check_crypto_exposure` read `market_value`, but portfolio snapshots
     store `value` — so existing positions counted as $0 and neither the crypto
     exposure cap nor the per-position cap ever bound. Now accepts both keys.
  6. `PerformanceTracker.record_trade` omitted `symbol`, so the per-asset-class
     Sharpe/profit-factor metrics were always computed over empty lists.
  7. `portfolio.rebalance` disabled — `_maybe_rebalance` called `check_rebalance()`
     with a mismatched signature; the TypeError was swallowed every cycle.
  Also added `scripts/alpha_validation.py` and `tests/test_alpha_fixes.py`, and
  corrected two stale tests that asserted deliberately-removed behaviour.

- **Per-symbol Ollama sentiment, SSCM=0.6, Grafana ranking (2026-04-05, commit 39c6f2d):** Per-symbol sentiment via Ollama llama3.2:3b with crypto-specific prompt. Both strategies read `llm_sentiment` for confidence modification. Aggregate sentiment gate in entry ranking. `single_sided_conviction_multiplier: 0.6` deployed (effective threshold 0.15 for single-sided signals). Entry ranking + market sentiment Grafana panels added to all account dashboards. News executor bumped to 2 workers.

- **Stale pending_sell_qty expiry (2026-04-05, commit ed2705c):** If order response never arrives (broker API down), `pending_sell_qty` stays set forever, blocking all sells. Added timestamp tracking + 5-min expiry sweep. Also: clean pop on zero (prevents residual float accumulation).

- **Ollama keep_services fix (2026-04-04, commit 9bfbeb7):** Ollama was in `stop_services` list. In crypto-only mode, equity market is always "closed" → Ollama killed every healthwatch cycle. Moved to `keep_services`.

- **pending_sell_qty deadlock fix (2026-04-04, commit ddb9757):** Two bugs caused all 33 symbols to be blocked for a week: (1) `close_position` never cleared `_pending_sell_qty` (only order queue did), (2) `floors_to_zero` kept it permanently set.

- **Strategic reset — crypto-only, 3→2 strategies (2026-03-21, commit b1bf267):** System was losing money consistently. Disabled equity trading, reduced to 2 active strategies (MR + momentum), killed meta-orchestrators, disabled guard rails. Switched to weighted combine mode with min_conviction 0.25.

- **Round 5-7 safety fixes (2026-03-09–15):** 38 fixes across all layers. See git log for details.

- **Binance broker; crypto 24/7 (2026-03-01–06):** `BinanceBroker` (Spot + Futures demo). Fractional trading. Crypto always market orders. Per-symbol circuit breaker.

- **v3.0 foundation (2026-02-01–13):** Extracted modules. 19 concurrency fixes. Two-phase dispatch. Pending notional race prevention.
