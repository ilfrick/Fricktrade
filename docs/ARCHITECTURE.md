<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Fricktrade v3.0 — Architecture

## Overview

Fricktrade is a multi-broker algorithmic trading system that trades US equities intraday and crypto 24/7. The system is built around a central `TradingAgent` that orchestrates market data, strategy signals, risk checks, and order execution across multiple broker accounts simultaneously.

The architecture separates concerns into discrete layers: data ingestion, signal generation, risk management, execution, and observability. Each layer is designed to fail independently without taking down the others.

---

## Component Map

```
┌─────────────────────────────────────────────────────────────┐
│                     Market Data Layer                        │
│  AlpacaMarketDataProvider  BinanceMarketDataProvider         │
│  HybridMarketDataProvider  MarketCacheService (Redis+file)   │
│  QuoteStream (disabled)    news_rss.py  alt_data.py          │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                     Symbol Manager                           │
│  load_universe() → AI filter → return ranker → 150 symbols  │
│  _build_symbol_batches() → per-broker routing                │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              Indicator Injection (~30 indicators)            │
│  compute_all_indicators() → market_state["indicators"]       │
│  EMA, RSI, ATR, BB, VWAP, Supertrend, Hurst, HMM regime...  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Strategy Layer                            │
│  9 strategies run in parallel per symbol                     │
│  Each returns {action, confidence, name}                     │
│  ConfidenceCalibrator transforms raw → calibrated prob       │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Vote Combiner                             │
│  strategy.combine: vote                                      │
│  Counts buy/sell/hold votes; majority wins                   │
│  Ties broken by rl_policy signal                             │
│  exit_vote_threshold: 2  buy_vote_threshold: 2               │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Risk Layer                                │
│  RiskManager: daily loss, max leverage, position size        │
│  Exposure caps: venue (NYSE 60%, Crypto 50%), sector         │
│  VaR/CVaR gating   Per-symbol circuit breaker (5%)          │
│  PDT guard   Min-hold gate   Pending notional locks          │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  Execution Engine                            │
│  _size_order() → half-Kelly × vol scale × time-of-day       │
│  SmartOrderRouter → TWAP/VWAP/POV or single market order     │
│  Limit order auto-upgrade (market → limit at mid+3bps)       │
│  Crypto: always single market order (no TWAP)                │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                     Order Queue                              │
│  FIFO min-heap ordered by earliest_at                        │
│  Pending sell guard (_pending_sell_qty)                      │
│  Pending buy dedup (_pending_buy_symbols, 900s TTL)          │
│  Pending notional reserve (_pending_notional)                │
│  Stuck-order timeout: cancel after 300s                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Broker Router                             │
│  _broker_map: {name → Broker}  _broker_states: {name → BS}  │
│  Parallel mode: multiple named Alpaca accounts               │
│  binance: Spot demo (crypto only)                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            │                             │
      Alpaca Paper                   Binance Demo
   (equities + crypto)             (crypto Spot only)
```

---

## Data Flow: One Trading Cycle

Every cycle (approximately every 60 seconds, or whenever a new 5-minute bar arrives):

1. **Market data fetch**: `AlpacaMarketDataProvider` pulls the latest 5-minute bars from the Redis market cache (populated by the `market-cache` service, which subscribes to Alpaca). Binance symbols use `BinanceMarketDataProvider` via `_HybridMarketDataProvider`.

2. **Symbol resolution**: `SymbolManager.refresh_dynamic_symbols()` queries Alpaca for all active assets, applies the AI filter and return ranker, and caps to 150 symbols. Position symbols are always included regardless of filter results.

3. **Venue gating**: Equity symbols are only processed when NYSE or Nasdaq is open. Crypto symbols (containing `/`) are processed 24/7.

4. **Indicator injection**: `compute_all_indicators()` builds ~30 technical indicators from price/volume history and stores them in `market_state["indicators"]`.

5. **Strategy execution**: All enabled strategies run `generate_signal(market_state)` and return `{action, confidence, name}`. The `ConfidenceCalibrator` applies per-strategy bin calibration to raw confidence values.

6. **Vote combination**: `_combine_signals()` tallies votes. `buy_vote_threshold: 2` means at least 2 strategies must vote buy to open a position. `exit_vote_threshold: 2` means at least 2 must vote sell to close.

7. **LLM overlay** (portfolio orchestrator disabled in `vote` mode): In the current `vote` mode, the portfolio LLM orchestrator is not instantiated. Tactical and strategic meta-orchestrators run in background threads independently.

8. **Risk checks**: `_check_execution_for_symbol()` applies the full guard stack: account blocked, kill switch, min hold minutes, pending sell guard, PDT, exposure caps, notional reserve.

9. **Position exit**: `_check_position_exit()` evaluates ATR stops, hard stop, trailing stop, take profit, partial take profit, alpha decay, regime-conditional time exit, and opportunity cost exit. Position exits run before new entries (two-phase dispatch).

10. **Order sizing**: `_size_order()` computes qty using half-Kelly × vol scale × time-of-day multiplier × account equity.

11. **Execution planning**: `_plan_execution()` calls `SmartOrderRouter` to select algo (TWAP/VWAP/POV). Crypto always bypasses TWAP and uses a single market order.

12. **Queue submission**: `order_queue.enqueue()` adds the order to the heap. The queue runs orders serially for each broker-symbol pair.

13. **Response processing**: `_flush_order_responses()` processes completed order results, releases pending guards, and records fills for P&L tracking and RL reward signals.

---

## Multi-Broker Architecture

### BrokerState

Each named broker account has a `BrokerState` dataclass holding:
- `risk`: a `RiskManager` instance
- `equity`, `cash`, `day_pnl_pct`, `day_deposits_baseline`
- `position_state`: `{symbol → {qty, avg_entry, opened_at, strategy, ...}}`
- `strategy_trades`: fill history for LLM win-rate stats
- Various pending and cooldown dicts

### Broker Routing

`execution.brokers.routing.mode: parallel` means all configured Alpaca accounts evaluate every symbol independently. Each account uses its own equity for position sizing (`buying_power_scaling: false`).

Symbol routing to Binance is handled by `build_symbols_by_broker()`, which assigns `/USDT`-quoted symbols to Binance and prevents them from appearing in Alpaca batches.

### Two-Phase Dispatch

`_run_symbol_batch()` separates symbols into:
1. **Holders** (currently have a position): processed first with `wait()` barrier
2. **Non-holders**: processed after all exit decisions complete

This guarantees that exit orders are submitted before entry orders, preventing the order queue from being backed up with entries while exits are pending.

---

## LLM Subsystem

### Current Active Components

| Component | File | Cadence | Cost |
|-----------|------|---------|------|
| Tactical Meta Orchestrator | `app/llm/tactical_meta_orchestrator.py` | Every 15 min | ~$0.02/day |
| Strategic Meta Orchestrator | `app/llm/meta_orchestrator.py` | Weekly (Sun 06 UTC) | Minimal |
| Post-Session Analyst | `app/llm/post_session.py` | Daily at session end | ~$0.10/session |
| Macro Regime Analyzer | `app/llm/macro_regime.py` | Every 4h | ~$0.01/call |
| Risk Interpreter | `app/llm/risk_interpreter.py` | On risk events | Low |
| Ollama Sentiment | `app/llm/ollama_sentiment.py` | Every 15 min | Free (local) |

### Currently Disabled

| Component | Reason |
|-----------|--------|
| Portfolio Orchestrator | `llm_orchestrator.mode: vote` — not instantiated |
| Per-symbol Sentiment | Vote mode skips `_enrich_signals()` entirely |
| Symbol Filter LLM | `llm.symbols_filter.enabled: false` |
| Learning Guardrail | `learning.guardrail.enabled: false` |

### Tactical Meta Orchestrator

Runs in a background thread every 15 minutes. Reads:
- Live strategy win rates
- Macro regime cache
- Last post-session report
- Order flow metrics (`_tmo_counters`)
- Per-broker health

Fires one Gemini 2.5 Flash call and proposes config changes. Changes are applied in two tiers:
1. Operational params (cooldowns, margins): applied immediately
2. Strategy weights, stop/TP: queued, applied after `apply_delay_minutes: 5`

All bounds are declared in `tactical_meta_orchestrator.bounds` in config. Changes logged to `/data/reports/meta_orch/changes.jsonl`.

### Strategic Meta Orchestrator

Runs weekly (Sunday 06:00 UTC) and on emergency triggers (3× D/F grades). Reads 7 days of post-session reports and the tactical changes log, then writes `strategic_baseline.json`. The tactical orchestrator uses these baselines as corridor constraints (±30% by default).

### LLM Budget

All LLM modules share a daily budget circuit breaker (`llm.cost.daily_budget_usd: 5.00`). All backends use Gemini 2.5 Flash by default (`$0.075/$0.30 per 1M tokens`). Gemini 2.5 Pro is intentionally avoided because it cannot disable thinking mode, making it expensive and slow for this use case.

---

## State Management

### Pending Guards

Several thread-safe dicts prevent duplicate or race-condition orders:

| Guard | Key | TTL | Purpose |
|-------|-----|-----|---------|
| `_pending_buy_symbols` | `(broker, symbol)` | 900s | Prevents re-buying same symbol while order in flight |
| `_pending_sell_qty` | `(broker, symbol)` | Until fill | Prevents stacking duplicate sell orders |
| `_pending_notional` | `broker` | Until fill | Reserves notional before order submission |
| `_stuck_cooldown` | `(broker, symbol)` | 15 min | Suppresses re-entry after timed-out buy |
| `_exit_backoff_until` | `(broker, symbol)` | 1/2/4/8/15 min | Exponential backoff after exit failures |

### Position State

`broker_state.position_state[symbol]` is populated each cycle by `performance.py:update_from_positions()` from the broker's live position data. Key fields:
- `qty`: current quantity
- `avg_entry`: cost basis (Binance uses last market price as fallback)
- `opened_at`: timestamp for min-hold and time-exit calculations
- `strategy`: name of the strategy that triggered the entry (for alpha decay exit)
- `peak_price`: highest price reached for trailing stop tracking

### Checkpoints

State is saved every 60 seconds to `/data/checkpoints/trader.json`. The healthwatch container monitors checkpoint age; if the loop becomes stale (> 900s), it restarts the trader.

---

## Threading Model

The main trading loop runs on the main thread. Background threads:

| Thread | Component | Purpose |
|--------|-----------|---------|
| `_meta_orch_thread` | TacticalMetaOrchestrator | 15-min LLM config tuning |
| `_strategic_orch_thread` | StrategicOrchestrator | Weekly baseline updates |
| `_reporting_thread` | TradingAgent | Prometheus metrics and checkpoint writes |
| `_learner_thread` | LiveRewardTracker | RL reward recording |
| News executor | `ThreadPoolExecutor` | Background news + Ollama calls (non-blocking) |

The order queue runs on the main thread. Each broker has its own `ThreadPoolExecutor` for parallel order submission, but responses are processed synchronously in `_flush_order_responses()`.

---

## Exit Logic Priority

Exits are evaluated in this priority order within `_check_position_exit()`:

1. Min-hold guard (suppress all exits for `min_hold_minutes: 15` after entry)
2. Hard stop (`risk.hard_stop_pct` / `risk.crypto.hard_stop_pct`)
3. Trailing stop (after price moves favorably past `trailing_stop_pct`)
4. Take profit (`take_profit_pct`)
5. Partial take profit (`partial_take_profit_pct`, sells `partial_take_profit_ratio` of position)
6. Alpha decay exit (entry strategy now signals sell, after `alpha_decay_exit.min_hold_minutes: 30`)
7. Regime-conditional time exit (Hurst-scaled hold time)
8. Opportunity cost exit (disabled by default)

Signal-path exits (from vote result) also go through the min-hold guard.

---

## Crypto-Specific Behavior

- Symbols containing `/` are crypto (e.g., `BTC/USD`, `ETH/USD`)
- Always use single market orders (no TWAP/limit order upgrade)
- Always fractional (`_is_fractional()` returns True for `/` symbols)
- Stops from `risk.crypto.*` section (harder defaults: `hard_stop_pct: 4.0%`)
- Exposure cap: 50% global, 95% for Binance (crypto-only broker)
- Alt-data gates: suppress longs when Fear & Greed < 20 or OI change < -5%
- `factor_model` strategy is excluded from crypto symbols (equity mean-reversion bias)
- `stat_arb_pairs` is disabled (long-only mode produces unhedged bets)

---

## Dust Position Handling

Positions with quantity < 1e-6 are considered dust. The system:
- Skips all position-exit checks for dust (commit `08c8979`)
- Does not call `close_position()` for dust (causes API errors)
- Sets 8h exit backoff if Binance rejects with `floors_to_zero` (sub-LOT_SIZE)
- Does not release `_pending_sell_qty` on `floors_to_zero` rejections (permanent block until restart)

---

## Key Design Decisions

**Vote mode instead of weights**: Weighted signal combination amplified noise when strategies had similar confidence values. Vote counting is more robust to calibration errors and prevents any one strategy from dominating.

**Crypto always market orders**: Limit orders at GTC at stale prices sit open indefinitely and block the queue. Market orders ensure immediate execution at the cost of minor slippage.

**Pending notional reserve**: Notional is reserved before enqueue (not after fill) to prevent leverage races in the gap between sizing and broker confirmation.

**Two-tier meta-orchestration**: The tactical orchestrator makes frequent small adjustments within safe bounds; the strategic orchestrator sets those bounds weekly. This prevents both over-fitting (tactical without bounds) and stagnation (no intra-week adaptation).

**Per-broker LLM awareness**: The portfolio orchestrator (when enabled) receives separate context per broker account, allowing it to recommend different actions for accounts with different equity, drawdown, and risk state.
