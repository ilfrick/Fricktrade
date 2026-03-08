<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Trading Agent Flow (v3.0)

This document describes the end-to-end flow for the trading agent as of v3.0.

---

## Main Loop (24/7 Crypto + Equity Market-Hours Gating)

```mermaid
flowchart TD
    A[Loop Start] --> B[Fetch Portfolio + Account Metrics]
    B --> C[Refresh News + Alt-Data\nfear/greed · OI · insider trades]
    C --> D[Ollama: Return Ranker Catalyst Scoring\nllama3.2:3b via HTTP]
    D --> E[Dynamic Symbol Refresh\nAI filter + universe → top 150]
    E --> F[Open Orders Manager\ncheck pending + stale GTC cleanup]
    F --> G{Equity Market Open?\nis_venue_open NYSE/Nasdaq}
    G -- No --> H[Crypto-Only Symbols\n'/' in symbol filter]
    G -- Yes --> I[All Active Symbols\nequity + crypto]
    H --> J[_run_symbol_batch]
    I --> J
    J --> K[Macro Regime + Weights\nGemini 4h TTL cache]
    K --> L[Per-Symbol Execution\n_check_execution_for_symbol\nparallel ThreadPool]
    L --> M[Metrics + Trace Flush\nPrometheus · decision trace · audit]
    M --> N[Wait Interval\n300s default]
    N --> A
```

---

## Per-Symbol Execution Flow

```mermaid
flowchart TD
    A[Symbol + Market Data\n5-min bars · OHLCV] --> B[compute_all_indicators\n~22 indicators injected into market_state]
    B --> C[LLM Sentiment\nGemini per-symbol 30min TTL]
    C --> D[Run All Strategies\n10 active strategies in parallel]
    D --> E[Signal Bias Guard\nblock directional override abuse]
    E --> F[_enrich_signals\nApply context multipliers to confidence\nregime × sentiment × vol × fear/greed/OI/RSI/insider]
    F --> G[LLM Strategy Orchestrator\nGemini per-symbol call]
    G --> H{LLM Decision\nor skip/fail?}
    H -- BUY/SELL --> I[Use LLM action + reduce_pct]
    H -- HOLD or fail --> J[_combine_signals\nweighted sum fallback]
    I --> K
    J --> K
    K[Guardrail Check\nIntradayMomentum confirm/veto] --> L{Guardrail\npasses?}
    L -- No --> HOLD[Hold]
    L -- Yes --> M[Account Flags + PDT Guard]
    M --> N[Risk Manager\nVaR · exposure · daily loss · leverage]
    N --> O{Risk OK?}
    O -- No --> SKIP[Skip / record_skip]
    O -- Yes --> P[_size_order\nHalf-Kelly · ATR stop · fractional]
    P --> Q[Order Execution\nsee Execution Flow]
```

---

## Signal Enrichment (_enrich_signals)

Adjusts each strategy's `confidence` in-place before the LLM sees the signals:

```mermaid
flowchart LR
    A[Raw Signal\naction + confidence] --> B[Regime Multiplier\ncrisis 0.6 · high_vol 0.8 · low_vol_trending 1.15]
    A --> C[LLM Sentiment Multiplier\n0.85 + 0.3 × score → 0.85–1.15]
    A --> D[Volatility Multiplier\nmax 0.7, 1.0 − ATR_excess × 10]
    B --> E[Combined Multiplier\nregime × sentiment × vol]
    C --> E
    D --> E
    E --> F{Crypto?}
    F -- Yes --> G[FearGreed Multiplier\n0.85 + 0.3 × fg/100\nOI ±10%]
    F -- No --> H[Equity Adjustments\nRSI extremes ±20%\nInsider net ±10–15%]
    G --> I[signal confidence × mult\nclamped 0–1\ncontext_mult stored in trace]
    H --> I
```

---

## LLM Strategy Orchestrator

```mermaid
flowchart TD
    A[Enriched Signals] --> B{Any signal confidence\n≥ min_signal_score 3%?}
    B -- No --> C[Cost Guard: skip LLM\nreturn hold → fallback to _combine_signals]
    B -- Yes --> D[Build Prompt ~400 tokens\nSymbol · Regime · RSI/ATR/VWAP/Hurst\nSentiment · FearGreed · OI\nStrategies+weights\nLast 5 P&L trades]
    D --> E[Gemini Flash API Call\ngemini-2.5-flash · temp=0.1]
    E --> F{JSON Response\naction · reduce_pct · reasoning}
    F -- parse OK --> G{action?}
    F -- parse fail --> H[Warning → return hold\nfallback to _combine_signals]
    G -- buy/sell --> I[Return action + reduce_pct\n+ best-matching strategy_name]
    G -- hold --> J[Return hold → fallback to _combine_signals]
    I --> K[Log: 'LLM orch → BUY/SELL symbol ...']
```

---

## Guardrail

```mermaid
flowchart TD
    A[Main Decision: buy/sell/hold] --> B{Guardrail Enabled?}
    B -- No --> Z[Pass Through]
    B -- Yes --> C{mode?}
    C -- confirm --> D[Run IntradayMomentum\nindependent instance]
    C -- veto --> D
    D --> E{Guardrail signal?}
    E -- confirm mode:\nboth agree → pass --> Z
    E -- confirm mode:\ndisagree → hold --> HOLD[Hold]
    E -- veto mode:\nguardrail=sell & main=buy → hold --> HOLD
    E -- veto mode:\notherwise → pass --> Z
    D --> F[entry_threshold_pct: 0.8%\n30-min lookback\nmomentum must exceed threshold]
```

---

## Execution Flow

```mermaid
flowchart TD
    A[Action + Qty + Symbol] --> B{Crypto?\n'/' in symbol}
    B -- Yes --> C[Market Order\nNo limit upgrade\nNo TWAP slicing\n24/7 · fractional · broker compat]
    B -- No --> D{Win prob ≥ high_urgency_threshold?}
    D -- Yes\nhigh conviction --> C
    D -- No --> E[Limit Order\nmid-price ± spread or 3bps offset]
    C --> F{Pending Buy\nfor broker+symbol?}
    E --> F
    F -- Yes --> SKIP[Skip: pending_order]
    F -- No --> G[Atomic Leverage Check\n_check_and_reserve_notional]
    G -- Over cap --> SKIP2[Skip: pending_leverage_cap]
    G -- OK --> H{Equity + Notional\n≥ min_notional 500?}
    H -- No or Crypto --> I[Single Market Order\nenqueue direct]
    H -- Yes equity --> J[SmartOrderRouter\nTWAP/VWAP/POV algo selection\n10 slices over duration_seconds]
    I --> K[Order Queue → Broker Submit]
    J --> K
    K --> L[_reserve_pending_buy\n900s dedup window]
    L --> M[trade_executed log + Prometheus]
```

---

## Exit Flow

```mermaid
flowchart TD
    A[Position in Portfolio] --> B{Exit Backoff\nActive?}
    B -- Yes --> SKIP[Skip Exit]
    B -- No --> C{PDT Force Swing?\nequity ≤ $2500 + daytrades ≥ 3}
    C -- Yes --> D[Force swing hold\nno same-day sell]
    C -- No --> E[_should_skip_exit\nHold time · Min profit · Stop loss]
    E -- Hold --> SKIP
    E -- Exit --> F{Dust Position?\nqty < 1e-6}
    F -- Yes --> G[broker.close_position\nforce-clear dust]
    F -- No --> H[Sell-to-Close\nmarket order\nqty = floor qty to avoid overshoot]
    H --> I{Fill Result?}
    I -- Completed --> J[Clear Backoff\nrecord P&L]
    I -- Rejected --> K[Exit Fail Count++\nExponential Backoff\n1/2/4/8/15 min cap]
```

---

## Dynamic Symbol Selection

```mermaid
flowchart TD
    A[Alpaca Active Universe\nequity + crypto active assets] --> B[Stablecoin Filter\nremove USDC/USDT/USDG pairs]
    B --> C[AI Filter Score\nReturn Ranker RF + Ollama catalyst\n+ PPO-based scoring]
    C --> D[Coverage Filter\nmin bars · price range]
    D --> E[Cap to max_symbols: 150]
    E --> F[Merge Held Positions\nalways included]
    F --> G{Equity Market Open?}
    G -- No → Crypto Only --> H[filter: '/' in symbol]
    G -- Yes --> I[All symbols]
    H --> J[_build_symbol_batches\nintersect with _symbols_by_broker\nto respect /USDT → Binance routing]
    I --> J
    J --> K[Per-Broker Symbol Sets\nAlpaca: /USD crypto + equities\nBinance: /USDT crypto]
```
