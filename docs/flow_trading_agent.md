<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Trading Agent Flow (master/dev)

This document describes the end-to-end flow for the trading agent in the master/dev branch.

## Trading Loop (High-Level)

```mermaid
flowchart TD
    A[Start Loop] --> B[Load Portfolio + Symbols]
    B --> C[Refresh News + Dynamic Symbols]
    C --> Ollama[Ollama LLM Gate]
    Ollama --> D[Update Metrics + Open Orders]
    D --> E[Read Ops State]
    E --> F{Market Open?}
    F -- No --> G[Sleep Interval]
    G --> A
    F -- Yes --> H["Per-Symbol Market Data (market-cache)"]
    H --> I[Generate Strategy Signals + Features]
    I --> J[Drift Monitor + Auto Rollback]
    J --> K[Orchestrator Direct Strategy Selection]
    K --> L[Guardrail Optional]
    L --> AccountFlags[Account Flags + Trading Limits]
    AccountFlags --> M["Risk + VaR/CVaR + Caps - configurable disable"]
    M --> N[Execution Algo + Queue]
    N --> O[Broker Router Submit]
    O --> P[Order Feedback + Metrics]
    P --> Q[Decision Trace + Audit/Compliance Logs]
    Q --> A
```

## Orchestrator Decision Path (Weight)

```mermaid
flowchart TD
    A[Signals from Active Strategies] --> B[Apply strategy_weights]
    B --> C[Multiply by signal confidence]
    C --> D[Sum buy_score / sell_score]
    D --> E{Score >= min_conviction?}
    E -- Yes --> F[Execute winning action]
    E -- No --> G[Hold]
```

## Execution Path (Market/Limit + Algos)

```mermaid
flowchart TD
    A[Action + Qty] --> B{Order Type?}
    B -- Limit --> C[Validate Limit Price]
    B -- Market --> D{Algo Enabled & Notional >= min?}
    D -- Yes --> E[Slice TWAP/VWAP/POV]
    D -- No --> F[Single Order]
    C --> F
    E --> PLC{Buy? Pending Leverage OK?}
    F --> PLC
    PLC -- No --> SKIP[Skip: pending_leverage_cap]
    PLC -- Yes --> G[Queue Order]
    G --> H[Update Portfolio Exposure In-Memory]
    H --> I[Broker Router Submit]
    I --> J[Update Open Orders + Metrics]
    J --> K[Release Pending Notional on Terminal]
```

## Exit Backoff Flow

```mermaid
flowchart TD
    A[Position Exit Trigger] --> B{Exit Backoff Active?}
    B -- Yes --> C[Skip Exit]
    B -- No --> D[Check Position Exit]
    D --> E{Should Exit?}
    E -- Yes --> F[Sell-to-Close via Queue]
    F --> G{Sell Result?}
    G -- Completed --> H[Clear Backoff]
    G -- Rejected --> I[Record Failure + Exponential Backoff]
    E -- No --> J[Hold]
```

## Dynamic Symbols (AI Filter)

```mermaid
flowchart TD
    A[Universe Broker Active] --> B[AI Filter Score]
    Bars["Market Cache (Redis + file)"] --> B
    News[News Catalyst] --> B
    B --> C[Coverage Filter]
    C --> D[Cap to max_symbols]
    D --> E[Merge Positions + Open Orders]
    E --> F[Active Symbol List]
    F --> FilteredCache["Filtered Symbols Cache - per symbol"]
    FilteredCache --> Bars
```
