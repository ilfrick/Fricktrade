# Trading Agent Flow (master/dev)

This document describes the end-to-end flow for the trading agent in the master/dev branch.

## Trading Loop (High-Level)

```mermaid
flowchart TD
    A[Start Loop] --> B[Load Portfolio + Symbols]
    B --> C[Refresh News + Dynamic Symbols]
    C --> D[Update Metrics + Open Orders]
    D --> E[Read Ops State]
    E --> F{Market Open?}
    F -- No --> G[Sleep Interval]
    G --> A
    F -- Yes --> H[Per-Symbol Market Data]
    H --> I[Generate Strategy Signals + Features]
    I --> J[Drift Monitor + Auto Rollback]
    J --> K[Orchestrator Direct Strategy Selection]
    K --> L[Guardrail Optional]
    L --> M[Risk + Limits + VaR/CVaR + Caps]
    M --> N[Execution Algo + Queue]
    N --> O[Broker Router Submit]
    O --> P[Order Feedback + Metrics]
    P --> Q[Decision Trace + Audit/Compliance Logs]
    Q --> A
```

## Orchestrator Decision Path (Direct)

```mermaid
flowchart TD
    A[Signals from All Active Strategies] --> B[RL Orchestrator Scores]
    B --> C{Mode=direct}
    C --> D[Pick Top Strategy]
    D --> E[Use Single Strategy Action]
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
    E --> G[Queue Child Orders]
    F --> H[Queue Order]
    G --> H
    H --> I[Broker Router Submit]
    I --> J[Update Open Orders + Metrics]
```

## Dynamic Symbols (AI Filter)

```mermaid
flowchart TD
    A[Universe Broker Active] --> B[AI Filter Score]
    B --> C[Cap to max_symbols]
    C --> D[Merge Positions + Open Orders]
    D --> E[Active Symbol List]
```
