<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Trading Agent Flow (v2.0)

This document describes the end-to-end flow for the trading agent in the v2.0 branch.

## Trading Loop (High-Level)

```mermaid
flowchart TD
    A[Start Loop] --> B[Load Portfolio + Symbols]
    B --> C[Refresh News + Dynamic Symbols]
    C --> D[Update Metrics + Open Orders]
    D --> E{Market Open?}
    E -- No --> F[Sleep Interval]
    F --> A
    E -- Yes --> G[Per-Symbol Market Data]
    G --> H[Generate Strategy Signals]
    H --> I[Orchestrator Direct Strategy Selection]
    I --> J[Guardrail Optional]
    J --> K[Risk + Limits + Sizing]
    K --> L[Execution Algo + Queue]
    L --> M[Broker Submit]
    M --> N[Order Feedback + Metrics]
    N --> A
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
    H --> I[Broker Submit]
    I --> J[Update Open Orders + Metrics]
```

## Dynamic Symbols (AI Filter)

```mermaid
flowchart TD
    A[Universe Alpaca Active] --> B[AI Filter Score]
    B --> C[Cap to max_symbols]
    C --> D[Merge Positions + Open Orders]
    D --> E[Active Symbol List]
```
