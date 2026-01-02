<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Top-Tier Backlog (Phase 1–2)

## Epics → Work Items

### EPIC 1: Quant-Grade Validation
1. Add walk-forward + regime tags to benchmark output (done)
2. Add bootstrap confidence intervals for return metrics
3. Add MC stress module with randomized slippage/spread shocks
4. Add benchmark scorecard (Sharpe/Sortino/Calmar/Ulcer)
5. Add comparison to buy-and-hold baseline per symbol set

### EPIC 2: Data Quality
6. Add data validator for OHLCV (gap/outlier checks)
7. Add adjustment layer for splits/dividends
8. Add data quality report per symbol

### EPIC 3: Execution Quality
9. Add market impact estimator (volume/volatility based)
10. Add adaptive execution algo selection
11. Add retry policy with risk-aware budget

### EPIC 4: Risk & Portfolio Governance
12. Add VaR/CVaR estimator with limits per broker
13. Add exposure caps by sector/venue
14. Add kill switch profiles (low/med/high volatility)

### EPIC 5: ML/RL Governance
15. Add model registry metadata + versioned artifacts
16. Add drift detection (feature distribution + PnL drop)
17. Add auto rollback to best model on drift

### EPIC 6: Monitoring & Compliance
18. Add decision audit trail with full feature snapshot
19. Add latency budget dashboard (data → decision → order)
20. Add compliance log export (CSV/JSON) per day
