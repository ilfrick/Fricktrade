<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Top-Tier Roadmap (Epics)

> **⚠️ HISTORICAL DOCUMENT** — Pre-v3.0 epics. See `docs/STATUS.md` for current status.

## EPIC 1: Quant-Grade Validation Pipeline
- Walk-forward + regime split automation
- Monte Carlo/bootstrapped confidence intervals
- Stress test suite (costs, spreads, liquidity)
- Automated benchmark report with scores per regime

## EPIC 2: Institutional Data Quality
- Data validation layer (gaps, spikes, split handling)
- Unified data schema across providers
- Latency and data health monitoring

## EPIC 3: Execution Quality & Market Impact
- Adaptive slicing (TWAP/VWAP/POV with liquidity)
- Market impact model + expected slippage
- Order lifecycle state machine with retry budget

## EPIC 4: Risk & Portfolio Governance
- VaR/CVaR + exposure caps by sector/broker
- Dynamic sizing (vol targeting + Kelly fraction)
- Automatic kill switches with regime-aware thresholds

## EPIC 5: ML/RL Governance & Model Ops
- Model registry + reproducible training
- Drift detection + auto rollback to best model
- Feature store and consistency checks

## EPIC 6: Monitoring & Compliance
- Audit trail of every decision path
- Latency budget dashboards
- Compliance-ready logs and retention policies
