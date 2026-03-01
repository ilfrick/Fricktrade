<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Fricktrade — Implementation Status & Roadmap

*Last updated: 2026-03-01 (bug fixes: close_position slash, Kelly sizing, stuck-order timeout; risk limits updated). See `AGENTS.md` for full history.*

---

## Current State (v3.0)

The system is **live on Alpaca paper** trading US equities (intraday) and crypto (24/7).
All core trading, risk, execution, and LLM components are implemented and deployed.

---

## What Is Complete

### Strategies
| Strategy | Status | Notes |
|---|---|---|
| `trend_following` | ✅ | Supertrend + VWAP deviation + RSI(14) + volume + regime crisis gate |
| `factor_model` | ✅ | Hurst-adaptive, stochastic + CCI, ROC, mean-reversion quality gate |
| `pattern_trading` | ✅ | ATR-based adaptive stop (2×ATR), partial take-profit, trailing exit |
| `stat_arb_pairs` | ✅ | Log-ratio spread, ADF cointegration, OLS hedge ratio, z_entry=2.0 |
| `top_movers_rf` | ✅ | Random-forest same-day nowcast + intraday low-zone entry |
| `crypto_momentum` | ✅ | Crypto-specific RSI/volume; 24/7 via Alpaca GTC |
| `crypto_mean_reversion` | ✅ | Bollinger/VWAP mean-reversion for crypto |
| `gap_reversal` | ✅ | Gap >2% reversal, 9:35–10:30 ET window, volume + RSI filter |
| `earnings_drift` | ✅ | PEAD — post-earnings gap ≥5% + volume ≥1.5×; Alpha Vantage calendar |

### Signal Processing
| Feature | Status |
|---|---|
| ~30-indicator injection (`compute_all_indicators()`) | ✅ |
| ConfidenceCalibrator (bin-based, per-strategy, 200-sample warm-up) | ✅ |
| HMM regime detection (3-state) | ✅ |
| Regime-aware weight adjustment (`_adjust_weights_for_regime()`) | ✅ |
| MacroRegimeAnalyzer (FRED VIX/DGS10/DXY + Claude, 4h TTL) | ✅ |
| Half-Kelly sizing from calibrated win probability | ✅ |

### Execution
| Feature | Status |
|---|---|
| Limit orders default (auto-upgrade market→limit at mid) | ✅ |
| SmartOrderRouter (Almgren-Chriss impact model) | ✅ |
| TWAP / VWAP / POV algo slicing | ✅ |
| `adaptive_slices()` (regime-aware: 2× slices in crisis) | ✅ |
| Fractional share support (`AlgoSlice.qty: float`, `fractional_slice()`) | ✅ |
| TCA feedback loop (EWMA slippage penalty, −50% max, 0.9×/day decay) | ✅ |
| Time-of-day scaling (open/close blocks, lunch 50%, crypto 00-04 UTC 70%) | ✅ |
| Stuck-order timeout (cancel + unblock queue after `max_order_age_seconds: 300`) | ✅ |

### Risk Management
| Feature | Status |
|---|---|
| ATR-based stops (1.5× equity, 2.5× crypto) | ✅ |
| Per-symbol circuit breaker (5% unrealized loss) | ✅ |
| PDT retry suppression (per-broker/symbol, daily reset) | ✅ |
| Pending notional race prevention (atomic reserve/release) | ✅ |
| Two-phase dispatch (exits complete before entries) | ✅ |
| Exposure caps (venue: NYSE/Nasdaq 60%, Crypto 50%; sector: Technology 35%, etc.) | ✅ |
| VaR/CVaR gating | ✅ |
| RiskEventInterpreter wired into `_handle_drift()` (pause circuit breaker) | ✅ |

### LLM Integration (`app/llm/`)
| Component | Status | Config key |
|---|---|---|
| `NewsSentimentAnalyzer` (Claude, 15-min cache) | ✅ enabled | `llm.sentiment.enabled` |
| `RiskEventInterpreter` (Claude, `_handle_drift()` pause gate) | ✅ enabled | `llm.risk_interpreter.enabled` |
| `MacroRegimeAnalyzer` (Claude, FRED, 4h TTL) | ✅ enabled | `llm.macro_regime.enabled` |
| `PostSessionAnalyst` (Gemini, cron 22:30 CET) | ✅ enabled | `llm.post_session.enabled` |
| `DailySymbolsFilter` (Gemini, pre-market + crypto 4h) | ✅ wired | `llm.symbols_filter.enabled: false`* |
| `MetaOrchestrator` (Claude, weekly weight rebalancing) | ✅ wired | `llm.meta_orchestrator.enabled: false`* |

*\* Disabled pending sufficient session history. Enable `symbols_filter` manually;
enable `meta_orchestrator` after ≥3 session reports exist in `data/session_reports/`.*

### Data
| Feature | Status |
|---|---|
| Alpaca as primary data provider (5m bars) | ✅ |
| Universe: `alpaca_active_all` — equities + crypto from Alpaca `get_all_assets()` (no hardcoded lists) | ✅ |
| `buying_power_scaling: false` — both accounts evaluate full 150-symbol universe; `_size_order` scales qty | ✅ |
| QuoteStream real-time bid/ask (alpaca-py WebSocket) | ✅ |
| Earnings calendar (Alpha Vantage CSV, daily refresh) | ✅ |
| Fear & Greed Index (alternative.me, no key needed) | ✅ |
| CoinGlass open interest (API key optional) | ✅ |
| SEC EDGAR insider trades (Form 4, public API) | ✅ |
| Return ranker (XGBoost, rolling 3-day window) | ✅ |
| Crypto training data collector (hourly cron `0 * * * *`) | ✅ |
| Walk-forward backtest with embargo gap | ✅ |

### Portfolio
| Feature | Status |
|---|---|
| PortfolioOptimizer (risk_parity covariance) | ✅ |
| RebalanceEngine (drift threshold 5%, min 1% trade) | ✅ |
| Cross-asset covariance (equities + crypto in same matrix) | ✅ |
| Sector/venue concentration limits | ✅ |
| Fractional shares (`trading_limits.fractional_shares: true`, `min_notional: 1.0`) | ✅ |

### GPU Allocation (GTX 1060 6 GB)
| Workload | Container | VRAM | Notes |
|---|---|---|---|
| Ollama llama3.2:3b | `ollama` | ~2.7 GB | Primary — 15–25× speedup (7s CPU → ~0.4s); `CUDA_VISIBLE_DEVICES=0`; `OLLAMA_KEEP_ALIVE=24h` |
| RL online training | `learner` | ~50–100 MB | Runs market-closed windows |
| AI filter (PPO) | `trader` | ~15–20 MB | Tiny model; re-enabled Mar 2026 |
| Headroom | — | ~3.1 GB | Prevents OOM recurrence |

GPU enable/disable state persists in `/data/gpu_state.json`. Re-enable with `scripts/enable_gpu.sh`.

**Root cause of prior Ollama exits**: `docker-compose.gpu.yml` mounted `/dev/nvidia*` into Ollama with `CUDA_VISIBLE_DEVICES=""` (empty string ≠ `"none"` for CUDA), causing the CUDA runner to crash on startup (exit 0, silent). Fixed: `CUDA_VISIBLE_DEVICES=0` (explicit device number).

---

## What Remains

### P1 — High Priority

#### RL Convergence Fix
**Status:** Identified, not started. RL is disabled (`orchestrator.rl.enabled: false`).

The RL orchestrator outputs near-uniform action probabilities (~33/33/33) — insufficient
samples and a reward signal too weak for credit assignment. The rule-based weight mode
is the current production path and is working well.

Work needed:
- Switch to continuous action space `[-1, 1]` (weight multiplier per strategy)
- Improved reward shaping: risk-adjusted return with transaction cost penalty
- 500k timestep training run
- Walk-forward validation: must beat rule-based baseline before re-enabling
- Curriculum learning: start with constrained action space, widen over time

Files: `app/agents/orchestrator.py`, `app/learning/`

#### PDT Force-Swing Mode
**Status:** ✅ Implemented (2026-03-01).

`_would_trigger_pdt_swing()` checks rolling `daytrade_count` from Alpaca account flags.
When count ≥ 3 and equity ≤ $2,500 (Alpaca threshold), exits are suppressed for equities
(crypto symbols with "/" are exempt). Configurable via `risk.pdt` block.

---

### P2 — Medium Priority

#### Per-Strategy Kill Switch
**Status:** ✅ Implemented (2026-03-01). `strategy.params.<name>.enabled: false` checked
in `_strategy_disabled_globally()` before signal evaluation.

#### Regime-Aware `min_conviction` Per Strategy
**Status:** ✅ Implemented (2026-03-01). `strategy.params.<name>.min_conviction_by_regime`
map overrides global `min_conviction` in `_combine_signals()` for the winning strategy
in the current regime.

#### Auto-Disable on Negative Sharpe
**Status:** ✅ Implemented (2026-03-01). `PerformanceTracker._neg_sharpe_count` tracks
consecutive negative-Sharpe reports; `get_neg_sharpe_weight_mult()` returns 0.5× after
N consecutive reports (default 3). Wired into `_adjust_weights_for_regime()`.

#### Factor Risk Pre-Trade Gate
**Status:** `RiskModel.factor_risk()` exists but is not called before order submission.

Wire `factor_risk()` into `_check_execution_for_symbol()` to block trades that would
increase factor concentration beyond threshold.

Files: `app/agents/trader.py`, `app/risk/manager.py`

#### Enable `meta_orchestrator` After Session History
**Status:** Wired but disabled. Trigger: `data/session_reports/` has ≥3 report files.

The MetaOrchestrator (Claude) reads post-session reports and adjusts strategy weights
weekly. Enable by setting `llm.meta_orchestrator.enabled: true` in config once enough
session history exists.

---

### P3 — Architectural / Lower Priority

#### Intent Semantic Separation
Replace the binary `buy`/`sell` action with `enter_long` / `exit_long` / `enter_short` /
`exit_short`. This would make the execution path unambiguous and remove several guards
that currently infer intent from position state.

#### OMS Pending Quantity Validation
Prevent oversell at the order management layer by tracking pending sell qty per symbol
and blocking new sell orders that would exceed held quantity.

#### Extend Extended-Hours Trading
`extended_hours` flag exists in the broker adapter but is not wired through the strategy
layer. Earnings drift signals in particular would benefit from pre-market execution.

---

## Cron Jobs to Set Up

| Script | Schedule | Purpose |
|---|---|---|
| `scripts/collect_return_ranker_data.py` | `30 21 * * 1-5` | Equity return-ranker daily training data |
| `scripts/collect_crypto_training_data.py` | `0 * * * *` | Crypto return-ranker hourly data |
| `scripts/decision_monitor.py` | `25 15 * * 1-5` | Session monitoring start |
| `scripts/post_session_analyst.py` | `30 22 * * 1-5` | Post-session LLM analysis |

---

## API Keys Needed

| Key | Service | Status | Impact if missing |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Claude LLM | **Required** | Sentiment, risk_interpreter, macro_regime disabled |
| `GOOGLE_GEMINI_API_KEY` | Gemini LLM | **Required** | symbols_filter, post_session disabled |
| `FRED_API_KEY` | FRED macro data | Optional | MacroRegimeAnalyzer falls back to yfinance `^VIX` |
| `ALPHA_VANTAGE_API_KEY` | Earnings calendar | Optional | EarningsDriftStrategy uses no-signal fallback |
| `COINGLASS_API_KEY` | Crypto OI | Optional | `crypto_oi_change_pct` not injected into market_state |

---

## Historical Plan Documents

The following planning documents in the repo root are **superseded** by this STATUS.md
and by `AGENTS.md`. They are kept as historical records:

- `fricktrade_implementation_plan.md` — original 28-Feb implementation plan (~95% complete)
- `piano_implementazione.md` — 5-Feb Italian priority list (~90% complete)
- `valutazione_fricktrade_v3.md` — 15-Feb diagnostic evaluation (all critical issues resolved)
- `fricktrade_review.md` — code review findings (all P0/P1 items resolved)
- `LLM_Integration_Design.md` — LLM design spec (fully implemented)
- `Fricktrade_Sintesi_Cross_Review_2026-02-18.md` — multi-model cross-review synthesis

The following docs in `docs/` predate v3.0 and describe planned work that is now complete
or superseded:

- `docs/improvement_roadmap.md` — Feb 2026; most items implemented in v3.0
- `docs/strategy_upgrade_plan.md` — Feb 2026; strategy upgrades complete
- `docs/top_tier_backlog.md` / `docs/top_tier_epics.md` — pre-v3.0 backlogs; largely done
