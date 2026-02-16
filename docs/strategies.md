<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Strategies

## Purpose
Generate buy/sell/hold/exit signals from market state.
Intraday signal metrics (30m/60m returns, early volume, runup/drawdown) are injected into
market state and influence actions via the signal bias guard.

## Active Strategies

- **Trend following**: `app/strategies/trend_following.py` — fast/slow MA crossover with breakout detection, **RSI(14) filter** (blocks overbought buys >=70, oversold sells <=30), **volume confirmation** (last bar >= 1.5x avg of prior 4 bars), **supertrend** alignment confirmation, **VWAP deviation** bonus, and **regime crisis gate** (blocks buys in `high_vol_crisis`). Extended exits: RSI > 75 or bearish supertrend flip. Confidence is a weighted composite of trend strength, RSI distance from extremes, supertrend alignment, and VWAP position. Emits `confidence` (0-1), `trend_strength`, and `rsi`.

- **Factor model**: `app/strategies/factor_model.py` — composite score from **momentum** (ROC indicator or 10-bar lookback), **liquidity**, **volatility**, and **mean-reversion** (stochastic + CCI when indicators available, else z-score fallback). **Hurst-adaptive weighting**: hurst > 0.5 (trending) boosts momentum 1.3x and reduces mr 0.5x; hurst < 0.5 (mean-reverting) boosts mr 1.5x and reduces momentum 0.6x. Trend quality gate (simplified ADX proxy) holds in choppy markets when trend_quality < 0.3 (skipped when hurst available). Default weights: momentum 0.5, liquidity 0.2, volatility 0.1, mean-reversion 0.15. Emits `confidence`, `score`, and `trend_quality`.

- **Pattern trading**: `app/strategies/pattern_trading.py` — chart pattern recognition (breakout + MA confirmation + higher highs/lows) with **ATR-based adaptive stop** (2x ATR with fixed `stop_loss_pct` as floor). Uses `atr()` from `app/learning/indicators.py`. Includes strategy-level partial TP, trailing stop, and stop loss.

- **Stat-arb pairs**: `app/strategies/stat_arb_pairs.py` — dynamic pair selection via **numpy ADF cointegration test** (pairs ranked by t-stat, threshold < -2.86 for ~5% significance). Uses **log-ratio spread**: `log(a) - beta * log(b)` with OLS hedge ratio from `np.polyfit`. Z-score entry at 2.0 (wider for selectivity), exit at 0.5. Pairs refresh every `refresh_minutes`. Emits `confidence` on actionable signals.

## Inactive Strategies (available but disabled by default)
- RL policy (`rl_policy`), fee-aware RL policy (`rl_policy_fees`): disabled — near-uniform output (~33/33/33), needs retraining.
- Market maker (`market_maker`): conflicts with directional strategies.
- Intraday momentum (`intraday_momentum`): superseded by trend_following.

## Extended Indicators
All strategies receive `market_state["indicators"]` — ~30 indicators computed by `compute_all_indicators()` from `app/learning/indicators.py`. Includes supertrend, vwap_dev, stochastic (K/D), CCI, ROC, hurst exponent, Bollinger bands, Keltner/Donchian channels, Ichimoku, OBV, MFI, parabolic SAR, and more. Strategies use these when available and fall back to inline computation otherwise.

## Confidence Calibration
`app/strategies/confidence_calibrator.py` — bin-based calibrator (no sklearn dependency). Tracks `(raw_confidence, was_profitable)` per strategy in a rolling window (200 samples). Maps raw confidence to empirical win-rate per bin with linear interpolation. Before 30 samples: returns `raw * 0.5` (conservative). Applied in `_combine_signals` before weighting.

## Regime-Weighted Signal Combiner
`_adjust_weights_for_regime()` in `trader.py` adjusts strategy weights based on the detected market regime before the signal scoring loop:
- **Low-vol trending** (regime 0): trend_following 1.3x, pattern_trading 1.2x, stat_arb 0.8x
- **High-vol crisis** (regime 2): trend_following 0.7x, stat_arb 1.4x, factor_model 1.2x
- **Normal** (regime 1): no adjustment

## Orchestration
- Orchestrator selects and weights strategies via `orchestrator.mode`.
- Default mode is `weight`: each strategy gets a fixed weight from `orchestrator.strategy_weights`, signals are combined by `calibrated_confidence * regime_adjusted_weight`.
- RL orchestrator (`orchestrator.rl.enabled: false` by default) is available but disabled — it adds latency without conviction when untrained.
- Combine mode: `priority` or `vote` (see `strategy.combine`).
- `strategy.min_conviction`: minimum weighted score required before acting (default 0.3). Below this threshold the agent holds.

### Global Account Activity Tracking
- **Purpose**: Penalizes prolonged inactivity across all symbols on an account
- **Implementation**: `AccountActivityTracker` in `app/agents/orchestrator.py`
- **Behavior**:
  - Tracks time since last trade across all symbols per account/broker
  - No penalty within `max_idle_minutes` threshold (default: 30 minutes)
  - Applies configurable penalty for excess idle time
  - Three penalty types: `linear`, `exponential` (default), or `step`
  - Resets timer on any trade across any symbol on the account
- **Use Case**: Encourages capital efficiency in multi-symbol portfolios

## Configuration
`config/config.yaml`:
- `strategy.name` or `strategy.names` — active strategy list (default: `[trend_following, factor_model, pattern_trading, stat_arb_pairs]`)
- `strategy.combine` — signal combination mode (`priority` or `vote`)
- `strategy.min_conviction` — minimum weighted score to act (default: 0.3)
- `strategy.params.trend_following.*` — fast/slow window, breakout/exit thresholds
- `strategy.params.factor_model.*` — weights (momentum, liquidity, volatility, mr_weight), buy/sell thresholds
- `strategy.params.stat_arb_pairs.*` — lookback, z_entry, z_exit, refresh_minutes, max_pairs
- `strategy.fee_aware.*` (fee guard)
- `strategy.signal_bias_guard.*`
- `pattern_trading.*` (selection filters, pattern params, entry/risk settings)
- `orchestrator.mode` — `weight` (default), `direct`, or `select`
- `orchestrator.strategy_weights` — per-strategy weight map (default: `{trend_following: 0.35, factor_model: 0.25, pattern_trading: 0.25, stat_arb_pairs: 0.15}`)
- `orchestrator.rl.*` — RL orchestrator (disabled by default)

## Usage
- Default: set `strategy.names: [trend_following, factor_model, pattern_trading, stat_arb_pairs]` with `strategy.combine: vote`.
- Single strategy: set `strategy.name: trend_following`.
- Add confidence: strategies emit `confidence` (0-1); `_combine_signals()` uses `confidence * strategy_weight` as effective weight.

## Strategy Signal Details

### Trend Following Filters
The RSI and volume filters prevent false signals in extreme conditions:
- **RSI filter**: Computed inline over the last 14 bars. Buys are blocked when RSI >= 70 (overbought), sells when RSI <= 30 (oversold). This avoids chasing momentum that is about to reverse.
- **Volume confirmation**: A buy requires the last bar's volume to be at least 1.5x the average of the prior 4 bars. Low-volume breakouts are ignored.
- **Supertrend confirmation**: When `indicators["supertrend"]` is available, bullish (1) confirms buy signals (+0.15 confidence), bearish (-1) penalizes them (-0.10). Vice versa for sells.
- **VWAP deviation**: Price above VWAP (`vwap_dev > 0`) gives a confidence bonus for buys; below VWAP gives a bonus for sells.
- **Regime crisis gate**: In `high_vol_crisis` regime, all buy signals are blocked (return hold).
- **Extended exits**: RSI > 75 triggers a sell (confidence 0.6). Bearish supertrend flip with negative trend triggers a sell (confidence 0.5).

### Factor Model Enhancements
- **Momentum**: Uses `indicators["roc"]` (rate of change) when available, else 10-bar lookback mean return.
- **Mean-reversion**: Uses `indicators["stoch_k"]` + `indicators["cci"]` for richer mr_score: stochastic oversold/overbought zones combined with CCI signal. Falls back to z-score of price vs 20-bar mean when indicators unavailable.
- **Hurst-adaptive weighting**: `indicators["hurst"]` dynamically adjusts factor weights. Trending (hurst > 0.5): momentum 1.3x, mr 0.5x. Mean-reverting (hurst < 0.5): momentum 0.6x, mr 1.5x. Bypasses trend quality gate.
- **Trend quality gate**: Ratio of net price movement to total absolute movement over 14 bars (simplified ADX proxy). When < 0.3, the market is choppy and the strategy holds. Skipped when hurst indicator is available.

### Stat-Arb Pairs Cointegration
- **Log-ratio spread**: `spread = log(a) - beta * log(b)` where beta is OLS hedge ratio from `np.polyfit(log_b, log_a, 1)`. Handles different-priced assets correctly (vs raw `a - b` difference).
- **Numpy ADF test**: Regresses spread differences on lagged spread levels, computes t-stat. Pairs with t-stat < -2.86 (~5% significance) are selected. Ranked by most negative t-stat (strongest cointegration).
- **Wider entry**: z_entry = 2.0 (was 1.5) for better selectivity.

### Pattern Trading ATR Stop
The stop loss is now adaptive: `stop = max(entry - 2*ATR(14), entry * (1 - stop_loss_pct))`. The fixed percentage acts as a floor, while ATR widens the stop in volatile markets and tightens it in calm ones.
