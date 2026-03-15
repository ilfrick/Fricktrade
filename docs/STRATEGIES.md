<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Strategy Reference

Fricktrade runs 9 active strategies simultaneously. Each strategy independently analyzes market data and votes buy/sell/hold. In `combine: vote` mode, votes are unweighted; a symbol receives a buy or sell signal when `buy_vote_threshold` (default 2) or `exit_vote_threshold` (default 2) strategies agree.

All strategies implement the `Strategy` base class from `app/strategies/base.py` and return a dict with `{action, confidence, name}`.

---

## Strategy Index

| # | Name | File | Asset Class |
|---|------|------|-------------|
| 1 | trend_following | `app/strategies/trend_following.py` | Both |
| 2 | factor_model | `app/strategies/factor_model.py` | Equities only |
| 3 | pattern_trading | `app/strategies/pattern_trading.py` | Both |
| 4 | stat_arb_pairs | `app/strategies/stat_arb_pairs.py` | Equities (disabled) |
| 5 | top_movers_rf | `app/strategies/top_movers_rf.py` | Both |
| 6 | crypto_momentum | `app/strategies/crypto_momentum.py` | Crypto only |
| 7 | crypto_mean_reversion | `app/strategies/crypto_mean_reversion.py` | Crypto only |
| 8 | gap_reversal | `app/strategies/gap_reversal.py` | Equities only |
| 9 | earnings_drift | `app/strategies/earnings_drift.py` | Equities only |

### Inactive (disabled by default)

| Name | File | Notes |
|------|------|-------|
| rl_policy | `app/strategies/rl_policy.py` | PPO policy; disabled pending convergence |
| rl_policy_fees | `app/strategies/rl_policy_fees.py` | Fee-aware RL variant |
| intraday_momentum | `app/strategies/intraday_momentum.py` | Superseded by crypto_momentum |
| market_maker | `app/strategies/market_maker.py` | Requires spread data (QuoteStream disabled) |

---

## 1. trend_following

**File**: `app/strategies/trend_following.py`
**Class**: `TrendFollowingStrategy`
**Asset class**: Both (equities and crypto)

### What it does

Generates buy signals when the fast EMA crosses above the slow EMA and price is in a breakout. Sells when fast EMA crosses below slow EMA or price drops enough below entry.

Uses pre-computed EMAs from `market_state["indicators"]` if available (injected by `compute_all_indicators()`), otherwise computes inline.

Additional filters from indicators when available:
- Supertrend direction (bullish = buy only)
- VWAP deviation (excessive deviation filters signal)
- RSI(14) gate (not overbought/oversold)
- Volume confirmation (current volume vs average)
- Regime crisis block (suppresses signals in `high_vol_crisis` regime)

### Signal conditions

**Buy**: fast EMA > slow EMA AND price > slow EMA AND breakout from recent range AND volume confirms
**Sell**: fast EMA < slow EMA AND price has dropped from peak by `exit_pct`
**Hold**: otherwise

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fast_window` | 10 | Fast EMA period (bars) |
| `slow_window` | 30 | Slow EMA period (bars) |
| `breakout_pct` | 0.3 | Minimum % gain from lookback low to trigger buy |
| `exit_pct` | 0.15 | % drop from recent high to trigger sell |
| `min_conviction_by_regime.low_vol_trending` | 0.25 | Lower threshold in trending markets |
| `min_conviction_by_regime.high_vol_crisis` | 0.50 | Higher threshold in volatile markets |

### Limitations

- EMA crossovers produce whipsaws in sideways markets
- Does not distinguish between trending and ranging conditions independently (relies on HMM regime from indicators)
- Crisis block (`high_vol_crisis`) may suppress valid trending signals during sharp but directional moves

---

## 2. factor_model

**File**: `app/strategies/factor_model.py`
**Class**: `FactorModelStrategy`
**Asset class**: Equities only (explicitly excluded from crypto symbols)

### What it does

Builds a composite factor score from momentum, liquidity, volatility, and mean-reversion components. The Hurst exponent (from indicators) adapts the weighting: high Hurst favors momentum; low Hurst favors mean-reversion.

Uses stochastic oscillator, CCI, and ROC from the indicator set when available.

### Signal conditions

**Buy**: composite factor score > `buy_threshold: 0.2`
**Sell**: composite factor score < `sell_threshold: -0.2`
**Hold**: score between thresholds

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `momentum_weight` | 0.5 | Base weight for momentum component |
| `liquidity_weight` | 0.2 | Weight for liquidity/volume component |
| `volatility_weight` | 0.1 | Weight for volatility component |
| `mr_weight` | 0.15 | Weight for mean-reversion component |
| `buy_threshold` | 0.2 | Minimum score for buy signal |
| `sell_threshold` | -0.2 | Maximum score for sell signal |

### Limitations

- Explicitly excluded from crypto (`"/" in symbol` guard in trader.py) — equity mean-reversion bias drove incorrect crypto signals
- Requires sufficient lookback data for Hurst exponent calculation; falls back to equal weights
- Mean-reversion component can signal sell during genuine uptrends in momentum markets

---

## 3. pattern_trading

**File**: `app/strategies/pattern_trading.py`
**Class**: `PatternTradingStrategy`
**Asset class**: Both

### What it does

Identifies chart pattern breakouts (moving average pullbacks, breakout from recent high) with ATR-based adaptive stops. Applies partial take-profit and trailing exit logic within the strategy signal itself.

### Signal conditions

**Buy**: price breaks above lookback high with volume confirmation AND recent pullback to MA
**Sell**: price drops by `stop_loss_pct` from entry OR partial TP target reached
**Hold**: otherwise

### Key parameters (in `pattern_trading` section of config)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `breakout_lookback_bars` | 20 | Bars to look back for breakout level |
| `volume_confirm_mult` | 1.5 | Required volume vs. average for confirmation |
| `stop_loss_pct` | 0.05 | Stop loss (5%) |
| `partial_take_profit_pct` | 0.1 | Partial take profit at 10% gain |
| `trailing_stop_pct` | 0.02 | Trailing stop at 2% from peak |

### Limitations

- Config section `pattern_trading` is separate from `strategy.params.pattern_trading` — these are selection filters (price, volume, spread), not signal parameters
- Pattern recognition is purely price-based; no structural pattern classification (head-and-shoulders etc.)
- Trailing and partial-TP logic within the strategy may conflict with the position-level stops in `_check_position_exit()`

---

## 4. stat_arb_pairs

**File**: `app/strategies/stat_arb_pairs.py`
**Class**: `StatArbPairsStrategy`
**Asset class**: Equities (disabled by default)

### Current status: DISABLED

Config: `strategy.params.stat_arb_pairs.enabled: false`

Reason: In long-only mode, stat_arb pair signals produce unhedged directional bets (you buy one leg but cannot short the other). Re-enable only when `trading_limits.allow_shorts: true` and short execution is working.

### What it does (when enabled)

Finds cointegrated equity pairs using ADF test. When the log-price ratio z-score exceeds `z_entry`, signals buy (undervalued leg). Exit when z-score reverts to `z_exit`.

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback` | 50 | Bars for cointegration test |
| `z_entry` | 1.5 | Z-score to trigger entry |
| `z_exit` | 0.5 | Z-score to trigger exit |
| `refresh_minutes` | 30 | How often to recompute pair rankings |
| `max_pairs` | 25 | Maximum active pairs |

### Limitations

- Class-level shared price cache (shared across all strategy instances for the same symbol pair) — see commit history for race condition fix
- ADF test requires sufficient history; produces no signals on fresh data
- Without shorting, signals are one-sided and not mean-reversion hedged

---

## 5. top_movers_rf

**File**: `app/strategies/top_movers_rf.py`
**Class**: `TopMoversRFStrategy`
**Asset class**: Both

### What it does

Loads a pre-trained Random Forest model from `/data/models/top_movers_rf/top_movers_rf.pkl`. Uses intraday features (session gain, relative volume, low-zone proximity) to score symbols. Signals buy when the nowcast score exceeds threshold AND the symbol is near its session low.

Can retrain automatically on start (`auto_train_on_start: true`) if sufficient training data is available in `/data`.

### Signal conditions

**Buy**: nowcast score > `buy_nowcast_min: 0.50` AND entry score > `buy_entry_min: 0.50` AND price near session low
**Sell**: exit score < `exit_score_max: 0.40` OR pullback from session high > `exit_pullback_from_high_pct: 1.2%`
**Hold**: otherwise or model not loaded

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `buy_nowcast_min` | 0.50 | Minimum RF nowcast probability |
| `buy_entry_min` | 0.50 | Minimum entry score |
| `buy_score_min` | 0.54 | Combined score threshold |
| `low_zone_tol_pct` | 0.35 | "Near session low" tolerance |
| `top_n` | 10 | Symbols to score per cycle |
| `runtime_interval` | 5m | Signal computation frequency |
| `training_interval` | 5m | Model update frequency |
| `entry_warmup_minutes` | 20 | Don't trade within first 20 min of session |
| `cutoff_minutes` | 30 | Stop trading 30 min before close |

### Limitations

- Model degrades if training data is stale; retraining interval is 24h
- Requires at least `min_training_days: 5` of data before producing useful signals
- The RF model is a generic intraday signal; no crypto-specific training
- Session-low detection does not work well for 24/7 crypto (no clear session)

---

## 6. crypto_momentum

**File**: `app/strategies/crypto_momentum.py`
**Class**: `CryptoMomentumStrategy`
**Asset class**: Crypto only (returns hold for non-`/` symbols)

### What it does

Short-term momentum strategy for crypto. Buys when returns across all three timeframes (fast/medium/slow) are positive. Confidence is proportional to the minimum cross-timeframe return. Only operates in long direction (no shorts for crypto).

### Signal conditions

**Buy**: 5m return > 0 AND 15m return > 0 AND 60m return > 0 AND at least one return > `min_return_pct`
**Hold**: otherwise (no sell signal generated — exits handled by position-level stops)

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fast_window` | 5 | Fast return window (bars) |
| `medium_window` | 15 | Medium return window (bars) |
| `slow_window` | 60 | Slow return window (bars) |
| `volume_mult` | 1.5 | Confidence boost multiplier when volume elevated |
| `min_return_pct` | 0.15 | Minimum return threshold to trigger signal |
| `trailing_stop_pct` | 2.0 | Strategy-internal trailing stop hint |

### Limitations

- No sell/exit signals; relies entirely on position-level stops and time exits
- Triple-positive requirement misses unidirectional moves that haven't propagated to all timeframes
- Low `min_return_pct` (0.15%) generates noise in flat markets; was 0.3% originally

---

## 7. crypto_mean_reversion

**File**: `app/strategies/crypto_mean_reversion.py`
**Class**: `CryptoMeanReversionStrategy`
**Asset class**: Crypto only

### What it does

Mean-reversion strategy using Bollinger Bands (default 20 period, 2σ) and RSI for crypto. Buys when price is oversold below the lower Bollinger Band with RSI confirmation. Uses VWAP deviation as an additional signal.

### Signal conditions

**Buy**: price < lower Bollinger Band AND RSI < `rsi_oversold: 30` AND recent drop > threshold
**Sell**: price > upper Bollinger Band OR RSI overbought
**Hold**: otherwise

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bb_period` | 20 | Bollinger Band period |
| `bb_std` | 2.0 | Bollinger Band standard deviation multiplier |
| `rsi_period` | 14 | RSI period |
| `rsi_oversold` | 30.0 | RSI oversold threshold (was 35; tightened to reduce noise) |
| `drop_window_bars` | 15 | Bars to measure recent drop over |
| `hard_stop_pct` | 3.0 | Strategy hard stop hint |

### Limitations

- Mean-reversion assumes prices revert; momentum regimes can produce sustained moves below Bollinger Band
- Pairing with `crypto_momentum` (trending signal) helps, but they can vote opposite sides simultaneously
- RSI threshold of 30 may miss moderate but real oversold conditions; threshold of 35 produced too many false signals

---

## 8. gap_reversal

**File**: `app/strategies/gap_reversal.py`
**Class**: `GapReversalStrategy`
**Asset class**: Equities only

### What it does

Post-gap-open reversal strategy. After a gap up or gap down at market open (measured as today's open vs. prior close), signals that gaps will partially fill. Restricted to the first hour of trading (9:35–10:30 ET window).

### Signal conditions

**Buy (gap down reversal)**: gap down > `min_gap_pct: 2.0%` AND RSI oversold AND volume confirms
**Sell (gap up reversal)**: gap up > `min_gap_pct: 2.0%` AND RSI overbought
**Hold**: outside trading window, or gap too small, or volume insufficient

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_gap_pct` | 2.0 | Minimum gap size to trigger |
| `target_fill_pct` | 50.0 | Expected gap fill % (exit target) |
| `stop_pct` | 0.5 | Stop loss % from entry |
| `volume_confirm_mult` | 2.0 | Required volume multiple |
| `max_trade_window_minutes` | 60 | Maximum time after open to enter |
| `rsi_period` | 5 | Short RSI for gap detection |
| `rsi_oversold` | 30.0 | Buy trigger |
| `rsi_overbought` | 70.0 | Sell trigger |

### Limitations

- Highly time-sensitive; outside 9:35–10:30 ET produces no signals
- Gap detection requires the prior session close, which may not be available in early-morning data
- False signals when gaps are driven by fundamental news (earnings, M&A) rather than overnight noise

---

## 9. earnings_drift

**File**: `app/strategies/earnings_drift.py`
**Class**: `EarningsDriftStrategy`
**Asset class**: Equities only

### What it does

Post-Earnings Announcement Drift (PEAD) strategy. After a large earnings-driven gap, academic research shows prices continue drifting in the gap direction for days to weeks. This strategy buys after a qualifying earnings gap and holds for a multi-day drift.

Requires Alpha Vantage API for the earnings calendar (`ALPHA_VANTAGE_API_KEY`). Falls back to no signals if the API key is not set.

### Signal conditions

**Buy**: symbol has earnings event within window AND gap > `min_gap_pct: 5.0%` AND volume > `min_volume_mult: 1.5×` average
**Hold/Sell**: position held for up to `hold_days: 20`; trailed by `trailing_stop_pct: 4.0%`

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_gap_pct` | 5.0 | Minimum earnings gap to trigger entry |
| `min_volume_mult` | 1.5 | Required volume multiple on earnings day |
| `hold_days` | 20 | Default hold period (multi-day PEAD) |
| `trailing_stop_pct` | 4.0 | Trailing stop from peak |

### Limitations

- Intended for multi-day holding; conflicts with intraday `min_hold_minutes` and time-exit settings
- Requires earnings calendar data from Alpha Vantage (free tier has rate limits)
- PEAD effect is documented but not consistent; works better for smaller-cap stocks with analyst underreaction
- The strategy votes in the same vote pool as intraday strategies, meaning 2 more intraday "hold" votes can override it

---

## Adding a New Strategy

See `docs/DEVELOPMENT.md` for the step-by-step guide to adding a new strategy to the system.

---

## Vote Mode Details

In `combine: vote` mode:
- All enabled strategies run every cycle for every symbol
- Each strategy casts one vote: `buy`, `sell`, or `hold`
- `"exit"` is normalized to `"sell"` before vote counting
- Votes are counted: if `buy_count >= buy_vote_threshold (2)` → buy signal; if `sell_count >= exit_vote_threshold (2)` → sell signal
- Ties go to `rl_policy` signal as tiebreaker (if enabled); otherwise hold
- Strategy weights (in config) are ignored entirely in vote mode
- A signal bias guard prevents extreme skew: if buy or sell votes represent > 65% of all non-hold votes, the signal is treated as suspicious noise
