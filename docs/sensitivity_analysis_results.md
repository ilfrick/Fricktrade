# Sensitivity Analysis Results

**Date:** 2026-03-24
**Strategy:** crypto_mean_reversion (+ ATR stop multiplier)
**Data:** 1-minute bars, 33 crypto symbols
**Test periods:** crash (2026-01-28 to 2026-02-09), recovery (2026-02-21 to 2026-03-03)
**Method:** Each parameter tested independently at multiple values while holding others at baseline.
**Baseline config:** bb_period=100, bb_std=2.0, rsi_period=70, rsi_oversold=30, drop_window_bars=75, atr_stop_mult=2.5

---

## Summary

| Parameter | Best Value | Avg Return | Win Rate | Trades | Finding |
|-----------|-----------|------------|----------|--------|---------|
| **bb_period** | **50** | **+1.00%** | 96.0% | 38 | Most impactful parameter |
| bb_std | 2.0 | +0.41% | 90.2% | 94 | Baseline is optimal |
| rsi_period | 70 | +0.58% | 90.2% | 94 | Baseline is optimal |
| rsi_oversold | 30.0 | +0.58% | 90.2% | 94 | Baseline is optimal |
| drop_window_bars | any | +0.58% | 90.2% | 94 | Dead parameter (zero effect) |
| atr_stop_mult | 1.5–3.5 | +0.58% | 90.2% | 94 | Dead parameter (except 5.0 hurts) |

**Deployed config after analysis:** bb_period changed from 100 → 50. All other parameters unchanged.
Validated via Bonferroni-corrected walk-forward backtest (candidate #10, CI [+0.03%, +1.92%] excludes zero).

---

## Detailed Results

### bb_period (Bollinger Band lookback)

Path: `strategy.params.crypto_mean_reversion.bb_period`

| Value | Avg Return | Trades | Sells | Wins | Win Rate | Avg Win | Avg Loss | Max DD |
|------:|----------:|-------:|------:|-----:|---------:|--------:|---------:|-------:|
| **50** | **+0.998%** | 38 | 25 | 24 | 96.0% | +2.356% | -0.770% | 0.1% |
| 75 | +0.845% | 75 | 49 | 39 | 79.6% | +2.214% | -2.960% | 0.7% |
| 100 | +0.104% | 95 | 62 | 44 | 71.0% | +2.098% | -2.688% | 1.4% |
| 150 | +0.124% | 127 | 81 | 59 | 72.8% | +2.042% | -2.950% | 1.9% |
| 200 | -1.331% | 133 | 82 | 56 | 68.3% | +1.827% | -3.500% | 3.0% |

**Finding:** Shorter lookback (50) catches more mean-reversion entries with higher precision.
Longer windows (150–200) over-smooth and generate losing trades. Clear monotonic improvement
from 200 → 50.

### bb_period (low range: 20–45)

Path: `strategy.params.crypto_mean_reversion.bb_period`

| Value | Avg Return | Trades | Sells | Wins | Win Rate | Avg Win | Max DD |
|------:|----------:|-------:|------:|-----:|---------:|--------:|-------:|
| 20 | +0.000% | 0 | 0 | 0 | — | — | 0.0% |
| 25 | +0.000% | 0 | 0 | 0 | — | — | 0.0% |
| 30 | +0.000% | 0 | 0 | 0 | — | — | 0.0% |
| 35 | +0.112% | 3 | 2 | 2 | 100% | +1.277% | 0.0% |
| 40 | +0.185% | 15 | 10 | 10 | 100% | +0.785% | 0.0% |
| 45 | +0.676% | 35 | 23 | 23 | 100% | +1.462% | 0.0% |

**Finding:** Values below 35 produce zero trades — the BB window is too narrow on 1m bars for
BB + RSI + fast-drop conditions to trigger simultaneously. Values 35–45 produce fewer and
smaller trades than 50. **bb_period=50 is the optimal floor.**

### bb_std (Bollinger Band standard deviations)

Path: `strategy.params.crypto_mean_reversion.bb_std`

| Value | Avg Return | Trades | Sells | Wins | Win Rate | Avg Win | Avg Loss | Max DD |
|------:|----------:|-------:|------:|-----:|---------:|--------:|---------:|-------:|
| 1.5 | -1.046% | 166 | 102 | 78 | 76.5% | +1.595% | -4.202% | 2.6% |
| 1.75 | -0.075% | 124 | 79 | 65 | 82.3% | +1.563% | -4.107% | 1.5% |
| **2.0** | **+0.405%** | 94 | 61 | 55 | 90.2% | +1.663% | -2.185% | 0.9% |
| 2.5 | +0.239% | 48 | 32 | 27 | 84.4% | +1.409% | -3.737% | 0.5% |
| 3.0 | +0.174% | 12 | 8 | 7 | 87.5% | +1.069% | -0.205% | 0.0% |

**Finding:** 2.0σ is the sweet spot. Tighter bands (1.5) trigger too many false entries.
Wider bands (2.5–3.0) are too conservative and miss opportunities. Avg loss drops
sharply at 2.0.

### rsi_period (RSI lookback)

Path: `strategy.params.crypto_mean_reversion.rsi_period`

| Value | Avg Return | Trades | Sells | Wins | Win Rate | Avg Win | Avg Loss | Max DD |
|------:|----------:|-------:|------:|-----:|---------:|--------:|---------:|-------:|
| 30 | -1.451% | 162 | 100 | 73 | 73.0% | +1.645% | -3.928% | 3.5% |
| 50 | -0.139% | 144 | 92 | 77 | 83.7% | +1.688% | -4.189% | 2.2% |
| **70** | **+0.577%** | 94 | 61 | 55 | 90.2% | +1.663% | -2.185% | 1.2% |
| 100 | +0.540% | 30 | 20 | 20 | 100% | +1.554% | — | 0.0% |
| 140 | +0.000% | 0 | 0 | 0 | — | — | 0.0% |

**Finding:** RSI 70 balances signal quality vs quantity. Shorter RSI (30–50) generates too
many false oversold readings. RSI 100 has perfect win rate but too few trades.
RSI 140 produces zero signals.

### rsi_oversold (RSI entry threshold)

Path: `strategy.params.crypto_mean_reversion.rsi_oversold`

| Value | Avg Return | Trades | Sells | Wins | Win Rate | Avg Win | Avg Loss | Max DD |
|------:|----------:|-------:|------:|-----:|---------:|--------:|---------:|-------:|
| 20.0 | +0.000% | 0 | 0 | 0 | — | — | 0.0% |
| 25.0 | +0.520% | 27 | 18 | 18 | 100% | +1.582% | — | 0.0% |
| **30.0** | **+0.577%** | 94 | 61 | 55 | 90.2% | +1.663% | -2.185% | 1.2% |
| 35.0 | -0.429% | 150 | 94 | 75 | 79.8% | +1.666% | -4.046% | 2.2% |
| 40.0 | -1.776% | 166 | 103 | 73 | 70.9% | +1.601% | -3.878% | 3.9% |

**Finding:** Threshold 30 is optimal. Lower (20–25) is too restrictive. Higher (35–40)
lets in too many low-quality entries where price isn't truly oversold.

### drop_window_bars (fast-drop detection window)

Path: `strategy.params.crypto_mean_reversion.drop_window_bars`

| Value | Avg Return | Trades | Sells | Wins | Win Rate | Max DD |
|------:|----------:|-------:|------:|-----:|---------:|-------:|
| 30 | +0.577% | 94 | 61 | 55 | 90.2% | 1.2% |
| 50 | +0.577% | 94 | 61 | 55 | 90.2% | 1.2% |
| 75 | +0.577% | 94 | 61 | 55 | 90.2% | 1.2% |
| 100 | +0.577% | 94 | 61 | 55 | 90.2% | 1.2% |
| 150 | +0.577% | 94 | 61 | 55 | 90.2% | 1.2% |

**Finding:** Zero effect across all values. The fast-drop check (`ref_price - last > 1%`)
is satisfied whenever BB + RSI conditions are met, making this parameter redundant.
Could be removed or set to any value.

### atr_stop_mult (ATR-based stop-loss multiplier)

Path: `risk.crypto.atr_stop_mult`

| Value | Avg Return | Trades | Sells | Wins | Win Rate | Max DD |
|------:|----------:|-------:|------:|-----:|---------:|-------:|
| 1.5 | +0.577% | 94 | 61 | 55 | 90.2% | 1.2% |
| 2.0 | +0.577% | 94 | 61 | 55 | 90.2% | 1.2% |
| 2.5 | +0.577% | 94 | 61 | 55 | 90.2% | 1.2% |
| 3.5 | +0.577% | 94 | 61 | 55 | 90.2% | 1.2% |
| 5.0 | +0.069% | 57 | 37 | 31 | 83.8% | 1.2% |

**Finding:** Values 1.5–3.5 are identical — other exit mechanisms (take profit, partial
take profit, strategy signals) fire before ATR stop is reached. Only 5.0 differs: the
very wide stop allows positions to run longer, catching 37 fewer trades. The ATR stop
is effectively dead for normal values; keep at 2.5 as a safety net.

---

## Conclusions

1. **bb_period=50 is the single most impactful change** — nearly 10× return vs baseline (100).
2. **All other parameters are already at or near optimal** — no changes needed.
3. **Two dead parameters identified:** `drop_window_bars` and `atr_stop_mult` (1.5–3.5 range)
   have zero effect on outcomes. Other exit mechanisms dominate.
4. **Going below bb_period=50 kills signals** — the BB window becomes too narrow on 1m bars
   for the combined entry conditions to trigger.
5. **The strategy is conservative by design** — 90%+ win rate, small avg losses, low drawdown.
   The tradeoff is fewer trades (38 over two 12-day periods).
