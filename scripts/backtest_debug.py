#!/usr/bin/env python3
"""Quick diagnostic: run a few bars through the strategies and print signals."""
from __future__ import annotations
import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

from app.utils.config import load_config
from app.learning.indicators import compute_all_indicators

cfg = load_config("config/config.yaml")
strat_params = cfg.get("strategy", {}).get("params", {})

# Load BTC data
df = pd.read_csv("/data/BTC_USD_5m.csv")
print(f"BTC/USD: {len(df)} rows, columns: {list(df.columns)}")
print(f"Date range: {df['Datetime'].iloc[0]} → {df['Datetime'].iloc[-1]}")

# Build a market_state like the backtest engine would
prices = df["Close"].values[-200:].tolist()
volumes = df["Volume"].values[-200:].tolist()
opens = df["Open"].values[-200:].tolist()
highs = df["High"].values[-200:].tolist()
lows = df["Low"].values[-200:].tolist()

market_state = {
    "prices": prices,
    "volumes": volumes,
    "opens": opens,
    "highs": highs,
    "lows": lows,
    "session_prices": prices[-60:],
    "session_volumes": volumes[-60:],
    "session_opens": opens[-60:],
    "session_highs": highs[-60:],
    "session_lows": lows[-60:],
    "session_bar_count": 60,
    "last_price": prices[-1],
    "session_volume": sum(volumes[-60:]),
    "relative_volume": 1.0,
    "session_gain_pct": (prices[-1] - prices[-60]) / prices[-60] * 100 if prices[-60] else 0,
    "spread_pct": None,
    "symbol": "BTC/USD",
    "catalyst": False,
}

# Compute indicators like the agent does
_closes = np.array(prices, dtype=float)
_opens = np.array(opens, dtype=float)
_highs = np.array(highs, dtype=float)
_lows = np.array(lows, dtype=float)
_vols = np.array(volumes, dtype=float)
indicators = compute_all_indicators(_opens, _highs, _lows, _closes, _vols)
market_state["indicators"] = indicators
print(f"\nIndicators computed: {sorted(indicators.keys())}")
print(f"  RSI: {indicators.get('rsi'):.2f}" if indicators.get('rsi') is not None else "  RSI: None")
print(f"  ATR: {indicators.get('atr'):.4f}" if indicators.get('atr') is not None else "  ATR: None")

# Test each strategy
print("\n--- Strategy Signals ---")

from app.strategies.crypto_momentum import CryptoMomentumStrategy
cm = CryptoMomentumStrategy(cfg)
sig = cm.generate_signal(market_state)
print(f"crypto_momentum: {sig}")

from app.strategies.crypto_mean_reversion import CryptoMeanReversionStrategy
cmr = CryptoMeanReversionStrategy(cfg)
sig = cmr.generate_signal(market_state)
print(f"crypto_mean_reversion: {sig}")

from app.strategies.trend_following import TrendFollowingStrategy
tf = TrendFollowingStrategy(cfg)
sig = tf.generate_signal(market_state)
print(f"trend_following: {sig}")

# Now try on multiple snapshots to find any buy signals
print("\n--- Scanning 8000 bars for buy signals ---")
buy_counts = {"crypto_momentum": 0, "crypto_mean_reversion": 0, "trend_following": 0}
total_bars = 0
for i in range(200, min(len(df), 8200)):
    total_bars += 1
    p = df["Close"].values[max(0, i-200):i].tolist()
    v = df["Volume"].values[max(0, i-200):i].tolist()
    o = df["Open"].values[max(0, i-200):i].tolist()
    h = df["High"].values[max(0, i-200):i].tolist()
    l = df["Low"].values[max(0, i-200):i].tolist()
    if len(p) < 20:
        continue
    ms = {
        "prices": p, "volumes": v, "opens": o, "highs": h, "lows": l,
        "session_prices": p[-60:], "session_volumes": v[-60:],
        "session_opens": o[-60:], "session_highs": h[-60:], "session_lows": l[-60:],
        "session_bar_count": min(60, len(p)),
        "last_price": p[-1], "session_volume": sum(v[-60:]),
        "relative_volume": 1.0,
        "session_gain_pct": (p[-1] - p[-60]) / p[-60] * 100 if len(p) >= 60 and p[-60] else 0,
        "spread_pct": None, "symbol": "BTC/USD", "catalyst": False,
    }
    _c = np.array(p, dtype=float)
    _o = np.array(o, dtype=float)
    _h = np.array(h, dtype=float)
    _l = np.array(l, dtype=float)
    _v = np.array(v, dtype=float)
    ms["indicators"] = compute_all_indicators(_o, _h, _l, _c, _v)

    try:
        s = cm.generate_signal(ms)
        if s.get("action") == "buy":
            buy_counts["crypto_momentum"] += 1
    except:
        pass
    try:
        s = cmr.generate_signal(ms)
        if s.get("action") == "buy":
            buy_counts["crypto_mean_reversion"] += 1
    except:
        pass
    try:
        s = tf.generate_signal(ms)
        if s.get("action") == "buy":
            buy_counts["trend_following"] += 1
    except:
        pass

print(f"Scanned {total_bars} bars")
for name, count in buy_counts.items():
    print(f"  {name}: {count} buy signals ({count/total_bars*100:.1f}%)")
