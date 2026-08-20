#!/usr/bin/env python3
"""Check how often 2+ strategies agree on buy across all symbols."""
from __future__ import annotations
import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from app.utils.config import load_config
from app.learning.indicators import compute_all_indicators

logging.basicConfig(level=logging.WARNING)

cfg = load_config("config/config.yaml")

from app.strategies.crypto_momentum import CryptoMomentumStrategy
from app.strategies.crypto_mean_reversion import CryptoMeanReversionStrategy
from app.strategies.trend_following import TrendFollowingStrategy

cm = CryptoMomentumStrategy(cfg)
cmr = CryptoMeanReversionStrategy(cfg)
tf = TrendFollowingStrategy(cfg)

import pandas as pd
data_dir = Path("/data")
symbols = [f.stem.replace("_5m", "").replace("_", "/") for f in data_dir.glob("*_USD_5m.csv")]
print(f"Testing {len(symbols)} symbols")

total_bars = 0
single_buy = 0
double_buy = 0
triple_buy = 0
per_strat = {"crypto_momentum": 0, "crypto_mean_reversion": 0, "trend_following": 0}

for sym in sorted(symbols):
    fname = sym.replace("/", "_")
    df = pd.read_csv(data_dir / f"{fname}_5m.csv")
    sym_buys_2 = 0
    for i in range(200, len(df), 3):  # sample every 3rd bar for speed
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
            "spread_pct": None, "symbol": sym, "catalyst": False,
        }
        ms["indicators"] = compute_all_indicators(
            np.array(o, dtype=float), np.array(h, dtype=float),
            np.array(l, dtype=float), np.array(p, dtype=float),
            np.array(v, dtype=float),
        )
        buys = []
        try:
            s = cm.generate_signal(ms)
            if s.get("action") == "buy":
                buys.append("cm")
                per_strat["crypto_momentum"] += 1
        except: pass
        try:
            s = cmr.generate_signal(ms)
            if s.get("action") == "buy":
                buys.append("cmr")
                per_strat["crypto_mean_reversion"] += 1
        except: pass
        try:
            s = tf.generate_signal(ms)
            if s.get("action") == "buy":
                buys.append("tf")
                per_strat["trend_following"] += 1
        except: pass

        if len(buys) >= 1: single_buy += 1
        if len(buys) >= 2:
            double_buy += 1
            sym_buys_2 += 1
        if len(buys) >= 3: triple_buy += 1

    if sym_buys_2 > 0:
        print(f"  {sym}: {sym_buys_2} bars with 2+ buy votes")

print(f"\nTotal bars sampled: {total_bars}")
print(f"Per-strategy buy counts:")
for name, count in per_strat.items():
    print(f"  {name}: {count} ({count/total_bars*100:.2f}%)")
print(f"\n1+ strategy buy: {single_buy} ({single_buy/total_bars*100:.2f}%)")
print(f"2+ strategies buy (threshold): {double_buy} ({double_buy/total_bars*100:.2f}%)")
print(f"3 strategies buy: {triple_buy} ({triple_buy/total_bars*100:.2f}%)")
