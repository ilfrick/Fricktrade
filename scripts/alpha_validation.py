#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""Alpha validation harness — does a strategy configuration actually have edge?

Built 2026-08-19 after the full audit found that crypto_mean_reversion has no
out-of-sample edge and that its gross edge is smaller than its own transaction
costs. Run this BEFORE putting capital behind any strategy change.

It answers three questions, in increasing order of strictness:

  1. signal   — does the entry signal predict forward returns at all, versus the
                unconditional baseline? (If not, no exit tuning can rescue it.)
  2. walk     — sequential walk-forward net expectancy per fold, after realistic
                costs. A real edge is positive in most folds, not just on average.
  3. holdout  — train on the first 75%, report the untouched last 25% ONCE.

Why per-trade expectancy rather than total return: total return is dominated by
position sizing and market drift, which hides whether each decision adds value.
Expectancy after costs is the thing that must be positive for the system to make
money at any size.

Usage:
    python3 scripts/alpha_validation.py signal
    python3 scripts/alpha_validation.py walk    --folds 6
    python3 scripts/alpha_validation.py holdout
    python3 scripts/alpha_validation.py all --rsi-period 30 --rsi-oversold 25

Costs default to config.yaml's backtest block (commission 0.05%/side, 3bps
slippage, 5bps spread) — i.e. ~0.21% per round trip. Any measured gross edge
below that is not tradeable.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path("/data")
HORIZONS_MIN = [15, 30, 60, 120, 240, 480]


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Costs:
    commission_pct: float = 0.05      # per side, percent
    slippage_bps: float = 3.0
    spread_bps: float = 5.0

    @property
    def side_bump(self) -> float:
        """Fraction added to buys / subtracted from sells (slippage + half spread)."""
        return (self.slippage_bps + self.spread_bps / 2.0) / 10_000.0

    @property
    def round_trip_pct(self) -> float:
        return 2.0 * (self.commission_pct + self.side_bump * 100.0)


# ---------------------------------------------------------------------------
# Strategy parameters (mirror config.yaml strategy.params.crypto_mean_reversion
# and the crypto exit ladder in trader._check_position_exit)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Entry:
    bb_period: int = 50
    bb_std: float = 2.0
    rsi_period: int = 70
    rsi_oversold: float = 30.0
    drop_window: int = 75
    drop_pct: float = 1.0
    trend_filter_bars: int = 0      # 0 disables; else require close > SMA(n)


@dataclass(frozen=True)
class ExitLadder:
    hard_stop: float = 5.0              # risk.crypto / MR override
    trailing: float = 4.5               # risk.crypto.trailing_stop_pct
    take_profit: float = 5.0            # risk.crypto.take_profit_pct
    partial_tp: float = 1.0             # risk.partial_take_profit_pct (0 disables)
    partial_ratio: float = 0.5
    sma_exit_min_profit: float = 0.5    # MR strategy sell; negative disables
    min_hold_bars: int = 15             # strategy.params.min_hold_minutes
    use_peak_detect: bool = True
    time_exit_bars: int = 0             # strategy.params.position_horizon_minutes


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    ag = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    al = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = ag / al.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(100.0)


def prepare(df: pd.DataFrame, e: Entry) -> dict:
    close = df["Close"].astype(float)
    sma = close.rolling(e.bb_period).mean()
    std = close.rolling(e.bb_period).std(ddof=1)
    lower = sma - e.bb_std * std
    upper = sma + e.bb_std * std
    rsi = wilder_rsi(close, e.rsi_period)
    ref = close.shift(e.drop_window)
    fast_drop = ((ref - close) / ref * 100.0) > e.drop_pct
    ok = (close < lower) & (rsi < e.rsi_oversold) & fast_drop & sma.notna() & std.notna()
    if e.trend_filter_bars > 0:
        long_sma = close.rolling(e.trend_filter_bars).mean()
        ok = ok & (close > long_sma) & long_sma.notna()
    return {
        "close": close.to_numpy(),
        "low": df["Low"].astype(float).to_numpy(),
        "sma": sma.to_numpy(),
        "upper": upper.to_numpy(),
        "rsi": rsi.to_numpy(),
        "entries": np.flatnonzero(ok.to_numpy()),
    }


# ---------------------------------------------------------------------------
# Trade simulation (mirrors the live exit priority order)
# ---------------------------------------------------------------------------

def simulate(arr: dict, x: ExitLadder, costs: Costs, max_hold_bars: int = 60 * 24 * 3) -> list[dict]:
    c, lo, sma, upper, rsi = arr["close"], arr["low"], arr["sma"], arr["upper"], arr["rsi"]
    n = len(c)
    bump = costs.side_bump
    comm = costs.commission_pct / 100.0
    trades: list[dict] = []
    cursor = 0

    for i in arr["entries"]:
        if i < cursor:
            continue                      # already in a position
        entry_px = c[i] * (1.0 + bump)
        peak = c[i]
        took_partial = False
        realized = 0.0
        remaining = 1.0
        reason = "max_hold"
        j_end = min(i + max_hold_bars, n - 1)
        j = i + 1

        while j <= j_end:
            px = c[j]
            peak = max(peak, px)
            if j - i >= x.min_hold_bars:
                pnl = (px - entry_px) / entry_px * 100.0
                stop_px = entry_px * (1.0 - x.hard_stop / 100.0)
                if lo[j] <= stop_px:
                    # Conservative fill: if the bar closed through the stop we
                    # assume the worse of stop price and close, never better.
                    fill = min(stop_px, px)
                    realized += remaining * (fill * (1 - bump) / entry_px - 1.0)
                    remaining, reason = 0.0, "stop"
                    break
                if x.trailing > 0 and (px - peak) / peak * 100.0 <= -x.trailing:
                    realized += remaining * (px * (1 - bump) / entry_px - 1.0)
                    remaining, reason = 0.0, "trailing"
                    break
                if x.take_profit > 0 and pnl >= x.take_profit:
                    realized += remaining * (px * (1 - bump) / entry_px - 1.0)
                    remaining, reason = 0.0, "take_profit"
                    break
                if x.partial_tp > 0 and not took_partial and pnl >= x.partial_tp:
                    took_partial = True
                    part = remaining * x.partial_ratio
                    realized += part * (px * (1 - bump) / entry_px - 1.0)
                    remaining -= part
                if (x.use_peak_detect and pnl > 0.5 and rsi[j] > 70.0
                        and upper[j] > 0 and px >= upper[j]):
                    realized += remaining * (px * (1 - bump) / entry_px - 1.0)
                    remaining, reason = 0.0, "peak_detect"
                    break
                if x.sma_exit_min_profit >= 0 and px >= sma[j] and pnl >= x.sma_exit_min_profit:
                    realized += remaining * (px * (1 - bump) / entry_px - 1.0)
                    remaining, reason = 0.0, "sma_exit"
                    break
                if x.time_exit_bars > 0 and (j - i) >= x.time_exit_bars:
                    realized += remaining * (px * (1 - bump) / entry_px - 1.0)
                    remaining, reason = 0.0, "time_exit"
                    break
            j += 1

        if remaining > 0.0:
            realized += remaining * (c[min(j, n - 1)] * (1 - bump) / entry_px - 1.0)
        legs = 2 + (1 if took_partial else 0)
        trades.append({
            "gross_pct": realized * 100.0,
            "net_pct": realized * 100.0 - legs * comm * 100.0,
            "bars": int(min(j, n - 1) - i),
            "reason": reason,
        })
        cursor = min(j, n - 1) + 1
    return trades


def evaluate(frames: dict[str, pd.DataFrame], e: Entry, x: ExitLadder, costs: Costs) -> dict:
    all_t: list[dict] = []
    per_sym: dict[str, float] = {}
    for sym, df in frames.items():
        t = simulate(prepare(df, e), x, costs)
        per_sym[sym] = sum(z["net_pct"] for z in t)
        all_t.extend(t)
    if not all_t:
        return {"trades": 0, "net_E": 0.0, "gross_E": 0.0, "win_rate": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "sym_pos": 0, "n_sym": len(frames)}
    net = np.array([t["net_pct"] for t in all_t])
    gross = np.array([t["gross_pct"] for t in all_t])
    w, l = net[net > 0], net[net <= 0]
    return {
        "trades": len(net),
        "net_E": float(net.mean()),
        "gross_E": float(gross.mean()),
        "win_rate": float(len(w) / len(net) * 100),
        "avg_win": float(w.mean()) if len(w) else 0.0,
        "avg_loss": float(l.mean()) if len(l) else 0.0,
        "sym_pos": int(sum(1 for v in per_sym.values() if v > 0)),
        "n_sym": len(frames),
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_frames(data_dir: Path, pattern: str = "*_USD_1m.csv") -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for p in sorted(data_dir.glob(pattern)):
        frames[p.stem] = pd.read_csv(
            p, usecols=["Datetime", "High", "Low", "Close"], parse_dates=["Datetime"]
        )
    if not frames:
        raise SystemExit(f"No CSVs matching {pattern} in {data_dir}")
    return frames


def slice_frames(frames: dict, lo_frac: float, hi_frac: float) -> dict:
    out = {}
    for s, df in frames.items():
        a, b = int(len(df) * lo_frac), int(len(df) * hi_frac)
        out[s] = df.iloc[a:b].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_signal(frames: dict, e: Entry, costs: Costs) -> None:
    print("\n=== SIGNAL PREDICTIVE VALUE ===")
    print("Forward return after an entry signal vs. the unconditional baseline.")
    print("An edge below the round-trip cost is not tradeable, however strong the t-stat.\n")
    cond = {h: [] for h in HORIZONS_MIN}
    base = {h: [] for h in HORIZONS_MIN}
    for df in frames.values():
        arr = prepare(df, e)
        c, idx, n = arr["close"], arr["entries"], len(arr["close"])
        for h in HORIZONS_MIN:
            if n <= h:
                continue
            fwd = (c[h:] - c[:-h]) / c[:-h] * 100.0
            base[h].append(fwd)
            valid = idx[idx < n - h]
            if valid.size:
                cond[h].append(fwd[valid])
    print(f"{'horizon':>8} {'signals':>9} {'signal fwd%':>12} {'baseline%':>11} "
          f"{'edge%':>8} {'t':>6}  tradeable?")
    print("-" * 72)
    for h in HORIZONS_MIN:
        if not cond[h]:
            continue
        cv, bv = np.concatenate(cond[h]), np.concatenate(base[h])
        edge = cv.mean() - bv.mean()
        t = edge / (cv.std(ddof=1) / np.sqrt(len(cv))) if len(cv) > 2 and cv.std() > 0 else 0.0
        mark = "yes" if edge > costs.round_trip_pct else "no"
        print(f"{h:>6}m {len(cv):>9} {cv.mean():>+12.4f} {bv.mean():>+11.4f} "
              f"{edge:>+8.4f} {t:>6.2f}  {mark}")
    print(f"\nround-trip cost = {costs.round_trip_pct:.3f}%")
    print("NOTE: t-stats are inflated — signals overlap in time and crypto symbols are\n"
          "      highly correlated, so effective sample size is far below the raw count.")


def cmd_walk(frames: dict, e: Entry, x: ExitLadder, costs: Costs, folds: int) -> None:
    print(f"\n=== WALK-FORWARD ({folds} sequential folds) ===")
    print(f"{'fold':>4} {'period':<21} {'trades':>7} {'gross E%':>9} {'net E%':>8} {'WR%':>6}")
    print("-" * 60)
    zero = Costs(0.0, 0.0, 0.0)
    nets, grosses = [], []
    for k in range(folds):
        sub = slice_frames(frames, k / folds, (k + 1) / folds)
        any_df = next(iter(sub.values()))
        period = f"{any_df['Datetime'].iloc[0]:%Y-%m-%d} → {any_df['Datetime'].iloc[-1]:%m-%d}"
        r = evaluate(sub, e, x, costs)
        g = evaluate(sub, e, x, zero)
        nets.append(r["net_E"])
        grosses.append(g["net_E"])
        print(f"{k+1:>4} {period:<21} {r['trades']:>7} {g['net_E']:>+9.3f} "
              f"{r['net_E']:>+8.3f} {r['win_rate']:>6.1f}")
    print("-" * 60)
    pos = sum(1 for v in nets if v > 0)
    print(f"{'':>4} {'MEAN':<21} {'':>7} {np.mean(grosses):>+9.3f} {np.mean(nets):>+8.3f}")
    print(f"\nfolds with positive NET expectancy: {pos}/{folds}")
    if pos <= folds // 2 or np.mean(nets) <= 0:
        print("VERDICT: no reliable edge — do not allocate capital to this configuration.")
    else:
        print("VERDICT: candidate shows edge; confirm on the hold-out before allocating.")


def cmd_holdout(frames: dict, e: Entry, x: ExitLadder, costs: Costs) -> None:
    print("\n=== TRAIN / HOLD-OUT ===")
    train = slice_frames(frames, 0.0, 0.75)
    hold = slice_frames(frames, 0.75, 1.0)
    print(f"{'split':<10} {'trades':>7} {'gross E%':>9} {'net E%':>8} {'WR%':>6} "
          f"{'win%':>6} {'loss%':>7} {'sym+':>6}")
    print("-" * 64)
    for name, sub in (("train", train), ("holdout", hold)):
        r = evaluate(sub, e, x, costs)
        print(f"{name:<10} {r['trades']:>7} {r['gross_E']:>+9.3f} {r['net_E']:>+8.3f} "
              f"{r['win_rate']:>6.1f} {r['avg_win']:>+6.2f} {r['avg_loss']:>+7.2f} "
              f"{r['sym_pos']:>3}/{r['n_sym']}")
    ho = evaluate(hold, e, x, costs)
    print()
    if ho["net_E"] > 0:
        print("HOLD-OUT PASSES: net expectancy positive on unseen data.")
    else:
        print("HOLD-OUT FAILS: negative net expectancy on unseen data. Do not deploy.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["signal", "walk", "holdout", "all"])
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--pattern", default="*_USD_1m.csv")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--rsi-period", type=int, default=Entry.rsi_period)
    ap.add_argument("--rsi-oversold", type=float, default=Entry.rsi_oversold)
    ap.add_argument("--bb-period", type=int, default=Entry.bb_period)
    ap.add_argument("--bb-std", type=float, default=Entry.bb_std)
    ap.add_argument("--trend-filter-bars", type=int, default=Entry.trend_filter_bars)
    ap.add_argument("--hard-stop", type=float, default=ExitLadder.hard_stop)
    ap.add_argument("--time-exit-bars", type=int, default=ExitLadder.time_exit_bars)
    ap.add_argument("--commission-pct", type=float, default=Costs.commission_pct)
    args = ap.parse_args()

    entry = Entry(bb_period=args.bb_period, bb_std=args.bb_std,
                  rsi_period=args.rsi_period, rsi_oversold=args.rsi_oversold,
                  trend_filter_bars=args.trend_filter_bars)
    ladder = ExitLadder(hard_stop=args.hard_stop, time_exit_bars=args.time_exit_bars)
    costs = Costs(commission_pct=args.commission_pct)

    frames = load_frames(Path(args.data_dir), args.pattern)
    print(f"symbols={len(frames)}  entry=rsi{entry.rsi_period}<{entry.rsi_oversold:g} "
          f"bb{entry.bb_period}/{entry.bb_std:g}  stop={ladder.hard_stop:g}%  "
          f"round_trip_cost={costs.round_trip_pct:.3f}%")

    if args.command in ("signal", "all"):
        cmd_signal(frames, entry, costs)
    if args.command in ("walk", "all"):
        cmd_walk(frames, entry, ladder, costs, args.folds)
    if args.command in ("holdout", "all"):
        cmd_holdout(frames, entry, ladder, costs)


if __name__ == "__main__":
    main()
