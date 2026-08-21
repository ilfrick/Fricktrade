#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""Download multi-year US equity DAILY bars from Alpaca for swing-horizon research.

Daily bars, not intraday, because accounts below the $25k PDT threshold cannot
day-trade — so any realistic equity strategy here holds for days to weeks, and
that is the horizon which must be validated.

Universe selection is deliberate. Hand-picking today's mega-caps would bake in
survivorship *and* selection bias, so this samples broadly from Alpaca's active
tradable list and then filters on liquidity measured from the downloaded bars.

    python3 scripts/download_equity_history.py --years 10 --max-symbols 300

SURVIVORSHIP BIAS — read before trusting any positive result:
Alpaca's asset list contains only currently-listed companies. Anything that was
delisted, acquired, or went to zero is absent. This flatters dip-buying badly:
every drawdown in this universe was, by construction, survived. A strategy that
FAILS here fails for real; a strategy that PASSES here needs heavy discounting
and a survivorship-free dataset before it means anything.
"""
from __future__ import annotations

import argparse
import logging
import re
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _credentials() -> tuple[str, str]:
    """Read Alpaca credentials from the environment or .env."""
    key = os.getenv("ALPACA_API_KEY", "").strip()
    sec = os.getenv("ALPACA_API_SECRET", "").strip()
    if not key or not sec:
        keys = [k.strip() for k in os.getenv("ALPACA_API_KEYS", "").split(",") if k.strip()]
        secs = [s.strip() for s in os.getenv("ALPACA_API_SECRETS", "").split(",") if s.strip()]
        if keys and secs:
            key, sec = keys[0], secs[0]
    if not key or not sec:
        env = Path(__file__).resolve().parent.parent / ".env"
        if env.exists():
            vals: dict[str, str] = {}
            for line in env.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    vals[k.strip()] = v.strip().strip('"').strip("'")
            key = key or vals.get("ALPACA_API_KEY", "")
            sec = sec or vals.get("ALPACA_API_SECRET", "")
            if not key or not sec:
                ks = [x.strip() for x in vals.get("ALPACA_API_KEYS", "").split(",") if x.strip()]
                ss = [x.strip() for x in vals.get("ALPACA_API_SECRETS", "").split(",") if x.strip()]
                if ks and ss:
                    key, sec = ks[0], ss[0]
    if not key or not sec:
        raise SystemExit("No Alpaca credentials found (ALPACA_API_KEY/SECRET or .env)")
    return key, sec


_CLEAN_TICKER = re.compile(r"^[A-Z]{1,5}$")


def _usable_ticker(sym: str) -> bool:
    """Reject malformed and non-US-listing tickers.

    The inactive list contains artefacts the data API rejects outright
    ('097ESC024', 'CALL_DELISTED') plus foreign ordinaries and ADRs traded OTC,
    which are not the universe under study.
    """
    from app.data.scanner import _is_excluded_symbol
    if not sym or not _CLEAN_TICKER.match(sym):
        return False
    if len(sym) == 5 and sym.endswith("F"):     # foreign ordinary, OTC
        return False
    return not _is_excluded_symbol(sym)


def build_universe(key: str, secret: str, max_symbols: int, seed: int,
                   include_delisted: bool = False) -> list[str]:
    """Random sample of US equities — not a curated winners list.

    With include_delisted, the sample is drawn from active AND inactive assets.
    Alpaca serves full daily history for delisted names (verified: SIVB ends
    2023-03-09, FRC ends 2023-04-28 at $3.51), so including them removes the
    survivorship bias that otherwise flatters any dip-buying strategy.
    """
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass, AssetStatus

    client = TradingClient(key, secret, raw_data=True)
    statuses = [AssetStatus.ACTIVE]
    if include_delisted:
        statuses.append(AssetStatus.INACTIVE)

    symbols: list[str] = []
    counts: dict[str, int] = {}
    for st in statuses:
        assets = client.get_all_assets(
            GetAssetsRequest(status=st, asset_class=AssetClass.US_EQUITY)
        )
        got = [a["symbol"] for a in assets
               if a.get("symbol") and _usable_ticker(a["symbol"])]
        counts[st.value] = len(got)
        symbols.extend(got)
    log.info("Universe pool: %s (total %d)",
             ", ".join(f"{k}={v}" for k, v in counts.items()), len(symbols))
    symbols = sorted(set(symbols))
    rng = random.Random(seed)
    rng.shuffle(symbols)
    return sorted(symbols[:max_symbols])


def download(symbols: list[str], years: int, key: str, secret: str,
             out_dir: Path, chunk: int = 100) -> dict[str, pd.DataFrame]:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(key, secret)
    end = datetime.now(timezone.utc) - timedelta(days=1)
    start = end - timedelta(days=int(years * 365.25))
    frames: dict[str, pd.DataFrame] = {}

    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        try:
            req = StockBarsRequest(symbol_or_symbols=batch, timeframe=TimeFrame.Day,
                                   start=start, end=end, adjustment="all")
            df = client.get_stock_bars(req).df
        except Exception as exc:
            # A single unusable ticker 400s the whole batch, so fall back to
            # halving it rather than discarding 100 good symbols.
            log.warning("Batch %d-%d failed (%s); retrying in halves", i, i + len(batch), exc)
            df = None
            if len(batch) > 1:
                mid = len(batch) // 2
                for half in (batch[:mid], batch[mid:]):
                    frames.update(download(half, years, key, secret, out_dir, chunk=len(half)))
            continue
        if df is None or df.empty:
            continue
        for sym in batch:
            try:
                sub = df.xs(sym, level=0)
            except KeyError:
                continue
            if sub.empty:
                continue
            out = pd.DataFrame({
                "Datetime": pd.to_datetime(sub.index).tz_localize(None),
                "Open": sub["open"].to_numpy(dtype=float),
                "High": sub["high"].to_numpy(dtype=float),
                "Low": sub["low"].to_numpy(dtype=float),
                "Close": sub["close"].to_numpy(dtype=float),
                "Volume": sub["volume"].to_numpy(dtype=float),
            })
            frames[sym] = out
        log.info("  fetched %d/%d symbols", len(frames), len(symbols))
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--max-symbols", type=int, default=300)
    ap.add_argument("--min-dollar-volume", type=float, default=5_000_000.0,
                    help="median daily $ volume required to keep a symbol")
    ap.add_argument("--min-bars", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-delisted", action="store_true",
                    help="sample from inactive/delisted assets too — removes survivorship bias")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent
                                             / "data" / "equity_1d"))
    args = ap.parse_args()

    key, secret = _credentials()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    universe = build_universe(key, secret, args.max_symbols, args.seed,
                              include_delisted=args.include_delisted)
    log.info("Sampled %d symbols; downloading %d years of daily bars", len(universe), args.years)
    frames = download(universe, args.years, key, secret, out_dir)

    kept = dropped_liq = dropped_short = 0
    for sym, df in frames.items():
        if len(df) < args.min_bars:
            dropped_short += 1
            continue
        dollar_vol = float((df["Close"] * df["Volume"]).median())
        if dollar_vol < args.min_dollar_volume:
            dropped_liq += 1
            continue
        df.to_csv(out_dir / f"{sym}_1d.csv", index=False)
        kept += 1

    log.info("Kept %d symbols → %s", kept, out_dir)
    log.info("Dropped: %d too illiquid, %d too little history", dropped_liq, dropped_short)
    if kept:
        any_df = next(iter(frames.values()))
        log.info("Range: %s → %s", any_df["Datetime"].min().date(), any_df["Datetime"].max().date())
    if args.include_delisted:
        log.info("Universe INCLUDES delisted names — survivorship bias removed.")
    else:
        log.warning("SURVIVORSHIP BIAS: universe contains only currently-listed names. "
                    "A positive result here is optimistic; a negative result is "
                    "trustworthy. Re-run with --include-delisted to remove it.")


if __name__ == "__main__":
    main()
