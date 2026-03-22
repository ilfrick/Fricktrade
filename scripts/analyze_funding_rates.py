#!/usr/bin/env python3
"""
Funding Rate Arbitrage Analysis — Binance USDT Perpetuals

Strategy:
  - When funding rate > min_rate (0.03%), go SHORT perp + LONG spot (collect funding)
  - When funding rate < exit_rate (0.01%) or flips negative, close both legs
  - Collect funding payments every 8h while position is open

Fee model (Binance with BNB discount):
  - Futures taker: 0.075% per side
  - Spot taker:    0.075% per side
  - Round-trip total: 4 * 0.075% = 0.30%

Risk:
  - Basis risk: spot vs perp price can diverge, creating mark-to-market drawdown
  - We model this via historical spot-perp spread (approximated from funding rate magnitude)

Output: ranked table of symbols by net annualized APY after fees.

Usage: python3 scripts/analyze_funding_rates.py
"""

import time
import sys
import math
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

TOP_N_SYMBOLS = 20          # top N by 24h quote volume
LOOKBACK_DAYS = 90          # historical window
MIN_FUNDING_RATE = 0.0003   # 0.03% — entry threshold
EXIT_FUNDING_RATE = 0.0001  # 0.01% — exit threshold
FEE_PER_SIDE = 0.00075      # 0.075% taker fee per side (with BNB discount)
ROUND_TRIP_FEE = 4 * FEE_PER_SIDE  # 0.30% total (open spot + open perp + close spot + close perp)
NOTIONAL_PER_TRADE = 10_000  # $10k notional per position
FUNDING_INTERVAL_HOURS = 8  # Binance standard
MAX_API_RETRIES = 3
API_DELAY = 0.15            # rate limit courtesy delay (seconds)
RISK_FREE_RATE = 0.05       # 5% annual for Sharpe calculation

# Binance public endpoints
BASE_URL = "https://fapi.binance.com"
TICKER_URL = f"{BASE_URL}/fapi/v1/ticker/24hr"
FUNDING_URL = f"{BASE_URL}/fapi/v1/fundingRate"

# ──────────────────────────────────────────────────────────────────────────────
# API Helpers
# ──────────────────────────────────────────────────────────────────────────────

session = requests.Session()
session.headers.update({"User-Agent": "FricktradeAnalysis/1.0"})


def api_get(url: str, params: Optional[dict] = None, retries: int = MAX_API_RETRIES) -> Optional[list]:
    """GET with retry and rate limit handling."""
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 5))
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 451:
                print(f"  ERROR: Binance API geo-blocked (HTTP 451). Use a VPN or proxy.")
                return None
            if resp.status_code == 403:
                print(f"  ERROR: Binance API access forbidden (HTTP 403).")
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError as e:
            print(f"  Connection error (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
        except requests.exceptions.Timeout:
            print(f"  Timeout (attempt {attempt+1}/{retries})")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
        except requests.exceptions.HTTPError as e:
            print(f"  HTTP error: {e}")
            return None
    return None


def get_top_symbols(n: int = TOP_N_SYMBOLS) -> list[str]:
    """Get top N USDT perp symbols by 24h quote volume."""
    print(f"Fetching 24h ticker data to rank symbols by volume...")
    data = api_get(TICKER_URL)
    if data is None:
        return []

    # Filter USDT pairs only, exclude stablecoins and weird pairs
    exclude_bases = {"USDC", "BUSD", "TUSD", "FDUSD", "DAI", "USDP", "EUR"}
    usdt_pairs = []
    for t in data:
        sym = t.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        base = sym.replace("USDT", "")
        if base in exclude_bases:
            continue
        try:
            vol = float(t.get("quoteVolume", 0))
        except (ValueError, TypeError):
            continue
        usdt_pairs.append((sym, vol))

    usdt_pairs.sort(key=lambda x: x[1], reverse=True)
    top = [s for s, _ in usdt_pairs[:n]]
    print(f"Top {n} symbols by 24h volume: {', '.join(top)}")
    return top


def fetch_funding_rates(symbol: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Fetch historical funding rates with pagination (max 1000 per call)."""
    end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ts = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    all_records = []
    current_start = start_ts

    while current_start < end_ts:
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "endTime": end_ts,
            "limit": 1000,
        }
        data = api_get(FUNDING_URL, params=params)
        if data is None or len(data) == 0:
            break

        all_records.extend(data)

        # Move start to after last record
        last_ts = data[-1]["fundingTime"]
        if last_ts <= current_start:
            break  # no progress, avoid infinite loop
        current_start = last_ts + 1

        if len(data) < 1000:
            break  # got all data

        time.sleep(API_DELAY)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df = df.sort_values("fundingTime").reset_index(drop=True)
    df = df.drop_duplicates(subset=["fundingTime"])
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Simulation Engine
# ──────────────────────────────────────────────────────────────────────────────

def simulate_funding_arb(df: pd.DataFrame, symbol: str) -> dict:
    """
    Simulate funding rate arbitrage on historical data.

    Strategy:
      - ENTRY when funding_rate > MIN_FUNDING_RATE (positive = longs pay shorts)
        We go: SHORT perp (collect funding) + LONG spot (hedge delta)
      - EXIT when funding_rate < EXIT_FUNDING_RATE or funding_rate < 0
      - While in position, collect funding_rate * notional every 8h

    Returns dict with performance metrics.
    """
    if df.empty:
        return _empty_result(symbol)

    in_position = False
    trades = []
    current_trade = None
    cumulative_pnl = 0.0
    peak_pnl = 0.0
    max_drawdown = 0.0
    daily_pnls = {}  # date -> pnl for that day

    # Track basis risk: model adverse price move as proportional to funding rate volatility
    # In real arb, basis risk comes from spot-perp spread widening against you
    # We approximate: basis move per period ~ N(0, funding_rate_std * some_multiplier)
    # For simplicity, we assume perfect hedge (zero basis risk on PnL),
    # but track hypothetical basis drawdown from funding rate reversals

    for i in range(len(df)):
        row = df.iloc[i]
        rate = row["fundingRate"]
        ts = row["fundingTime"]
        day_key = ts.strftime("%Y-%m-%d")

        if not in_position:
            # Check entry condition: rate must be positive and above threshold
            if rate > MIN_FUNDING_RATE:
                in_position = True
                current_trade = {
                    "symbol": symbol,
                    "entry_time": ts,
                    "entry_rate": rate,
                    "funding_collected": 0.0,
                    "n_funding_periods": 0,
                    "entry_fee": ROUND_TRIP_FEE * NOTIONAL_PER_TRADE,
                    "rates": [],
                    # Basis risk tracking: model as cumulative adverse move
                    "basis_pnl": 0.0,
                    "min_basis_pnl": 0.0,
                }
        else:
            # We are in position — collect funding this period
            # When we're SHORT perp and rate is positive, we RECEIVE funding
            # When rate is negative, we PAY funding
            funding_payment = rate * NOTIONAL_PER_TRADE
            current_trade["funding_collected"] += funding_payment
            current_trade["n_funding_periods"] += 1
            current_trade["rates"].append(rate)

            # Model basis risk: random walk component proportional to rate change
            # In practice, when funding spikes, basis can widen against you temporarily
            if i > 0:
                prev_rate = df.iloc[i - 1]["fundingRate"]
                # Basis adverse move: if rate drops suddenly, perp price may move against short
                # Model: basis PnL impact = -abs(rate_change) * notional * scaling_factor
                rate_change = rate - prev_rate
                # If rate drops while we're short perp, basis moves against us
                basis_impact = -abs(rate_change) * NOTIONAL_PER_TRADE * 0.5
                current_trade["basis_pnl"] += basis_impact
                current_trade["min_basis_pnl"] = min(
                    current_trade["min_basis_pnl"], current_trade["basis_pnl"]
                )

            # Check exit condition
            exit_signal = rate < EXIT_FUNDING_RATE or rate < 0

            # Also exit on last data point
            if exit_signal or i == len(df) - 1:
                in_position = False
                exit_time = ts
                hold_hours = (exit_time - current_trade["entry_time"]).total_seconds() / 3600

                # Net PnL = funding collected - round-trip fees
                net_pnl = current_trade["funding_collected"] - current_trade["entry_fee"]
                cumulative_pnl += net_pnl

                # Track peak and drawdown
                peak_pnl = max(peak_pnl, cumulative_pnl)
                drawdown = peak_pnl - cumulative_pnl
                max_drawdown = max(max_drawdown, drawdown)

                # Assign PnL to day for Sharpe calculation
                if day_key not in daily_pnls:
                    daily_pnls[day_key] = 0.0
                daily_pnls[day_key] += net_pnl

                trades.append({
                    "symbol": symbol,
                    "entry_time": current_trade["entry_time"],
                    "exit_time": exit_time,
                    "hold_hours": hold_hours,
                    "n_funding_periods": current_trade["n_funding_periods"],
                    "funding_collected": current_trade["funding_collected"],
                    "fees_paid": current_trade["entry_fee"],
                    "net_pnl": net_pnl,
                    "avg_rate": (
                        sum(current_trade["rates"]) / len(current_trade["rates"])
                        if current_trade["rates"]
                        else 0
                    ),
                    "max_basis_dd": abs(current_trade["min_basis_pnl"]),
                })
                current_trade = None

        # Track daily PnL for funding collected while in position
        if in_position and current_trade:
            funding_today = rate * NOTIONAL_PER_TRADE
            if day_key not in daily_pnls:
                daily_pnls[day_key] = 0.0
            # Note: we add raw funding here; fee is deducted at trade close

    return _compute_summary(symbol, trades, daily_pnls, cumulative_pnl, max_drawdown, df)


def _empty_result(symbol: str) -> dict:
    return {
        "symbol": symbol,
        "total_trades": 0,
        "total_pnl": 0.0,
        "total_funding": 0.0,
        "total_fees": 0.0,
        "annualized_apy_pct": 0.0,
        "avg_hold_hours": 0.0,
        "avg_pnl_per_trade": 0.0,
        "win_rate_pct": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
        "avg_funding_rate_bps": 0.0,
        "funding_periods_total": 0,
    }


def _compute_summary(
    symbol: str,
    trades: list[dict],
    daily_pnls: dict,
    cumulative_pnl: float,
    max_drawdown: float,
    df: pd.DataFrame,
) -> dict:
    if not trades:
        return _empty_result(symbol)

    trade_df = pd.DataFrame(trades)
    total_trades = len(trades)
    total_funding = trade_df["funding_collected"].sum()
    total_fees = trade_df["fees_paid"].sum()
    total_pnl = trade_df["net_pnl"].sum()
    avg_hold = trade_df["hold_hours"].mean()
    winners = trade_df[trade_df["net_pnl"] > 0]
    win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0

    # Annualized APY: total return on capital over the observation period, annualized
    data_span_days = (df["fundingTime"].max() - df["fundingTime"].min()).total_seconds() / 86400
    if data_span_days > 0:
        period_return = total_pnl / NOTIONAL_PER_TRADE
        annualized_return = (1 + period_return) ** (365 / data_span_days) - 1
    else:
        annualized_return = 0.0

    # Sharpe ratio from daily PnL series
    if daily_pnls:
        pnl_series = pd.Series(daily_pnls)
        # Fill missing days with 0
        all_days = pd.date_range(
            start=df["fundingTime"].min().date(),
            end=df["fundingTime"].max().date(),
            freq="D",
        )
        full_series = pd.Series(0.0, index=[d.strftime("%Y-%m-%d") for d in all_days])
        for k, v in daily_pnls.items():
            if k in full_series.index:
                full_series[k] = v
        daily_returns = full_series / NOTIONAL_PER_TRADE
        mean_daily = daily_returns.mean()
        std_daily = daily_returns.std()
        if std_daily > 0:
            daily_rf = RISK_FREE_RATE / 365
            sharpe = (mean_daily - daily_rf) / std_daily * math.sqrt(365)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Average funding rate in bps
    avg_rate_bps = df["fundingRate"].mean() * 10000

    return {
        "symbol": symbol,
        "total_trades": total_trades,
        "total_pnl": total_pnl,
        "total_funding": total_funding,
        "total_fees": total_fees,
        "annualized_apy_pct": annualized_return * 100,
        "avg_hold_hours": avg_hold,
        "avg_pnl_per_trade": total_pnl / total_trades if total_trades > 0 else 0,
        "win_rate_pct": win_rate,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "avg_funding_rate_bps": avg_rate_bps,
        "funding_periods_total": int(trade_df["n_funding_periods"].sum()),
        "max_basis_dd_per_trade": trade_df["max_basis_dd"].max() if "max_basis_dd" in trade_df.columns else 0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("FUNDING RATE ARBITRAGE ANALYSIS — Binance USDT Perpetuals")
    print("=" * 80)
    print()
    print(f"Parameters:")
    print(f"  Lookback:           {LOOKBACK_DAYS} days")
    print(f"  Entry threshold:    {MIN_FUNDING_RATE*100:.3f}% ({MIN_FUNDING_RATE*10000:.1f} bps)")
    print(f"  Exit threshold:     {EXIT_FUNDING_RATE*100:.4f}% ({EXIT_FUNDING_RATE*10000:.1f} bps)")
    print(f"  Round-trip fees:    {ROUND_TRIP_FEE*100:.2f}%")
    print(f"  Notional per trade: ${NOTIONAL_PER_TRADE:,.0f}")
    print(f"  Risk-free rate:     {RISK_FREE_RATE*100:.1f}% annual")
    print()

    # Step 1: Get top symbols
    symbols = get_top_symbols(TOP_N_SYMBOLS)
    if not symbols:
        print("\nFATAL: Could not fetch symbols from Binance API.")
        print("Possible causes: geo-blocking, network error, API down.")
        print("Try using a VPN or check https://fapi.binance.com/fapi/v1/ticker/24hr in browser.")
        sys.exit(1)

    # Step 2: Fetch funding rates and simulate for each symbol
    results = []
    for idx, symbol in enumerate(symbols, 1):
        print(f"\n[{idx}/{len(symbols)}] {symbol}: fetching funding rates...", end=" ", flush=True)
        df = fetch_funding_rates(symbol, LOOKBACK_DAYS)
        if df.empty:
            print("NO DATA")
            results.append(_empty_result(symbol))
            continue
        print(f"{len(df)} records, ", end="", flush=True)

        # Simulate
        result = simulate_funding_arb(df, symbol)
        results.append(result)
        print(
            f"{result['total_trades']} trades, "
            f"PnL=${result['total_pnl']:+.2f}, "
            f"APY={result['annualized_apy_pct']:+.1f}%"
        )
        time.sleep(API_DELAY)

    # Step 3: Build summary table
    print("\n")
    print("=" * 80)
    print("RESULTS SUMMARY — Ranked by Annualized APY (net of fees)")
    print("=" * 80)

    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values("annualized_apy_pct", ascending=False)

    # Format display table
    display_cols = {
        "symbol": "Symbol",
        "total_trades": "Trades",
        "funding_periods_total": "Funding Periods",
        "total_funding": "Gross Funding ($)",
        "total_fees": "Total Fees ($)",
        "total_pnl": "Net PnL ($)",
        "annualized_apy_pct": "Ann. APY (%)",
        "avg_hold_hours": "Avg Hold (hrs)",
        "win_rate_pct": "Win Rate (%)",
        "max_drawdown": "Max DD ($)",
        "sharpe_ratio": "Sharpe",
        "avg_funding_rate_bps": "Avg Rate (bps)",
    }

    display_df = summary_df[list(display_cols.keys())].rename(columns=display_cols)

    # Format numbers
    for col in ["Gross Funding ($)", "Total Fees ($)", "Net PnL ($)", "Max DD ($)"]:
        display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")
    for col in ["Ann. APY (%)", "Win Rate (%)", "Avg Rate (bps)"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
    display_df["Avg Hold (hrs)"] = display_df["Avg Hold (hrs)"].apply(lambda x: f"{x:.1f}")
    display_df["Sharpe"] = display_df["Sharpe"].apply(lambda x: f"{x:.2f}")

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 20)
    print()
    print(display_df.to_string(index=False))

    # Step 4: Aggregate statistics
    print("\n")
    print("=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)

    profitable = summary_df[summary_df["total_pnl"] > 0]
    losing = summary_df[summary_df["total_pnl"] <= 0]
    total_pnl_all = summary_df["total_pnl"].sum()
    total_funding_all = summary_df["total_funding"].sum()
    total_fees_all = summary_df["total_fees"].sum()
    total_capital = NOTIONAL_PER_TRADE * len(symbols)  # if running all simultaneously

    print(f"  Symbols analyzed:     {len(symbols)}")
    print(f"  Profitable symbols:   {len(profitable)}")
    print(f"  Losing symbols:       {len(losing)}")
    print(f"  Total gross funding:  ${total_funding_all:,.2f}")
    print(f"  Total fees paid:      ${total_fees_all:,.2f}")
    print(f"  Fee drag ratio:       {total_fees_all/total_funding_all*100:.1f}% of gross funding" if total_funding_all > 0 else "  Fee drag ratio: N/A")
    print(f"  Total net PnL:        ${total_pnl_all:,.2f}")
    print(f"  Capital deployed:     ${total_capital:,.0f} (if all 20 simultaneously)")
    data_span = LOOKBACK_DAYS
    if total_capital > 0:
        period_return = total_pnl_all / total_capital
        ann_return = (1 + period_return) ** (365 / data_span) - 1
        print(f"  Portfolio return:     {period_return*100:.2f}% over {data_span} days")
        print(f"  Portfolio ann. APY:   {ann_return*100:.2f}%")

    # Top/bottom performers
    print(f"\n  --- Top 5 by APY ---")
    for _, row in summary_df.head(5).iterrows():
        print(f"    {row['symbol']:12s}  APY={row['annualized_apy_pct']:+7.2f}%  Sharpe={row['sharpe_ratio']:.2f}  PnL=${row['total_pnl']:+.2f}")

    print(f"\n  --- Bottom 5 by APY ---")
    for _, row in summary_df.tail(5).iterrows():
        print(f"    {row['symbol']:12s}  APY={row['annualized_apy_pct']:+7.2f}%  Sharpe={row['sharpe_ratio']:.2f}  PnL=${row['total_pnl']:+.2f}")

    # Step 5: Verdict
    print("\n")
    print("=" * 80)
    print("VERDICT: IS FUNDING RATE ARBITRAGE WORTH BUILDING?")
    print("=" * 80)

    avg_sharpe = summary_df["sharpe_ratio"].mean()
    median_apy = summary_df["annualized_apy_pct"].median()
    pct_profitable = len(profitable) / len(summary_df) * 100 if len(summary_df) > 0 else 0
    avg_win_rate = summary_df["win_rate_pct"].mean()
    fee_drag_pct = (total_fees_all / total_funding_all * 100) if total_funding_all > 0 else 100

    print()
    verdict_points = []

    # Check 1: Is median APY attractive?
    if median_apy > 15:
        verdict_points.append(f"  [STRONG] Median APY {median_apy:.1f}% exceeds 15% threshold")
    elif median_apy > 5:
        verdict_points.append(f"  [MODERATE] Median APY {median_apy:.1f}% — decent but not exceptional")
    elif median_apy > 0:
        verdict_points.append(f"  [WEAK] Median APY {median_apy:.1f}% — barely covers opportunity cost")
    else:
        verdict_points.append(f"  [NEGATIVE] Median APY {median_apy:.1f}% — strategy loses money on average")

    # Check 2: Fee drag
    if fee_drag_pct < 30:
        verdict_points.append(f"  [STRONG] Fee drag {fee_drag_pct:.0f}% of gross — fees manageable")
    elif fee_drag_pct < 60:
        verdict_points.append(f"  [WARNING] Fee drag {fee_drag_pct:.0f}% of gross — fees eating profits")
    else:
        verdict_points.append(f"  [CRITICAL] Fee drag {fee_drag_pct:.0f}% of gross — fees destroy the edge")

    # Check 3: Win rate
    if avg_win_rate > 65:
        verdict_points.append(f"  [STRONG] Avg win rate {avg_win_rate:.0f}% — consistent edge")
    elif avg_win_rate > 50:
        verdict_points.append(f"  [OK] Avg win rate {avg_win_rate:.0f}% — marginal edge")
    else:
        verdict_points.append(f"  [WEAK] Avg win rate {avg_win_rate:.0f}% — no consistent edge")

    # Check 4: Sharpe
    if avg_sharpe > 2.0:
        verdict_points.append(f"  [EXCELLENT] Avg Sharpe {avg_sharpe:.2f} — strong risk-adjusted returns")
    elif avg_sharpe > 1.0:
        verdict_points.append(f"  [GOOD] Avg Sharpe {avg_sharpe:.2f} — acceptable risk-adjusted returns")
    elif avg_sharpe > 0:
        verdict_points.append(f"  [MARGINAL] Avg Sharpe {avg_sharpe:.2f} — risk-adjusted returns below threshold")
    else:
        verdict_points.append(f"  [BAD] Avg Sharpe {avg_sharpe:.2f} — negative risk-adjusted returns")

    # Check 5: What % of symbols profitable
    if pct_profitable > 75:
        verdict_points.append(f"  [STRONG] {pct_profitable:.0f}% of symbols profitable — broad edge")
    elif pct_profitable > 50:
        verdict_points.append(f"  [OK] {pct_profitable:.0f}% of symbols profitable — selective opportunity")
    else:
        verdict_points.append(f"  [WEAK] {pct_profitable:.0f}% of symbols profitable — concentrated/fragile")

    for p in verdict_points:
        print(p)

    # Final recommendation
    print()
    strong_count = sum(1 for p in verdict_points if "[STRONG]" in p or "[EXCELLENT]" in p or "[GOOD]" in p)
    weak_count = sum(1 for p in verdict_points if "[WEAK]" in p or "[CRITICAL]" in p or "[BAD]" in p or "[NEGATIVE]" in p)

    if strong_count >= 3:
        print("  >>> RECOMMENDATION: YES — Build this strategy.")
        print("      Funding rate arb shows a real, measurable edge across multiple symbols.")
        print("      Priority: implement with maker-only limit orders to cut fee drag further.")
    elif strong_count >= 2 and weak_count <= 1:
        print("  >>> RECOMMENDATION: CONDITIONAL YES — Build with improvements.")
        print("      Edge exists but is thin. Key improvements needed:")
        print("      1. Switch to maker-only orders (save ~0.15% round-trip)")
        print("      2. Cherry-pick top 5 symbols only (discard low-APY symbols)")
        print("      3. Add dynamic entry threshold based on recent funding volatility")
    elif weak_count >= 3:
        print("  >>> RECOMMENDATION: NO — Do not build.")
        print("      Fee drag destroys the funding edge. At current Binance fee tiers,")
        print("      this strategy is not viable unless you qualify for VIP fee discounts")
        print("      (VIP1+ = 0.04% maker / 0.04% taker, saving ~50% on fees).")
    else:
        print("  >>> RECOMMENDATION: MAYBE — Paper trade first.")
        print("      Results are mixed. Run a 30-day paper trade on the top 5 symbols.")
        print("      If paper results confirm >10% APY with Sharpe >1.0, go live with small size.")

    print()
    print("  Key assumptions / limitations:")
    print("  - Perfect fills assumed (no slippage on entry/exit)")
    print("  - No basis risk modeled in PnL (only tracked as hypothetical drawdown)")
    print("  - Spot leg costs not included (would need spot margin or full collateral)")
    print("  - Capital efficiency: need ~2x notional (spot + futures margin)")
    print("  - Real APY should be halved if using 2x capital for both legs")
    print("  - Funding rates are backward-looking; future rates may differ")
    print()

    # Additional: what if we used maker orders?
    maker_fee = 0.0002  # 0.02% maker on Binance futures
    spot_taker = 0.00075  # still taker on spot
    improved_rt = 2 * maker_fee + 2 * spot_taker  # 0.04% + 0.15% = 0.19%
    fee_savings = (ROUND_TRIP_FEE - improved_rt) * NOTIONAL_PER_TRADE
    total_trades_all = summary_df["total_trades"].sum()
    potential_savings = fee_savings * total_trades_all
    print(f"  OPTIMIZATION NOTE: Switching futures to maker-only orders")
    print(f"    Current round-trip: {ROUND_TRIP_FEE*100:.2f}%")
    print(f"    Optimized round-trip: {improved_rt*100:.2f}%")
    print(f"    Savings per trade: ${fee_savings:.2f}")
    print(f"    Total savings ({int(total_trades_all)} trades): ${potential_savings:,.2f}")
    print(f"    Revised portfolio PnL: ${total_pnl_all + potential_savings:,.2f}")
    if total_capital > 0:
        revised_return = (total_pnl_all + potential_savings) / total_capital
        revised_ann = (1 + revised_return) ** (365 / data_span) - 1
        print(f"    Revised portfolio ann. APY: {revised_ann*100:.2f}%")
    print()


if __name__ == "__main__":
    main()
