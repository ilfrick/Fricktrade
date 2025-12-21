from pathlib import Path
import logging
import time
import pandas as pd
import yfinance as yf


def _download_with_retries(
    symbol: str,
    period: str,
    interval: str,
    proxy: str | None,
    retries: int = 3,
):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            data = yf.download(
                tickers=symbol,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
                proxy=proxy or None,
            )
            if data is not None and not data.empty:
                return data
        except Exception as exc:
            last_err = exc
            logging.warning("yfinance download failed for %s (attempt %d/%d): %s", symbol, attempt, retries, exc)
        time.sleep(attempt * 2)
    for attempt in range(1, retries + 1):
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                period=period,
                interval=interval,
                auto_adjust=True,
                actions=False,
                proxy=proxy or None,
            )
            if data is not None and not data.empty:
                return data
        except Exception as exc:
            last_err = exc
            logging.warning("yfinance history failed for %s (attempt %d/%d): %s", symbol, attempt, retries, exc)
        time.sleep(attempt * 2)
    if last_err:
        raise last_err
    return None


def _clamp_lookback(interval: str, lookback_days: int) -> int:
    # yfinance limits for intraday intervals
    if interval == "1m":
        return min(lookback_days, 7)
    if interval in {"2m", "5m", "15m", "30m"}:
        return min(lookback_days, 60)
    if interval in {"60m", "90m", "1h"}:
        return min(lookback_days, 730)
    return lookback_days


def download_yfinance(
    symbols: list[str],
    interval: str,
    lookback_days: int,
    out_dir: str,
    proxy: str = "",
    rate_limit_seconds: int = 2,
    start: str = "",
    end: str = "",
) -> list[Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    files = []
    clamped_days = _clamp_lookback(interval, lookback_days)
    period = f"{clamped_days}d"
    for symbol in symbols:
        if start:
            data = yf.download(
                tickers=symbol,
                start=start,
                end=end or None,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
                proxy=proxy or None,
            )
        else:
            data = _download_with_retries(symbol, period, interval, proxy)
        if data is None or data.empty:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            data = data.copy()
            data.columns = data.columns.get_level_values(-1)
        ordered = ["Open", "High", "Low", "Close", "Volume"]
        if all(col in data.columns for col in ordered):
            data = data[ordered]
        file_path = out_path / f"{symbol.replace('.', '_')}_{interval}.csv"
        data.to_csv(file_path, index_label="Datetime")
        files.append(file_path)
        time.sleep(rate_limit_seconds)
    return files
