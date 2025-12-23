from __future__ import annotations

from pathlib import Path
import io
import logging
import time

import pandas as pd
import requests

from app.data.downloader import download_yfinance, download_alpaca_bars


def _save_ohlcv(df: pd.DataFrame, out_dir: str, symbol: str, interval: str) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"{symbol.replace('.', '_')}_{interval}.csv"
    df.to_csv(file_path, index_label="Datetime")
    return file_path


def _download_stooq(symbol: str, interval: str, out_dir: str) -> Path | None:
    interval_map = {"1d": "d", "1w": "w", "1m": "m"}
    stooq_interval = interval_map.get(interval, "d")
    url = f"https://stooq.com/q/d/l/?s={symbol}&i={stooq_interval}"
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        logging.warning("Stooq download failed for %s: status %s", symbol, resp.status_code)
        return None
    df = pd.read_csv(io.StringIO(resp.text))
    if df.empty:
        return None
    df = df.rename(
        columns={
            "Date": "Datetime",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        }
    )
    return _save_ohlcv(df, out_dir, symbol, interval)


def _download_alphavantage(symbol: str, interval: str, api_key: str, out_dir: str) -> Path | None:
    function = "TIME_SERIES_DAILY_ADJUSTED"
    params = {"function": function, "symbol": symbol, "apikey": api_key, "outputsize": "full"}
    resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
    if resp.status_code != 200:
        logging.warning("Alphavantage download failed for %s: status %s", symbol, resp.status_code)
        return None
    payload = resp.json()
    series = payload.get("Time Series (Daily)")
    if not series:
        logging.warning("Alphavantage response missing data for %s", symbol)
        return None
    rows = []
    for ts, values in series.items():
        rows.append(
            {
                "Datetime": ts,
                "Open": float(values["1. open"]),
                "High": float(values["2. high"]),
                "Low": float(values["3. low"]),
                "Close": float(values["4. close"]),
                "Volume": float(values["6. volume"]),
            }
        )
    df = pd.DataFrame(rows).sort_values("Datetime")
    return _save_ohlcv(df, out_dir, symbol, interval)


def ingest_from_config(cfg: dict) -> list[Path]:
    data_cfg = cfg.get("data", {})
    sources = data_cfg.get("sources", [])
    output_dir = data_cfg.get("output_dir", cfg["backtest"]["data_dir"])
    files: list[Path] = []

    for source in sources:
        if not source.get("enabled", True):
            continue
        provider = source.get("provider", "yfinance")
        symbols = source.get("symbols", data_cfg.get("symbols", []))
        interval = source.get("interval", data_cfg.get("interval", "1d"))
        rate_limit = int(source.get("rate_limit_seconds", data_cfg.get("rate_limit_seconds", 2)))

        if provider == "yfinance":
            files.extend(
                download_yfinance(
                    symbols,
                    interval=interval,
                    lookback_days=int(source.get("lookback_days", data_cfg.get("lookback_days", 30))),
                    out_dir=output_dir,
                    proxy=source.get("proxy", data_cfg.get("proxy", "")),
                    rate_limit_seconds=rate_limit,
                    start=source.get("start", data_cfg.get("start", "")),
                    end=source.get("end", data_cfg.get("end", "")),
                )
            )
            continue

        if provider == "alpaca":
            alpaca_cfg = cfg.get("brokers", {}).get("alpaca", {})
            api_key = source.get("api_key", alpaca_cfg.get("api_key", ""))
            api_secret = source.get("api_secret", alpaca_cfg.get("api_secret", ""))
            files.extend(
                download_alpaca_bars(
                    symbols,
                    interval=interval,
                    out_dir=output_dir,
                    api_key=api_key,
                    api_secret=api_secret,
                    start=source.get("start", data_cfg.get("start", "")),
                    end=source.get("end", data_cfg.get("end", "")),
                    rate_limit_seconds=rate_limit,
                )
            )
            continue

        if provider == "stooq":
            for symbol in symbols:
                result = _download_stooq(symbol, interval, output_dir)
                if result:
                    files.append(result)
                time.sleep(rate_limit)
            continue

        if provider == "alphavantage":
            api_key = source.get("api_key", "")
            if not api_key:
                logging.warning("Alphavantage API key missing; skipping")
                continue
            for symbol in symbols:
                result = _download_alphavantage(symbol, interval, api_key, output_dir)
                if result:
                    files.append(result)
                time.sleep(rate_limit)
            continue

        logging.warning("Unknown data provider: %s", provider)

    return files
