# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import argparse
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import torch # Added torch import for GPU error handling
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from ib_insync import IB, Stock, util

from app.agents.trader import TradingAgent
from app.brokers.alpaca import AlpacaBroker
from app.brokers.ibkr import IBKRBroker
from app.brokers.router import BrokerRouter
from app.brokers.config_utils import get_alpaca_account_cfg, iter_alpaca_accounts, iter_ibkr_accounts
from app.backtest.engine import run_backtest
from app.backtest.agent_engine import run_agent_backtest
from app.data.downloader import download_yfinance
from app.data.ingestion import ingest_from_config
from app.data.yfinance_utils import fetch_yfinance_bars
from app.data.market_cache import build_market_cache, build_market_cache_config
from app.learning.train_rl import train_from_config
from app.learning.pretrain_orchestrator import run_pretrain
from app.learning.evaluate import evaluate_from_config
from app.monitoring.metrics import start_metrics_server
from app.utils.config import load_config
from app.utils.logging import setup_logging
from app.utils.signal_features import compute_signal_metrics_from_window
from app.utils.gpu_state import is_gpu_disabled, disable_gpu_until_restart # Import GPU state utilities


def _empty_market_state() -> dict:
    return {
        "prices": [],
        "volumes": [],
        "qty": 1,
        "exposure_pct": 1.0,
        "short_exposure_pct": 0.0,
        "leverage": 1.0,
        "last_price": None,
        "opens": [],
        "highs": [],
        "lows": [],
        "session_volume": 0.0,
        "relative_volume": 0.0,
        "session_gain_pct": 0.0,
        "spread_pct": None,
        "signal_30m_return_pct": 0.0,
        "signal_60m_return_pct": 0.0,
        "signal_early_volume_pct": 0.0,
        "signal_runup_pct": 0.0,
        "signal_drawdown_pct": 0.0,
        "signal_abs_move": 0.0,
        "signal_runup_abs": 0.0,
        "signal_drawdown_abs": 0.0,
        "last_bar_ts": None,
    }


def _market_state_from_yf(
    symbol: str,
    lookback: int,
    interval: str,
    session_gain_mode: str,
    cache=None,
    cache_only: bool = False,
):
    data = None
    if cache is not None:
        cached = cache.get_bars([symbol], interval, max_age_seconds=_interval_seconds(interval), lowercase=False)
        data = cached.get(symbol)
        if cache_only and (data is None or data.empty):
            return _empty_market_state()
    if data is None or data.empty:
        bars = fetch_yfinance_bars([symbol], lookback, interval, batch_size=1, lowercase=False)
        data = bars.get(symbol)
    if data is None or data.empty:
        return _empty_market_state()
    return _market_state_from_df(data, lookback, interval, session_gain_mode)


def _alpaca_timeframe(interval: str) -> TimeFrame:
    if interval.endswith("m"):
        return TimeFrame(int(interval[:-1]), TimeFrameUnit.Minute)
    if interval.endswith("h"):
        return TimeFrame(int(interval[:-1]), TimeFrameUnit.Hour)
    if interval.endswith("d"):
        return TimeFrame(int(interval[:-1]), TimeFrameUnit.Day)
    return TimeFrame(1, TimeFrameUnit.Day)


def _interval_seconds(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1]) * 60
    if interval.endswith("h"):
        return int(interval[:-1]) * 3600
    if interval.endswith("d"):
        return int(interval[:-1]) * 86400
    return 60


def _bars_for_lookback(lookback_days: int, interval: str) -> int:
    if interval.endswith("m"):
        minutes = max(int(interval[:-1]), 1)
        per_day = max(int(390 / minutes), 1)
    elif interval.endswith("h"):
        hours = max(int(interval[:-1]), 1)
        per_day = max(int(6.5 / hours), 1)
    elif interval.endswith("d"):
        per_day = 1
    else:
        per_day = 1
    return max(per_day * max(lookback_days, 1), 1)


def _market_state_from_df(data: pd.DataFrame, lookback_days: int, interval: str, session_gain_mode: str) -> dict:
    if data is None or data.empty:
        return _empty_market_state()
    if isinstance(data.index, pd.MultiIndex):
        data = data.copy()
        data.index = data.index.get_level_values(-1)
    if "close" in data.columns:
        data = data.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
    if not data.empty:
        data = data.copy()
        for col in ("Open", "High", "Low", "Close"):
            if col in data.columns:
                data[col] = data[col].ffill().bfill()
        if "Volume" in data.columns:
            data["Volume"] = data["Volume"].fillna(0.0)
    if "Close" not in data.columns:
        return _empty_market_state()
    close = data["Close"]
    volume = data["Volume"] if "Volume" in data else None
    open_ = data["Open"] if "Open" in data else None
    high = data["High"] if "High" in data else None
    low = data["Low"] if "Low" in data else None
    lookback_bars = _bars_for_lookback(lookback_days, interval)
    prices = close.iloc[-lookback_bars:].to_numpy(dtype=float).tolist()
    volumes = volume.iloc[-lookback_bars:].to_numpy(dtype=float).tolist() if volume is not None else []
    opens = open_.iloc[-lookback_bars:].to_numpy(dtype=float).tolist() if open_ is not None else []
    highs = high.iloc[-lookback_bars:].to_numpy(dtype=float).tolist() if high is not None else []
    lows = low.iloc[-lookback_bars:].to_numpy(dtype=float).tolist() if low is not None else []
    last_price = prices[-1] if prices else None
    avg_volume = float(sum(volumes) / len(volumes)) if volumes else 0.0
    session_volume = float(sum(volumes)) if volumes else 0.0
    rel_volume = float(volumes[-1] / avg_volume) if avg_volume else 0.0
    session_gain_pct = _session_gain_pct(data, prices, session_gain_mode)
    last_bar_ts = data.index[-1].to_pydatetime()
    state = {
        "prices": prices,
        "volumes": volumes,
        "qty": 1,
        "exposure_pct": 1.0,
        "short_exposure_pct": 0.0,
        "leverage": 1.0,
        "last_price": last_price,
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "session_volume": session_volume,
        "relative_volume": rel_volume,
        "session_gain_pct": session_gain_pct,
        "spread_pct": None,
        "last_bar_ts": last_bar_ts,
    }
    state.update(
        compute_signal_metrics_from_window(
            prices=prices,
            volumes=volumes,
            highs=highs or None,
            lows=lows or None,
            interval=interval,
        )
    )
    return state


def _fetch_bars(client: StockHistoricalDataClient, req: StockBarsRequest, timeout: int, retries: int):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    for attempt in range(retries + 1):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(client.get_stock_bars, req)
            try:
                return future.result(timeout=timeout).df
            except TimeoutError:
                if attempt >= retries:
                    return None
            except Exception as exc:
                if attempt >= retries:
                    logging.warning("Alpaca bars fetch failed: %s", exc)
                    return None
    return None


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


class AlpacaMarketDataProvider:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        lookback_days: int,
        interval: str,
        session_gain_mode: str,
        feed: str = "iex",
        chunk_size: int = 200,
        timeout_seconds: int = 10,
        retries: int = 2,
    ):
        self._client = StockHistoricalDataClient(api_key, api_secret)
        self._lookback = lookback_days
        self._interval = interval
        self._session_gain_mode = session_gain_mode
        self._feed = feed
        self._chunk_size = chunk_size
        self._timeout_seconds = timeout_seconds
        self._retries = retries
        self._cache: dict[str, dict] = {}
        self._cache_at: datetime | None = None

    def prepare(self, symbols: list[str]) -> None:
        if not symbols:
            return
        now = datetime.now(timezone.utc)
        refresh_seconds = _interval_seconds(self._interval)
        if self._cache_at and (now - self._cache_at).total_seconds() < refresh_seconds:
            return
        start = now - timedelta(days=self._lookback)
        timeframe = _alpaca_timeframe(self._interval)
        cache: dict[str, dict] = {}
        for chunk in _chunked(symbols, self._chunk_size):
            req = StockBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=timeframe,
                start=start,
                end=now,
                feed=self._feed,
                adjustment="raw",
            )
            data = _fetch_bars(self._client, req, self._timeout_seconds, self._retries)
            if data is None or data.empty:
                continue
            if isinstance(data.index, pd.MultiIndex):
                for symbol in chunk:
                    try:
                        df = data.xs(symbol, level=0)
                    except KeyError:
                        continue
                    cache[symbol] = _market_state_from_df(
                        df,
                        self._lookback,
                        self._interval,
                        self._session_gain_mode,
                    )
            else:
                symbol = chunk[0]
                cache[symbol] = _market_state_from_df(
                    data,
                    self._lookback,
                    self._interval,
                    self._session_gain_mode,
                )
        self._cache = cache
        self._cache_at = now

    def __call__(self, symbol: str) -> dict:
        if symbol in self._cache:
            return self._cache.get(symbol, _empty_market_state())
        state = self._fetch_symbol(symbol)
        if state:
            self._cache[symbol] = state
            return state
        return _empty_market_state()

    def _fetch_symbol(self, symbol: str) -> dict | None:
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=self._lookback)
        timeframe = _alpaca_timeframe(self._interval)
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=timeframe,
            start=start,
            end=now,
            feed=self._feed,
            adjustment="raw",
        )
        data = _fetch_bars(self._client, req, self._timeout_seconds, self._retries)
        if data is None or data.empty:
            return None
        if isinstance(data.index, pd.MultiIndex):
            try:
                df = data.xs(symbol, level=0)
            except KeyError:
                return None
        else:
            df = data
        return _market_state_from_df(df, self._lookback, self._interval, self._session_gain_mode)


def _ibkr_bar_size(interval: str) -> str:
    if interval.endswith("m"):
        minutes = int(interval[:-1])
        return f"{minutes} min"
    if interval.endswith("h"):
        hours = int(interval[:-1])
        return f"{hours} hour"
    if interval.endswith("d"):
        return "1 day"
    return "1 min"


class IBKRMarketDataProvider:
    def __init__(
        self,
        ib: IB,
        lookback_days: int,
        interval: str,
        session_gain_mode: str,
        exchange: str = "SMART",
        currency: str = "USD",
    ):
        self._ib = ib
        self._lookback = lookback_days
        self._interval = interval
        self._session_gain_mode = session_gain_mode
        self._exchange = exchange
        self._currency = currency
        self._cache: dict[str, dict] = {}
        self._cache_at: datetime | None = None

    def prepare(self, symbols: list[str]) -> None:
        if not symbols:
            return
        now = datetime.now(timezone.utc)
        refresh_seconds = _interval_seconds(self._interval)
        if self._cache_at and (now - self._cache_at).total_seconds() < refresh_seconds:
            return
        cache: dict[str, dict] = {}
        duration = f"{self._lookback} D"
        bar_size = _ibkr_bar_size(self._interval)
        for symbol in symbols:
            contract = Stock(symbol, self._exchange, self._currency)
            try:
                bars = self._ib.reqHistoricalData(
                    contract,
                    endDateTime="",
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1,
                )
            except Exception as exc:
                logging.warning("IBKR bars fetch failed for %s: %s", symbol, exc)
                continue
            if not bars:
                continue
            df = util.df(bars)
            cache[symbol] = _market_state_from_df(
                df,
                self._lookback,
                self._interval,
                self._session_gain_mode,
            )
        self._cache = cache
        self._cache_at = now

    def __call__(self, symbol: str) -> dict:
        if symbol in self._cache:
            return self._cache.get(symbol, _empty_market_state())
        state = self._fetch_symbol(symbol)
        if state:
            self._cache[symbol] = state
            return state
        return _empty_market_state()

    def _fetch_symbol(self, symbol: str) -> dict | None:
        now = datetime.now(timezone.utc)
        duration = f"{self._lookback} D"
        bar_size = _ibkr_bar_size(self._interval)
        contract = Stock(symbol, self._exchange, self._currency)
        try:
            bars = self._ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
        except Exception as exc:
            logging.warning("IBKR bars fetch failed for %s: %s", symbol, exc)
            return None
        if not bars:
            return None
        df = util.df(bars)
        return _market_state_from_df(df, self._lookback, self._interval, self._session_gain_mode)


class YFinanceMarketDataProvider:
    def __init__(
        self,
        lookback_days: int,
        interval: str,
        session_gain_mode: str,
        chunk_size: int = 100,
        cache=None,
        cache_only: bool = False,
    ):
        self._lookback = lookback_days
        self._interval = interval
        self._session_gain_mode = session_gain_mode
        self._chunk_size = chunk_size
        self._market_cache = cache
        self._cache_only = cache_only
        self._cache: dict[str, dict] = {}
        self._cache_at: datetime | None = None

    def prepare(self, symbols: list[str]) -> None:
        if not symbols:
            return
        now = datetime.now(timezone.utc)
        refresh_seconds = _interval_seconds(self._interval)
        if self._cache_at and (now - self._cache_at).total_seconds() < refresh_seconds:
            return
        cache: dict[str, dict] = {}
        for chunk in _chunked(symbols, self._chunk_size):
            cache.update(self._fetch_chunk(chunk))
        self._cache = cache
        self._cache_at = now

    def __call__(self, symbol: str) -> dict:
        if symbol in self._cache:
            return self._cache.get(symbol, _empty_market_state())
        fetched = self._fetch_chunk([symbol])
        if fetched:
            self._cache.update(fetched)
            return self._cache.get(symbol, _empty_market_state())
        return _empty_market_state()

    def _fetch_chunk(self, symbols: list[str]) -> dict[str, dict]:
        cache: dict[str, dict] = {}
        missing = symbols
        if self._market_cache is not None:
            cached = self._market_cache.get_bars(
                symbols,
                self._interval,
                max_age_seconds=_interval_seconds(self._interval),
                lowercase=False,
            )
            for symbol, frame in cached.items():
                cache[symbol] = _market_state_from_df(
                    frame,
                    self._lookback,
                    self._interval,
                    self._session_gain_mode,
                )
            missing = [symbol for symbol in symbols if symbol not in cached]
            if self._cache_only:
                return cache
        if missing:
            bars = fetch_yfinance_bars(
                missing,
                self._lookback,
                self._interval,
                batch_size=len(missing) if missing else 1,
                lowercase=False,
            )
            for symbol, frame in bars.items():
                cache[symbol] = _market_state_from_df(
                    frame,
                    self._lookback,
                    self._interval,
                    self._session_gain_mode,
                )
            if self._market_cache is not None and bars:
                self._market_cache.set_bars(bars, self._interval, ttl_seconds=_interval_seconds(self._interval))
        return cache

    def _state_from_frame(self, frame: pd.DataFrame) -> dict | None:
        if frame is None or frame.empty:
            return None
        if "Close" not in frame.columns and "close" not in frame.columns:
            return None
        if "close" in frame.columns:
            frame = frame.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            )
        frame = frame.copy()
        for col in ("Open", "High", "Low", "Close"):
            if col in frame.columns:
                frame[col] = frame[col].ffill().bfill()
        if "Volume" in frame.columns:
            frame["Volume"] = frame["Volume"].fillna(0.0)
        if any(
            col in frame.columns and frame[col].isna().any()
            for col in ("Open", "High", "Low", "Close")
        ):
            return None
        return _market_state_from_df(frame, self._lookback, self._interval, self._session_gain_mode)


class MultiBrokerMarketDataProvider:
    def __init__(self, providers: dict[str, object], routing: dict | None = None):
        self._providers = providers
        self._routing = routing or {}

    def _resolve_broker(self, symbol: str) -> str | None:
        symbols_map = self._routing.get("symbols", {}) if isinstance(self._routing, dict) else {}
        if symbol in symbols_map:
            return str(symbols_map[symbol])
        default = None
        if isinstance(self._routing, dict):
            default = self._routing.get("default")
        if default:
            return str(default)
        return next(iter(self._providers.keys()), None)

    def prepare(self, symbols: list[str]) -> None:
        buckets: dict[str, list[str]] = {}
        for symbol in symbols:
            broker_name = self._resolve_broker(symbol)
            if broker_name and broker_name in self._providers:
                buckets.setdefault(broker_name, []).append(symbol)
        for name, bucket in buckets.items():
            provider = self._providers.get(name)
            if provider and hasattr(provider, "prepare"):
                provider.prepare(bucket)

    def __call__(self, symbol: str) -> dict:
        broker_name = self._resolve_broker(symbol)
        provider = self._providers.get(broker_name) if broker_name else None
        if provider:
            return provider(symbol)
        return _empty_market_state()


def _session_gain_pct(data, prices: list[float], mode: str) -> float:
    if data is None or data.empty or not prices:
        return 0.0
    try:
        if mode == "session":
            first_price = prices[0]
            return (prices[-1] - first_price) / first_price * 100.0 if first_price else 0.0
        if len(data.index) < 2:
            return 0.0
        prev_close = data["Close"].iloc[-2]
        if prev_close:
            return (prices[-1] - prev_close) / prev_close * 100.0
        return 0.0
    except Exception:
        return 0.0


def _build_broker(cfg: dict):
    brokers: dict[str, object] = {}
    paper = os.getenv("TRADING_MODE", "paper").lower() == "paper"
    market_cfg = cfg.get("market", {}) if isinstance(cfg, dict) else {}
    default_currency = str(market_cfg.get("default_currency") or "USD").upper()
    symbol_currencies = market_cfg.get("symbol_currencies", {}) if isinstance(market_cfg, dict) else {}
    ibkr_base_cfg = cfg.get("brokers", {}).get("ibkr", {}) if isinstance(cfg, dict) else {}
    ibkr_default_currency = str(ibkr_base_cfg.get("currency") or default_currency).upper()
    for account in iter_alpaca_accounts(cfg):
        try:
            broker = AlpacaBroker(
                account.get("api_key", ""),
                account.get("api_secret", ""),
                account.get("base_url", ""),
                paper=paper,
                name=account["name"],
            )
            if not broker.is_connected():
                logging.warning("Alpaca account %s unavailable; skipping.", account["name"])
                continue
            brokers[account["name"]] = broker
        except Exception as exc:
            logging.warning("Alpaca account %s failed to initialize: %s", account.get("name", "unknown"), exc)
            continue
    for account in iter_ibkr_accounts(cfg):
        try:
            currency = str(account.get("currency") or ibkr_default_currency or default_currency).upper()
            if not currency:
                currency = default_currency
            broker = IBKRBroker(
                account.get("host", "127.0.0.1"),
                int(account.get("port", 7497)),
                int(account.get("client_id", 1)),
                name=account["name"],
                account_id=account.get("account_id", ""),
                currency=currency,
                symbol_currencies=symbol_currencies,
            )
            if not broker.is_connected():
                logging.warning("IBKR account %s unavailable; skipping.", account["name"])
                continue
            brokers[account["name"]] = broker
        except Exception as exc:
            logging.warning("IBKR account %s failed to initialize: %s", account.get("name", "unknown"), exc)
            continue
    exec_cfg = cfg.get("execution", {}).get("brokers", {})
    if len(brokers) > 1:
        if exec_cfg.get("enabled", False):
            return BrokerRouter(brokers, exec_cfg.get("routing", {}))
        logging.warning("Multiple brokers configured but routing disabled; defaulting to BrokerRouter.")
        return BrokerRouter(brokers, exec_cfg.get("routing", {}))
    if len(brokers) == 1:
        return next(iter(brokers.values()))
    return None


def _primary_alpaca_cfg(cfg: dict) -> dict:
    return get_alpaca_account_cfg(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/app/config/config.yaml")
    sub = parser.add_subparsers(dest="cmd")

    trade_parser = sub.add_parser("trade")
    backtest_parser = sub.add_parser("backtest")
    download_parser = sub.add_parser("download")
    ingest_parser = sub.add_parser("ingest")
    api_parser = sub.add_parser("api")
    train_parser = sub.add_parser("train")
    online_parser = sub.add_parser("online-train")
    eval_parser = sub.add_parser("evaluate")
    pretrain_orch_parser = sub.add_parser("pretrain-orchestrator")

    for parser_item in (
        trade_parser,
        backtest_parser,
        download_parser,
        ingest_parser,
        api_parser,
        train_parser,
        online_parser,
        eval_parser,
        pretrain_orch_parser,
    ):
        parser_item.add_argument("--config", default="/app/config/config.yaml")

    download_parser.add_argument("--symbols", nargs="*", default=[])

    args = parser.parse_args()
    cfg = load_config(args.config)
    log_cfg = cfg.get("logging", {})
    setup_logging(
        cfg["app"]["log_level"],
        file_path=log_cfg.get("file_path"),
        max_bytes=int(log_cfg.get("max_bytes", 5_000_000)),
        backup_count=int(log_cfg.get("backup_count", 5)),
    )

    start_metrics_server(cfg["monitoring"]["prometheus_port"])

    if args.cmd == "download":
        out_dir = cfg["backtest"]["data_dir"]
        symbols = args.symbols or cfg["data"]["symbols"]
        download_yfinance(
            symbols,
            cfg["data"]["interval"],
            cfg["data"]["lookback_days"],
            out_dir,
            proxy=cfg["data"].get("proxy", ""),
            rate_limit_seconds=cfg["data"].get("rate_limit_seconds", 2),
            start=cfg["data"].get("start", ""),
            end=cfg["data"].get("end", ""),
        )
        logging.info("Download complete")
        return

    if args.cmd == "ingest":
        files = ingest_from_config(cfg)
        logging.info("Ingested %d files", len(files))
        return

    if args.cmd == "backtest":
        mode = cfg.get("backtest", {}).get("mode", "agent")
        if mode == "agent":
            result = run_agent_backtest(cfg)
            logging.info("Agent backtest result: %s", result.__dict__)
        else:
            initial_use_gpu = cfg["backtest"].get("use_gpu", True) and not is_gpu_disabled()
            use_gpu_for_run = initial_use_gpu

            for attempt in range(2): # Try once with initial setting, once with CPU if GPU fails
                try:
                    result = run_backtest(
                        cfg["backtest"]["data_dir"],
                        cfg["backtest"]["start"],
                        cfg["backtest"]["end"],
                        cfg["backtest"]["initial_cash"],
                        cfg["backtest"]["commission_pct"],
                        interval=cfg["data"].get("interval"),
                        use_gpu=use_gpu_for_run,
                    )
                    logging.info("Backtest result: %s", result)
                    break # Success, exit retry loop
                except (torch.cuda.OutOfMemoryError, tf.errors.ResourceExhaustedError) as exc:
                    if use_gpu_for_run: # If we were trying with GPU
                        logging.warning("CUDA out of memory during backtest: %s. Falling back to CPU for current and future runs.", exc)
                        disable_gpu_until_restart()
                        use_gpu_for_run = False # Force CPU for next attempt
                        continue # Retry with CPU
                    else: # If we already tried with CPU and still failed
                        logging.error("Backtest failed on CPU after GPU error: %s", exc)
                        raise # Re-raise the error
                except Exception as exc:
                    logging.error("Unknown error during backtest: %s", exc)
                    raise
        return

    if args.cmd == "api":
        import uvicorn

        uvicorn.run("app.api.server:app", host="0.0.0.0", port=8000, reload=False)
        return

    if args.cmd == "train":
        training_cfg = cfg.get("learning", {}).get("training", {})
        resume = bool(training_cfg.get("resume", True))
        model_path = train_from_config(cfg, resume=resume)
        logging.info("Training complete. Model saved to %s", model_path)
        return

    if args.cmd == "online-train":
        from app.learning.online_update import run_online_updates

        run_online_updates(cfg)
        return

    if args.cmd == "evaluate":
        report = evaluate_from_config(cfg)
        logging.info("Evaluation complete. Avg return %.2f%%", report["average"]["return_pct"])
        return

    if args.cmd == "pretrain-orchestrator":
        run_pretrain(args.config)
        logging.info("Orchestrator pretraining complete.")
        return

    if args.cmd == "trade":
        broker = _build_broker(cfg)
        agent = TradingAgent(broker, cfg)
        symbols = cfg["data"]["symbols"]
        cache_cfg = build_market_cache_config(cfg.get("market_cache", {}))
        market_cache = build_market_cache(cfg.get("market_cache", {}))
        provider = str(cfg.get("data", {}).get("provider", "yfinance")).lower()
        if provider == "alpaca":
            alpaca_cfg = _primary_alpaca_cfg(cfg)
            api_key = alpaca_cfg.get("api_key", "")
            api_secret = alpaca_cfg.get("api_secret", "")
            dyn_cfg = cfg.get("data", {}).get("dynamic_symbols", {})
            feed = dyn_cfg.get("feed", "iex")
            if api_key and api_secret:
                market_data_provider = AlpacaMarketDataProvider(
                    api_key,
                    api_secret,
                    cfg["data"]["lookback_days"],
                    cfg["data"]["interval"],
                    cfg["data"].get("session_gain_mode", "gap"),
                    feed=feed,
                    timeout_seconds=int(dyn_cfg.get("timeout_seconds", 10)),
                    retries=int(dyn_cfg.get("retries", 2)),
                )
            else:
                logging.warning("Alpaca provider selected but credentials missing; falling back to yfinance.")
                market_data_provider = lambda s: _market_state_from_yf(
                    s,
                    cfg["data"]["lookback_days"],
                    cfg["data"]["interval"],
                    cfg["data"].get("session_gain_mode", "gap"),
                    cache=market_cache,
                    cache_only=cache_cfg.cache_only,
                )
        elif provider == "brokers":
            providers: dict[str, object] = {}
            routing_cfg = cfg.get("execution", {}).get("brokers", {}).get("routing", {})
            alpaca_cfg = _primary_alpaca_cfg(cfg)
            api_key = alpaca_cfg.get("api_key", "")
            api_secret = alpaca_cfg.get("api_secret", "")
            dyn_cfg = cfg.get("data", {}).get("dynamic_symbols", {})
            feed = dyn_cfg.get("feed", "iex")
            if api_key and api_secret:
                providers["alpaca"] = AlpacaMarketDataProvider(
                    api_key,
                    api_secret,
                    cfg["data"]["lookback_days"],
                    cfg["data"]["interval"],
                    cfg["data"].get("session_gain_mode", "gap"),
                    feed=feed,
                    timeout_seconds=int(dyn_cfg.get("timeout_seconds", 10)),
                    retries=int(dyn_cfg.get("retries", 2)),
                )
            if cfg.get("brokers", {}).get("ibkr", {}).get("enabled", False):
                ibkr_cfg = cfg.get("brokers", {}).get("ibkr", {})
                ib = None
                if hasattr(broker, "ib"):
                    ib = broker.ib
                elif hasattr(broker, "brokers"):
                    try:
                        broker_map = broker.brokers if isinstance(broker.brokers, dict) else broker.brokers()
                    except Exception:
                        broker_map = {}
                    ibkr_broker = broker_map.get("ibkr") if isinstance(broker_map, dict) else None
                    if ibkr_broker is not None and hasattr(ibkr_broker, "ib"):
                        ib = ibkr_broker.ib
                if ib is None:
                    ib = IB()
                    client_id = int(ibkr_cfg.get("client_id", 1)) + 1
                    try:
                        ib.connect(
                            ibkr_cfg.get("host", "127.0.0.1"),
                            ibkr_cfg.get("port", 7497),
                            clientId=client_id,
                        )
                    except Exception as exc:
                        logging.warning("IBKR market data connection failed: %s", exc)
                        ib = None
                if ib is not None:
                    providers["ibkr"] = IBKRMarketDataProvider(
                        ib,
                        cfg["data"]["lookback_days"],
                        cfg["data"]["interval"],
                        cfg["data"].get("session_gain_mode", "gap"),
                    )
            if providers:
                market_data_provider = MultiBrokerMarketDataProvider(providers, routing_cfg)
            else:
                logging.warning("Broker providers unavailable; falling back to yfinance.")
                market_data_provider = YFinanceMarketDataProvider(
                    cfg["data"]["lookback_days"],
                    cfg["data"]["interval"],
                    cfg["data"].get("session_gain_mode", "gap"),
                    cache=market_cache,
                    cache_only=cache_cfg.cache_only,
                )
        elif provider == "yfinance":
            market_data_provider = YFinanceMarketDataProvider(
                cfg["data"]["lookback_days"],
                cfg["data"]["interval"],
                cfg["data"].get("session_gain_mode", "gap"),
                cache=market_cache,
                cache_only=cache_cfg.cache_only,
            )
        else:
            market_data_provider = YFinanceMarketDataProvider(
                cfg["data"]["lookback_days"],
                cfg["data"]["interval"],
                cfg["data"].get("session_gain_mode", "gap"),
                cache=market_cache,
                cache_only=cache_cfg.cache_only,
            )
        agent.loop(symbols, market_data_provider, 60)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
