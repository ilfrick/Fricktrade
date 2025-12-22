import argparse
import logging
import os
from pathlib import Path

import yfinance as yf

from app.agents.trader import TradingAgent
from app.brokers.alpaca import AlpacaBroker
from app.brokers.ibkr import IBKRBroker
from app.backtest.engine import run_backtest
from app.data.downloader import download_yfinance
from app.data.ingestion import ingest_from_config
from app.learning.train_rl import train_from_config
from app.learning.evaluate import evaluate_from_config
from app.monitoring.metrics import start_metrics_server
from app.utils.config import load_config
from app.utils.logging import setup_logging


def _market_state_from_yf(symbol: str, lookback: int, interval: str, session_gain_mode: str):
    data = yf.download(tickers=symbol, period=f"{lookback}d", interval=interval, auto_adjust=True, progress=False)
    if data is None or data.empty:
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
        }
    if getattr(data.columns, "nlevels", 1) > 1:
        data = data.copy()
        if "Close" in data.columns.get_level_values(0):
            data.columns = data.columns.get_level_values(0)
        else:
            data.columns = data.columns.get_level_values(-1)
    if "Close" not in data.columns:
        return {
            "prices": [],
            "volumes": [],
            "qty": 1,
            "exposure_pct": 1.0,
            "short_exposure_pct": 0.0,
            "leverage": 1.0,
            "last_price": None,
        }
    close = data["Close"]
    volume = data["Volume"] if "Volume" in data else None
    open_ = data["Open"] if "Open" in data else None
    high = data["High"] if "High" in data else None
    low = data["Low"] if "Low" in data else None
    if isinstance(close, type(data)):
        close = close.iloc[:, 0]
    if volume is not None and isinstance(volume, type(data)):
        volume = volume.iloc[:, 0]
    prices = close.iloc[-lookback:].tolist()
    volumes = volume.iloc[-lookback:].tolist() if volume is not None else []
    opens = open_.iloc[-lookback:].tolist() if open_ is not None else []
    highs = high.iloc[-lookback:].tolist() if high is not None else []
    lows = low.iloc[-lookback:].tolist() if low is not None else []
    last_price = prices[-1] if prices else None
    avg_volume = float(sum(volumes) / len(volumes)) if volumes else 0.0
    session_volume = float(sum(volumes)) if volumes else 0.0
    rel_volume = float(volumes[-1] / avg_volume) if avg_volume else 0.0
    session_gain_pct = _session_gain_pct(data, prices, session_gain_mode)
    return {
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
    }


def _session_gain_pct(data, prices: list[float], mode: str) -> float:
    if data is None or data.empty or not prices:
        return 0.0
    try:
        if mode == "session":
            first_price = prices[0]
            return (prices[-1] - first_price) / first_price * 100.0 if first_price else 0.0
        prior_data = data.iloc[:-1]
        prev_close = None
        if not prior_data.empty:
            prev_close = prior_data["Close"].iloc[-1]
        if prev_close:
            return (prices[-1] - prev_close) / prev_close * 100.0
        return 0.0
    except Exception:
        return 0.0


def _build_broker(cfg: dict):
    if cfg["brokers"]["ibkr"]["enabled"]:
        return IBKRBroker(
            cfg["brokers"]["ibkr"]["host"],
            cfg["brokers"]["ibkr"]["port"],
            cfg["brokers"]["ibkr"]["client_id"],
        )
    paper = os.getenv("TRADING_MODE", "paper").lower() == "paper"
    return AlpacaBroker(
        cfg["brokers"]["alpaca"]["api_key"],
        cfg["brokers"]["alpaca"]["api_secret"],
        cfg["brokers"]["alpaca"]["base_url"],
        paper=paper,
    )


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

    for parser_item in (
        trade_parser,
        backtest_parser,
        download_parser,
        ingest_parser,
        api_parser,
        train_parser,
        online_parser,
        eval_parser,
    ):
        parser_item.add_argument("--config", default="/app/config/config.yaml")

    download_parser.add_argument("--symbols", nargs="*", default=[])

    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg["app"]["log_level"])

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
        result = run_backtest(
            cfg["backtest"]["data_dir"],
            cfg["backtest"]["start"],
            cfg["backtest"]["end"],
            cfg["backtest"]["initial_cash"],
            cfg["backtest"]["commission_pct"],
            interval=cfg["data"].get("interval"),
            use_gpu=cfg["backtest"].get("use_gpu", True),
        )
        logging.info("Backtest result: %s", result)
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

    if args.cmd == "trade":
        broker = _build_broker(cfg)
        agent = TradingAgent(broker, cfg)
        symbols = cfg["data"]["symbols"]
        agent.loop(
            symbols,
            lambda s: _market_state_from_yf(
                s,
                cfg["data"]["lookback_days"],
                cfg["data"]["interval"],
                cfg["data"].get("session_gain_mode", "gap"),
            ),
            60,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
