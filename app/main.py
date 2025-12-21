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
from app.monitoring.metrics import start_metrics_server
from app.utils.config import load_config
from app.utils.logging import setup_logging


def _market_state_from_yf(symbol: str, lookback: int):
    data = yf.download(tickers=symbol, period=f"{lookback}d", interval="1m", auto_adjust=True, progress=False)
    prices = data["Close"].tolist()[-lookback:]
    return {
        "prices": prices,
        "qty": 1,
        "exposure_pct": 1.0,
        "short_exposure_pct": 0.0,
        "leverage": 1.0,
    }


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

    for name in ("trade", "backtest", "download", "api"):
        sub.add_parser(name).add_argument("--config", default="/app/config/config.yaml")

    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg["app"]["log_level"])

    start_metrics_server(cfg["monitoring"]["prometheus_port"])

    if args.cmd == "download":
        out_dir = cfg["backtest"]["data_dir"]
        download_yfinance(cfg["data"]["symbols"], cfg["data"]["interval"], cfg["data"]["lookback_days"], out_dir)
        logging.info("Download complete")
        return

    if args.cmd == "backtest":
        result = run_backtest(
            cfg["backtest"]["data_dir"],
            cfg["backtest"]["start"],
            cfg["backtest"]["end"],
            cfg["backtest"]["initial_cash"],
            cfg["backtest"]["commission_pct"],
        )
        logging.info("Backtest result: %s", result)
        return

    if args.cmd == "api":
        import uvicorn

        uvicorn.run("app.api.server:app", host="0.0.0.0", port=8000, reload=False)
        return

    if args.cmd == "trade":
        broker = _build_broker(cfg)
        agent = TradingAgent(broker, cfg)
        symbol = cfg["data"]["symbols"][0]
        agent.loop(symbol, lambda s: _market_state_from_yf(s, cfg["data"]["lookback_days"]), 60)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
