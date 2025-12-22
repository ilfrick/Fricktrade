from __future__ import annotations

import random

from app.agents.orchestrator import MLStrategyOrchestrator
from app.utils.config import load_config
from app.brokers.alpaca import AlpacaBroker
from app.brokers.ibkr import IBKRBroker
from app.agents.trader import TradingAgent
from app.data.scanner import load_universe


def run_pretrain(config_path: str) -> None:
    cfg = load_config(config_path)
    orchestrator_cfg = cfg.get("orchestrator", {})
    if not orchestrator_cfg.get("ml", {}).get("enabled", False):
        raise SystemExit("ML orchestrator is disabled in config.")

    broker = _build_broker(cfg)
    agent = TradingAgent(broker, cfg)
    orchestrator = MLStrategyOrchestrator(orchestrator_cfg)
    data_cfg = _resolve_data_cfg(cfg, orchestrator_cfg)
    orchestrator.run_pretrain(
        agent._strategy_names,  # uses existing strategy list
        agent._build_strategy,
        agent._strategy_params,
        data_cfg,
    )


def _build_broker(cfg: dict):
    if cfg["brokers"]["ibkr"]["enabled"]:
        return IBKRBroker(
            cfg["brokers"]["ibkr"]["host"],
            cfg["brokers"]["ibkr"]["port"],
            cfg["brokers"]["ibkr"]["client_id"],
        )
    return AlpacaBroker(
        cfg["brokers"]["alpaca"]["api_key"],
        cfg["brokers"]["alpaca"]["api_secret"],
        cfg["brokers"]["alpaca"]["base_url"],
    )


def _resolve_data_cfg(cfg: dict, orchestrator_cfg: dict) -> dict:
    data_cfg = dict(cfg.get("data", {}))
    pretrain_cfg = orchestrator_cfg.get("ml", {}).get("pretrain", {})
    symbols_source = str(pretrain_cfg.get("symbols_source", "data"))
    max_symbols = int(pretrain_cfg.get("max_symbols", 20))
    if symbols_source != "alpaca_active_random":
        return data_cfg
    alpaca_cfg = cfg.get("brokers", {}).get("alpaca", {})
    api_key = alpaca_cfg.get("api_key", "")
    api_secret = alpaca_cfg.get("api_secret", "")
    universe = load_universe(api_key, api_secret, "alpaca_active", max_universe=max(500, max_symbols))
    if universe:
        sample = random.sample(universe, min(len(universe), max_symbols))
        data_cfg["symbols"] = sample
    return data_cfg
