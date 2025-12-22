from __future__ import annotations

from app.agents.orchestrator import MLStrategyOrchestrator
from app.utils.config import load_config
from app.brokers.alpaca import AlpacaBroker
from app.brokers.ibkr import IBKRBroker
from app.agents.trader import TradingAgent


def run_pretrain(config_path: str) -> None:
    cfg = load_config(config_path)
    orchestrator_cfg = cfg.get("orchestrator", {})
    if not orchestrator_cfg.get("ml", {}).get("enabled", False):
        raise SystemExit("ML orchestrator is disabled in config.")

    broker = _build_broker(cfg)
    agent = TradingAgent(broker, cfg)
    orchestrator = MLStrategyOrchestrator(orchestrator_cfg)
    orchestrator.run_pretrain(
        agent._strategy_names,  # uses existing strategy list
        agent._build_strategy,
        agent._strategy_params,
        cfg.get("data", {}),
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
