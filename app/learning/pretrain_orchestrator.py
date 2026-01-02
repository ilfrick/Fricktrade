# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import random
import logging

from app.agents.orchestrator import RLStrategyOrchestrator
from app.utils.config import load_config
from app.brokers.alpaca import AlpacaBroker
from app.brokers.ibkr import IBKRBroker
from app.brokers.router import BrokerRouter
from app.brokers.config_utils import get_alpaca_account_cfg, iter_alpaca_accounts, iter_ibkr_accounts
from app.agents.trader import TradingAgent
from app.data.scanner import load_universe


def run_pretrain(config_path: str) -> None:
    cfg = load_config(config_path)
    orchestrator_cfg = cfg.get("orchestrator", {})
    if not orchestrator_cfg.get("rl", {}).get("enabled", False):
        raise SystemExit("RL orchestrator is disabled in config.")

    broker = _build_broker(cfg)
    agent = TradingAgent(broker, cfg)
    orchestrator = RLStrategyOrchestrator(cfg)
    data_cfg = _resolve_data_cfg(cfg, orchestrator_cfg)
    orchestrator.run_pretrain(
        agent._strategy_names,  # uses existing strategy list
        agent._build_strategy,
        agent._strategy_params,
        data_cfg,
    )


def _build_broker(cfg: dict):
    brokers: dict[str, object] = {}
    for account in iter_alpaca_accounts(cfg):
        try:
            broker = AlpacaBroker(
                account.get("api_key", ""),
                account.get("api_secret", ""),
                account.get("base_url", ""),
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
            broker = IBKRBroker(
                account.get("host", "127.0.0.1"),
                int(account.get("port", 7497)),
                int(account.get("client_id", 1)),
                name=account["name"],
                account_id=account.get("account_id", ""),
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
        return BrokerRouter(brokers, exec_cfg.get("routing", {}))
    if len(brokers) == 1:
        return next(iter(brokers.values()))
    return None


def _resolve_data_cfg(cfg: dict, orchestrator_cfg: dict) -> dict:
    data_cfg = dict(cfg.get("data", {}))
    pretrain_cfg = orchestrator_cfg.get("rl", {}).get("pretrain", {})
    symbols_source = str(pretrain_cfg.get("symbols_source", "data"))
    max_symbols = int(pretrain_cfg.get("max_symbols", 20))
    if symbols_source != "alpaca_active_random":
        return data_cfg
    alpaca_cfg = get_alpaca_account_cfg(cfg)
    api_key = alpaca_cfg.get("api_key", "")
    api_secret = alpaca_cfg.get("api_secret", "")
    universe = load_universe(api_key, api_secret, "alpaca_active", max_universe=max(500, max_symbols))
    if universe:
        sample = random.sample(universe, min(len(universe), max_symbols))
        data_cfg["symbols"] = sample
    return data_cfg
