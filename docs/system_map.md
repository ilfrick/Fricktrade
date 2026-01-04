<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# System Map

## Architecture Diagram

```mermaid
flowchart LR
    subgraph Entry[Entrypoints]
        CLI[app/main.py CLI]
        Compose[docker-compose.yml]
    end
    subgraph Core[Trading Loop]
        Trader[TradingAgent]
        Strategies[Strategies]
        Orchestrator[RL Orchestrator]
        Risk[Risk Manager]
        Exec[Execution Engine]
        Queue[Order Queue]
        Routing[Broker Routing]
    end
    subgraph Data[Data & Symbols]
        Scanner[Dynamic Scanner]
        AIFilter[AI Symbol Filter (PPO)]
        News[News Catalysts]
        MarketData[Market Data Providers]
    end
    subgraph Learning[Learning]
        RLTrain[PPO Training]
        RLOnline[PPO Online Updates]
        Registry[Model Registry]
        Drift[Drift Monitor]
    end
    subgraph Backtest[Backtesting & Benchmarks]
        AgentBT[Agent Backtest]
        LegacyBT[Legacy SMA]
        Bench[Benchmark Runner]
    end
    subgraph Observability[Monitoring & Ops]
        Metrics[Prometheus Metrics]
        Grafana[Grafana Dashboards]
        Healthwatch[Healthwatch]
        Audit[Audit/Compliance Logs]
        API[FastAPI Config UI]
    end
    subgraph Brokers[Brokers]
        Alpaca[Alpaca]
        IBKR[IBKR]
    end

    CLI --> Trader
    Compose --> Trader
    Trader --> Strategies --> Orchestrator --> Risk --> Exec --> Queue --> Routing --> Brokers
    Scanner --> Trader
    AIFilter --> Trader
    News --> Strategies
    MarketData --> Trader
    RLTrain --> Registry --> Trader
    RLOnline --> Registry
    Drift --> Trader
    Trader --> Metrics --> Grafana
    Healthwatch --> Trader
    Trader --> Audit
    API --> Trader
    AgentBT --> Trader
    LegacyBT --> Bench
```


## Entrypoints
- `app/main.py`: CLI entrypoint for `trade`, `backtest`, `download`, `ingest`, `api`, `train`, `online-train`, `evaluate`, `pretrain-orchestrator`.
- `docker-compose.yml`: service orchestration (trader, api, learner, tests-when-closed, healthwatch, etc).

## Trading Loop (Core Runtime)
- `app/agents/trader.py`: main loop orchestration; symbol refresh, market gating, strategy execution,
  risk checks, order queue, metrics, audit/compliance.
- Flow: symbols -> market state -> strategies -> orchestrator -> risk -> execution -> metrics/logging.

## Strategies
- `app/strategies/`: signal generators (intraday momentum, trend_following, factor_model,
  stat_arb_pairs, market_maker, pattern_trading, rl_policy, rl_policy_fees).
- `app/strategies/base.py`: strategy interface.

## Orchestrator
- `app/agents/orchestrator.py`: RL policy-gradient selector over strategy signals; uses AI filter
  features and order feedback; online updates + pretraining.

## Risk
- `app/risk/manager.py`: core limits (exposure, leverage, daily loss).
- `app/risk/haircut.py`: stress/liquidity haircuts.
- Risk gating is integrated in `app/agents/trader.py`.

## Execution
- `app/execution/executor.py`: broker-agnostic order placement.
- `app/execution/order_queue.py`: FIFO queue and feedback loop.
- `app/execution/routing.py`: multi-broker routing logic.
- `app/brokers/`: Alpaca/IBKR adapters + router.

## Data and Market State
- `app/data/downloader.py`: yfinance historical download.
- `app/data/ingestion.py`: ingestion from yfinance/alpaca/stooq/alphavantage.
- `app/data/scanner.py`: symbol universe and price filter.
- `app/data/ai_filter.py`: PPO-based symbol scorer with online updates.
- `app/data/news.py`: broker-backed news/catalyst fetch.

## Learning (Trading Policy)
- `app/learning/train_rl.py`: offline PPO training.
- `app/learning/online_update.py`: online PPO updates.
- `app/learning/evaluate.py`: evaluation.
- `app/learning/env.py`: Gymnasium environment.
- `app/learning/features.py`: observation construction.
- `app/learning/registry.py`: model registry + active pointer.
- `app/learning/drift.py`: feature/PnL drift detection.

## Backtesting and Benchmarking
- `app/backtest/agent_engine.py`: agent-aligned backtests (real loop).
- `app/backtest/engine.py`: legacy SMA backtest.
- `scripts/benchmark_runner.py`: benchmark suite and scenarios.

## Monitoring and Ops
- `app/monitoring/metrics.py`: Prometheus metrics.
- `app/monitoring/healthwatch.py`: health scheduler + ops state file.
- `app/monitoring/audit.py`: audit/compliance logging.
- Grafana dashboards: `grafana/provisioning/dashboards/*.json`.
- Prometheus config: `prometheus/prometheus.yml` + `prometheus/alerts.yml`.

## API and UI
- `app/api/server.py`: FastAPI config UI, schema, restart trigger.

## Configuration
- `config/config.yaml`: all runtime config; supports `${ENV_VAR}` interpolation.
