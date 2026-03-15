<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Fricktrade v3.0

Fricktrade is an automated algorithmic trading system for US equities (NYSE, Nasdaq) and crypto (24/7). It runs multiple signal-generating strategies in parallel, combines their votes, and routes orders through Alpaca and Binance brokers with layered risk controls, LLM-assisted oversight, and full observability.

**Current mode**: Paper trading on Alpaca (one or more paper accounts), with Binance Spot demo enabled.

## What It Does

- Trades US equities intraday (9:30–16:00 ET) and crypto continuously (24/7)
- Runs 9 active strategies simultaneously; their buy/sell/hold votes are tallied per symbol
- A majority vote (configurable threshold) triggers an order; ties are broken by `rl_policy`
- Orders route through Alpaca (equities + crypto) and Binance Spot (crypto only)
- A tactical LLM meta-orchestrator (Gemini 2.5 Flash) runs every 15 minutes to adjust config params within declared safe bounds
- A strategic meta-orchestrator runs weekly to set long-horizon baselines
- Prometheus + Grafana provide real-time observability; Alertmanager handles email alerts

## Architecture Overview

```
Market Data (Alpaca 5m bars + Binance)
        |
        v
 Market Cache (Redis + file)
        |
        v
 Symbol Manager (dynamic universe, AI filter, return ranker)
        |
        v
 Indicator Injection (~30 indicators per symbol)
        |
        v
 9 Strategies run in parallel per symbol
  [trend_following, factor_model, pattern_trading, stat_arb_pairs,
   top_movers_rf, crypto_momentum, crypto_mean_reversion,
   gap_reversal, earnings_drift]
        |
        v
 Vote Combiner  (majority vote; rl_policy tiebreaker)
        |
        v
 Risk Manager (daily loss cap, exposure caps, VaR/CVaR, stops)
        |
        v
 Execution Engine (limit orders, TWAP/VWAP/POV slicing)
        |
        v
 Order Queue (FIFO heap, pending guards, feedback loop)
        |
        v
 Broker Router -> Alpaca (paper) / Binance (spot demo)

Background threads:
 - Tactical Meta Orchestrator (every 15 min, Gemini)
 - Strategic Meta Orchestrator (weekly, Gemini)
 - Post-Session Analyst (daily, Gemini)
 - Macro Regime Analyzer (every 4h, Gemini + FRED)
 - RL online learner (every 60 min when market closed)
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU optional (required for Ollama news sentiment; CPU fallback available)
- Alpaca paper trading account (free at alpaca.markets)
- Google Gemini API key (for LLM meta-orchestrators and post-session analysis)

### Step 1: Clone and configure

```bash
git clone <repo>
cd Fricktrade
cp .env.example .env
```

Edit `.env` and fill in credentials. At minimum:

```
ALPACA_API_KEY=your_paper_key
ALPACA_API_SECRET=your_paper_secret
GOOGLE_GEMINI_API_KEY=your_gemini_key
```

### Step 2: Start the stack

```bash
./scripts/compose_up.sh
```

This script auto-detects GPU and uses `docker-compose.gpu.yml` overlay when available. Per-account Grafana dashboards are generated automatically from `.env`.

For CPU-only:
```bash
docker compose up -d --build
```

### Step 3: Verify

| Endpoint | URL |
|----------|-----|
| API health | http://localhost:18081/health |
| Config UI | http://localhost:18081/ui |
| Grafana dashboards | http://localhost:3002 (admin / `GRAFANA_PASSWORD` from `.env`) |
| Prometheus | http://localhost:9090 |
| Trader metrics | http://localhost:8001/metrics |

### Step 4: Watch the logs

```bash
docker compose logs -f trader
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ALPACA_API_KEY` | Yes | Alpaca API key (paper or live) |
| `ALPACA_API_SECRET` | Yes | Alpaca API secret |
| `BINANCE_API_KEY` | No | Binance API key (demo or live) |
| `BINANCE_API_SECRET` | No | Binance API secret |
| `BINANCE_BASE_URL` | No | Binance endpoint; default `https://demo-api.binance.com` |
| `GOOGLE_GEMINI_API_KEY` | Yes* | Required for LLM meta-orchestrators and post-session analysis |
| `ANTHROPIC_API_KEY` | No | Claude backend (optional alternative to Gemini) |
| `FRED_API_KEY` | No | FRED macro data; macro regime analyzer falls back to yfinance `^VIX` without it |
| `ALPHA_VANTAGE_API_KEY` | No | Earnings calendar for `earnings_drift` strategy |
| `COINGLASS_API_KEY` | No | CoinGlass OI data for crypto alt-data signals |
| `FRICKTRADE_API_TOKEN` | No | API auth token when `api.auth.enabled: true` |
| `GRAFANA_PASSWORD` | No | Grafana admin password (default: `admin`) |
| `KILL_SWITCH_CODE` | No | Required code to arm the kill switch |
| `ALERTMANAGER_SMTP_*` | No | SMTP settings for email alerts |

*Gemini key is required unless you disable all LLM features (`llm.enabled: false`).

## Services

| Service | Port | Purpose |
|---------|------|---------|
| `trader` | 8001 | Live trading loop + Prometheus metrics |
| `api` | 18081 | FastAPI config, health, and web UI |
| `market-cache` | — | Redis-backed 5m bar cache service |
| `redis` | 6379 | In-memory cache backend |
| `ollama` | 11434 | Local LLM (llama3.2:3b) for news catalyst extraction |
| `learner` | — | RL online training (runs when market is closed) |
| `prometheus` | 9090 | Metrics scrape |
| `grafana` | 3002 | Dashboards |
| `alertmanager` | 9094 | Email/webhook alerting |
| `healthwatch` | 9105 | Market-hours scheduler (stops/starts services) |
| `autoheal` | — | Auto-restarts unhealthy containers |
| `docker-socket-proxy` | — | Isolated Docker socket for autoheal/healthwatch |
| `calendar-updater` | — | Weekly market holiday refresh |
| `tests-when-closed` | — | Runs pytest + backtests when equity market is closed |
| `daily-report` | — | Daily top-movers email + training data export |

## Key Configuration

All configuration lives in `config/config.yaml`. Key sections:

| Section | Purpose |
|---------|---------|
| `brokers.*` | Broker credentials and account definitions |
| `strategy.combine` | `vote` (current) or `weights` |
| `strategy.names` | List of active strategies |
| `strategy.params.*` | Per-strategy and shared parameters |
| `risk.*` | Loss caps, stops, exposure limits |
| `risk.crypto.*` | Crypto-specific stops and limits |
| `trading_limits.*` | Min notional, fractional shares, shorts |
| `execution.*` | Order handling, limit orders, TWAP config |
| `llm.*` | LLM backends and module enables |
| `llm_orchestrator.*` | Portfolio orchestrator (currently in `vote` mode) |
| `tactical_meta_orchestrator.*` | 15-min Gemini tuner config and bounds |
| `data.dynamic_symbols.*` | Universe, AI filter, return ranker |
| `market.*` | Venue definitions, trading hours, holidays |

See `docs/DEPLOYMENT.md` for a full config walkthrough.

## Common Commands

```bash
# Download historical data
docker compose run --rm trader python3 -m app.main download \
  --config /app/config/config.yaml --symbols AAPL MSFT

# Run backtest
docker compose run --rm trader python3 -m app.main backtest \
  --config /app/config/config.yaml

# Walk-forward backtest
python3 scripts/benchmark_runner.py --config config/config.yaml --walk-forward

# Live trading (already the default container command)
docker compose run --rm trader python3 -m app.main trade \
  --config /app/config/config.yaml

# Re-enable GPU after OOM
./scripts/enable_gpu.sh
```

## Documentation

| File | Contents |
|------|---------|
| `docs/ARCHITECTURE.md` | System design, data flow, threading model |
| `docs/DEPLOYMENT.md` | Full deployment guide, all config keys |
| `docs/STRATEGIES.md` | All 9 strategies: what they do, parameters, limitations |
| `docs/DEVELOPMENT.md` | Developer guide: adding strategies, brokers, testing |
| `docs/OPERATIONS.md` | Operational runbook: monitoring, common issues, manual overrides |
| `docs/STATUS.md` | Current implementation status and roadmap |
| `AGENTS.md` | Commit-by-commit implementation history |
| `scripts/README.md` | All scripts with usage examples |

## Paper vs Live Trading

The default configuration uses Alpaca paper trading endpoints (`base_url: https://paper-api.alpaca.markets`). To switch to live:

1. Change `brokers.alpaca.base_url` to `https://api.alpaca.markets`
2. Replace paper API keys with live keys in `.env`
3. Review and tighten `risk.*` parameters before going live

Binance is currently configured for demo (`BINANCE_BASE_URL=https://demo-api.binance.com`). For live Binance, remove or blank `BINANCE_BASE_URL`.

## License

GNU Affero General Public License v3.0 (AGPLv3). See `LICENSE`.
Third-party attributions: `THIRD_PARTY_NOTICES.md`.
