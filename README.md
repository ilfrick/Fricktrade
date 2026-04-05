<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Fricktrade v3.0

Fricktrade is a personal algorithmic trading system for cryptocurrency. It runs two signal-generating strategies in weighted mode, combines their confidences against a conviction threshold, and routes orders through Alpaca (paper) and Binance (Spot demo) with layered risk controls, per-symbol LLM sentiment, and full observability via Grafana.

**Current mode**: Paper trading — Alpaca (2 paper accounts, crypto-only) + Binance Spot demo.

## What It Does

- Trades crypto 24/7 via 2 active strategies: `crypto_mean_reversion` (60% weight) and `crypto_momentum` (40% weight)
- Weighted combine: strategy confidences × weights vs `min_conviction` threshold (0.25, reduced to 0.15 when single-sided via SSCM)
- Per-symbol LLM sentiment (Ollama llama3.2:3b) modifies strategy confidence: momentum uses directional multiplier, MR uses contrarian logic
- Aggregate market sentiment gates entry ranking: bearish sentiment penalizes all entry scores by 30%
- Risk management: trailing stops, hard stops, ATR stops, circuit breakers, vol targeting, position sizing via Half-Kelly
- Prometheus + Grafana provide real-time observability; Alertmanager handles email alerts

## Architecture

```
Market Data (Alpaca + Binance, 1m bars)
        |
        v
 Symbol Manager (dynamic universe, AI filter, return ranker)
        |
        v
 Indicator Injection (~30 indicators per symbol)
        |
        v
 2 Strategies run per symbol
  [crypto_mean_reversion, crypto_momentum]
        |
        v
 Weighted Combine (confidence × weight vs min_conviction)
        |
        v
 Entry Ranking (BB %B, MFI, catalyst, funding, sentiment gate)
        |
        v
 Risk Manager (stops, drawdown cap, vol targeting)
        |
        v
 Order Queue (FIFO heap, pending guards, two-phase dispatch)
        |
        v
 Broker Router -> Alpaca (paper) / Binance (Spot demo)

Background:
 - Per-symbol sentiment (Ollama, 900s TTL)
 - Aggregate sentiment (Ollama, every 15 min)
 - Macro Regime (Gemini + FRED, every 4h)
 - Post-Session Analyst (Gemini, daily)
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU optional (Ollama runs on CPU, slower first call ~23s)
- Alpaca paper trading account (free at alpaca.markets)
- Google Gemini API key (for macro regime and post-session analysis)

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
| `GOOGLE_GEMINI_API_KEY` | Yes* | Required for macro regime and post-session analysis |
| `ANTHROPIC_API_KEY` | No | Claude backend (optional) |
| `FRED_API_KEY` | No | FRED macro data; macro regime falls back to yfinance `^VIX` without it |
| `COINGLASS_API_KEY` | No | CoinGlass OI data for crypto alt-data signals |
| `FRICKTRADE_API_TOKEN` | No | API auth token when `api.auth.enabled: true` |
| `GRAFANA_PASSWORD` | No | Grafana admin password (default: `admin`) |
| `KILL_SWITCH_CODE` | No | Required code to arm the kill switch |
| `ALERTMANAGER_SMTP_*` | No | SMTP settings for email alerts |

*Gemini key is required unless you disable all LLM features (`llm.enabled: false`).

## Documentation

**`AGENTS.md` is the single authoritative operational document.** It contains:

- Current system state with exact config values
- Strategy logic with entry/exit conditions
- Signal combine flow and conviction thresholds
- Risk management parameters and exit priority
- LLM integration and per-symbol sentiment flow
- Docker service reference with memory limits
- Operational runbook (common issues, debugging, rebuild)
- Monitoring reference (Prometheus metrics, Grafana, traces)
- Common pitfalls
- Extension guide

Files in `docs/` predate the v3.0 strategic reset (Mar 2026) and should be treated as archived planning material.

## Common Commands

```bash
# Rebuild and restart trader
docker compose build trader && docker compose up -d --force-recreate trader
# Also rebuild tests-when-closed (shares same codebase):
docker compose build tests-when-closed && docker compose up -d --force-recreate tests-when-closed

# Download backtest data
python3 scripts/download_crypto_backtest_data.py

# Run backtest
docker compose run --rm -T trader python3 -m app.main backtest

# Walk-forward backtest
python3 scripts/benchmark_runner.py --config config/config.yaml --walk-forward

# Check decision traces
docker exec fricktrade-trader-1 tail -5 /data/reports/decision_trace/trace_$(date -u +%Y-%m-%d).jsonl | python3 -m json.tool

# Push to both remotes
git push origin v3.0 && git push github v3.0
```

## License

GNU Affero General Public License v3.0 (AGPLv3). See `LICENSE`.
Third-party attributions: `THIRD_PARTY_NOTICES.md`.
