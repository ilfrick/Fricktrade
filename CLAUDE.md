# Fricktrade — Project Context

## What This Is
Fricktrade is a personal algorithmic trading system (~24,000+ lines Python) trading cryptocurrency (Spot) via Alpaca paper and Binance Spot demo. It uses two weighted strategies (crypto_mean_reversion + crypto_momentum), per-symbol LLM sentiment via Ollama, and Gemini for macro regime analysis and post-session review.

## Architecture Overview
- Core trading engine: `app/agents/trader.py` — order management, position tracking, two-phase dispatch
- Strategy layer: 2 active strategies in weighted combine mode (MR 60%, momentum 40%)
- Data pipeline: Alpaca + Binance 1m bars, ~30 indicators, order book depth
- LLM integration: per-symbol Ollama sentiment, aggregate sentiment, Gemini macro regime, post-session analyst
- Risk management: trailing/hard/ATR stops, vol targeting, circuit breakers, Half-Kelly sizing
- Infrastructure: Docker-native, Prometheus + Grafana, structured logging, state persistence

## Authoritative Documentation
**`AGENTS.md`** is the single authoritative operational document. It contains all current config values, strategy logic, risk parameters, LLM flow, Docker services, operational runbook, and common pitfalls. Refer to it for any operational question.

Files in `docs/` predate the v3.0 strategic reset and are archived planning material.

## Known Issues
- Alpha generation unproven — infrastructure solid but edge not yet demonstrated in live trading
- Binance demo dust positions (~30 sub-LOT_SIZE) cycle through circuit breaker → 8h backoff every restart (normal, demo API limitation)
- `tests-when-closed` writes to same decision trace file as trader (filter by timestamp)

## Market Context
- Primary market: crypto (Alpaca paper + Binance Spot demo)
- Equity trading disabled (`asset_filter: crypto_only`)
- 24/7 operation

## Development Practices
- Python 3.11+, type hints expected
- Docker containers for deployment
- Git for version control; push to both `origin` (housefz) and `github` remotes
- No CI/CD pipeline — manual testing (193 tests, 16 skip without tensorflow/prometheus)
- Single developer (Nicola)
