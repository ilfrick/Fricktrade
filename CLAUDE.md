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
**`AGENTS.md`** is the single authoritative operational document. Living docs are in `docs/`
(see `docs/README.md`): the development plan (`docs/PLAN-2026-Q4.md`) and the 2026-11-14 audit
(`docs/audit/`). `docs/archive/` is pre-v3.0 planning material.

## Known Issues
- Alpha generation unproven — and the 2026-11-14 audit found the validation harness itself has
  five measured methodological failures, so "no edge" is an open hypothesis, not a settled result.
  See `docs/audit/2026-11-14-bug.md` section B.
- Five critical execution bugs are latent only because the alpha gate currently places no orders
  (`docs/audit/2026-11-14-bug.md` section C). Fix them before ever unblocking the gate.
- Binance demo dust positions (~30 sub-LOT_SIZE) cycle through circuit breaker → 8h backoff every
  restart (normal, demo API limitation)
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
