# Fricktrade — Project Context

## What This Is
Fricktrade is a personal algorithmic trading system (~24,000+ lines Python) trading cryptocurrency perpetual futures (Binance/Bybit). It integrates reinforcement learning and dual-LLM reasoning (Claude for quality-critical tasks, Gemini for high-volume tasks).

## Architecture Overview
- Core trading engine: order management, position tracking, portfolio management
- Strategy layer: multiple configurable trading strategies
- Data pipeline: market data ingestion, storage, feature engineering
- RL orchestrator: reinforcement learning buy/hold/sell decision engine
- LLM integration: sentiment analysis, symbol filtering, post-session analysis, meta-orchestration, risk interpretation
- Risk management: position sizing, drawdown controls, exposure limits
- Infrastructure: Docker-native, structured logging, state persistence

## Known Issues
- Phantom short-entry signals (P0 bug, may still be present)
- Involuntary short position bug (P0)
- Strategy weights being ignored in some execution paths
- Concurrency/thread safety issues in shared state
- Broad exception catching masking real errors
- Alpha generation rated ~6.5-7/10 — infrastructure solid but edge unproven

## Market Context
- Primary market: crypto perpetual futures (Binance/Bybit)
- Strategic rationale: structural inefficiencies favorable to retail algo trading (funding rates, liquidation cascades, 24/7 markets, high leverage available)
- Secondary market: US equities (being phased out as primary focus)

## Key Files to Know
- Review the project structure with `find . -name "*.py" | head -50` and `cat` key entry points
- Strategy files contain the core trading logic
- Look for config files for exchange API setup and strategy parameters

## Development Practices
- Python 3.11+, type hints expected
- Docker containers for deployment
- Git for version control
- No CI/CD pipeline currently — manual testing
- Single developer (Nicola)
