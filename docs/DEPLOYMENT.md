<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Deployment Guide

This guide covers everything needed to deploy Fricktrade from scratch: Docker setup, broker account creation, API keys, config walkthrough, and monitoring.

---

## Prerequisites

- Docker Engine >= 24 and Docker Compose >= 2.20
- Linux host (tested on Ubuntu 22.04+)
- Minimum 8 GB RAM, 20 GB disk
- NVIDIA GPU optional; CPU-only is fully supported

For GPU support: NVIDIA driver >= 525, `nvidia-container-toolkit` installed, Docker configured with `nvidia` runtime.

---

## Broker Account Setup

### Alpaca (Required)

1. Sign up at [alpaca.markets](https://alpaca.markets)
2. Create a **Paper Trading** account (free, no approval needed)
3. Generate API keys under "API Keys" in your paper trading dashboard
4. The default config uses two named accounts (`Realistic` and `Higher`) both pointing to the same Alpaca paper credentials — they share one API key but maintain separate `BrokerState` for risk and P&L tracking

For live trading: create a live brokerage account, pass KYC, and replace the paper keys.

### Binance (Optional)

1. Sign up at [binance.com](https://binance.com)
2. For demo (recommended to start): no account needed; use `BINANCE_BASE_URL=https://demo-api.binance.com` with any test keys from the demo dashboard
3. Generate API keys under "API Management"
4. Set `brokers.binance.enabled: true` in `config.yaml` to activate
5. Currently Spot mode only (`futures: false`); set `futures: true` for FAPI endpoints

---

## Environment Variables

Copy the template and fill in values:

```bash
cp .env.example .env
```

### Required

| Variable | Description |
|----------|-------------|
| `ALPACA_API_KEY` | Alpaca API key |
| `ALPACA_API_SECRET` | Alpaca API secret |
| `GOOGLE_GEMINI_API_KEY` | Google Gemini API key for LLM meta-orchestrators |

### Optional but Recommended

| Variable | Description | Impact if missing |
|----------|-------------|-------------------|
| `FRED_API_KEY` | Federal Reserve FRED API | Macro regime analyzer falls back to yfinance `^VIX` |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage | `earnings_drift` strategy produces no signals |
| `COINGLASS_API_KEY` | CoinGlass open interest | Crypto OI alt-data signal skipped |
| `ANTHROPIC_API_KEY` | Anthropic Claude | Only needed if switching any module to `backend: claude` |

### Optional Services

| Variable | Description |
|----------|-------------|
| `BINANCE_API_KEY` | Binance API key (for Binance broker) |
| `BINANCE_API_SECRET` | Binance API secret |
| `BINANCE_BASE_URL` | Binance endpoint; default `https://demo-api.binance.com` |
| `FRICKTRADE_API_TOKEN` | API auth token (only if `api.auth.enabled: true`) |
| `GRAFANA_PASSWORD` | Grafana admin password (default: `admin`) |
| `KILL_SWITCH_CODE` | Confirmation code required to arm the kill switch |
| `ALERTMANAGER_SMTP_SMARTHOST` | SMTP relay for email alerts (e.g., `smtp.gmail.com:587`) |
| `ALERTMANAGER_SMTP_FROM` | Alert sender email |
| `ALERTMANAGER_SMTP_USERNAME` | SMTP auth username |
| `ALERTMANAGER_SMTP_PASSWORD` | SMTP auth password or app password |
| `ALERTMANAGER_SMTP_REQUIRE_TLS` | `true` for TLS (default) |
| `ALERTMANAGER_EMAIL_TO` | Alert recipient email |

---

## Starting the Stack

### Auto-detect GPU (recommended)

```bash
./scripts/compose_up.sh
```

Detects GPU availability and adds `docker-compose.gpu.yml` overlay if found. Also generates per-account Grafana dashboards from `.env`.

### CPU only

```bash
docker compose up -d --build
```

### GPU explicit

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

---

## Stopping and Restarting

```bash
# Stop all services
docker compose down

# Restart a single service
docker compose restart trader

# Restart with config reload (trader reads config.yaml on start)
docker compose restart trader

# Full rebuild (after code changes)
docker compose up -d --build trader
```

The trader also exposes a restart endpoint:
```bash
curl -X POST http://localhost:18081/restart
```

---

## Configuration Walkthrough

All configuration is in `config/config.yaml`. Values support `${ENV_VAR}` interpolation.

### Application

```yaml
app:
  name: autotrader
  env: prod
  timezone: Europe/Rome   # Used for log timestamps; trading uses UTC internally
  log_level: INFO
```

### API

```yaml
api:
  auth:
    enabled: false              # Set true to require X-API-Key or Authorization: Bearer header
    token_env: FRICKTRADE_API_TOKEN
```

### Market Configuration

```yaml
market:
  open_mode: any              # 'any' = open if ANY trading venue is open (Crypto = always open)
  trading_venues:
    - NYSE
    - Nasdaq
    - Crypto
```

`is_market_open()` with `open_mode: any` returns `True` 24/7 because Crypto venue never closes. Use `is_venue_open(cfg, 'NYSE')` to check equity-only hours.

### Brokers

```yaml
brokers:
  alpaca:
    enabled: true
    base_url: https://paper-api.alpaca.markets   # Change to https://api.alpaca.markets for live
    api_key: ${ALPACA_API_KEY}
    api_secret: ${ALPACA_API_SECRET}
    accounts:
      - name: Realistic
        asset_filter: both      # both | crypto_only | equity_only
        risk:
          hard_stop_pct: 1.0    # Per-account risk override
      - name: Higher
        asset_filter: both
  binance:
    enabled: true               # Set false to disable Binance
    futures: false              # Spot only; set true for FAPI
    base_url: ${BINANCE_BASE_URL}   # blank = live; demo = https://demo-api.binance.com
    api_key: ${BINANCE_API_KEY}
    api_secret: ${BINANCE_API_SECRET}
    risk:
      max_crypto_exposure_pct: 95.0   # crypto-only broker; global 50% cap doesn't apply
```

### Risk

```yaml
risk:
  enabled: true
  max_daily_loss_pct: 3.0           # Block all new entries after 3% daily loss
  hard_stop_pct: 0.8                # Equity hard stop (% below avg entry)
  trailing_stop_pct: 5.0            # Equity trailing stop
  take_profit_pct: 1.5              # Equity take profit
  partial_take_profit_pct: 1.0      # Partial exit at this gain
  partial_take_profit_ratio: 0.5    # Sell this fraction of position
  circuit_breaker_drawdown_pct: 5.0  # Per-symbol circuit breaker threshold
  max_positions: 30
  max_portfolio_leverage: 1.5
  var:
    enabled: true
    max_var_pct: 4.0
    max_cvar_pct: 6.0
  exposure_caps:
    venues:
      NYSE: 60.0      # Max 60% in NYSE equities
      Crypto: 50.0    # Max 50% in crypto (global)
  pdt:
    force_swing: true              # Hold overnight instead of triggering PDT
    equity_threshold: 2500         # Alpaca PDT restriction threshold
  crypto:
    hard_stop_pct: 4.0             # Crypto hard stop (wider; more volatile)
    trailing_stop_pct: 1.5
    take_profit_pct: 4.0
    max_crypto_exposure_pct: 50.0  # Overridden per-broker (Binance: 95%)
```

### Trading Limits

```yaml
trading_limits:
  allow_shorts: false
  min_notional: 10.0              # Minimum USD order value (Alpaca crypto minimum)
  fractional_shares: true         # Enables sub-share sizing for equities
  crypto_order_margin: 0.97       # 3% haircut on crypto buys to absorb price drift
```

### Strategy

```yaml
strategy:
  combine: vote                   # 'vote' (current) or 'weights'
  names:                          # Active strategies
    - trend_following
    - factor_model
    - pattern_trading
    - stat_arb_pairs
    - top_movers_rf
    - crypto_momentum
    - crypto_mean_reversion
    - gap_reversal
    - earnings_drift
  params:
    min_hold_minutes: 15          # Suppress all exits for 15 min after entry
    exit_vote_threshold: 2        # Close position when ≥2 strategies say sell
    buy_vote_threshold: 2         # Open position when ≥2 strategies say buy
    alpha_decay_exit:
      enabled: true               # Exit when entry strategy reverses
      min_hold_minutes: 30
    regime_hold_minutes:
      enabled: true               # Hurst-scaled hold time
      base_minutes: 120
      trending_multiplier: 2.0    # Hold up to 240 min in trending regime
      mean_reverting_multiplier: 0.75   # Hold 90 min in mean-reverting regime
```

To disable a strategy without restarting, add `enabled: false` under its config:
```yaml
    trend_following:
      enabled: false
```

### Execution

```yaml
execution:
  symbol_executor_workers: 4
  stuck_cooldown_minutes: 15      # Suppress re-entry after timed-out buy
  stuck_blacklist_after: 3        # Blacklist symbol after this many consecutive stuck orders
  stuck_blacklist_hours: 2        # Blacklist duration
  limit_orders:
    enabled: true                 # Auto-upgrade market → limit at mid-price
    high_urgency_threshold: 0.8   # win_prob ≥ this → keep market order
  algos:
    default: twap
    min_notional: 500.0           # Only slice orders above this value
    twap:
      duration_seconds: 120
      slices: 4
  retry:
    max_order_age_seconds: 300    # Cancel stuck orders after 5 min
```

### Data

```yaml
data:
  provider: alpaca
  interval: 5m
  lookback_days: 7
  dynamic_symbols:
    enabled: true
    max_symbols: 150              # Total active universe after filtering
    universe: alpaca_active_all   # Equities + crypto from Alpaca
    crypto_quote_currencies: [USD]
    ai_filter:
      enabled: true               # PPO-based symbol filter
      return_ranker:
        enabled: true             # XGBoost return ranker overlay
```

### LLM

```yaml
llm:
  enabled: true
  macro_regime:
    enabled: true
    refresh_hours: 4
    backend: gemini
  sentiment:
    enabled: false                # Disabled in vote mode
  post_session:
    enabled: true
  risk_interpreter:
    enabled: true
  cost:
    daily_budget_usd: 5.00

llm_orchestrator:
  enabled: true
  mode: vote                      # 'vote' = no LLM calls per cycle; 'portfolio' = Gemini per cycle
  model: gemini-2.5-flash

tactical_meta_orchestrator:
  enabled: true
  interval_minutes: 15
  apply_delay_minutes: 5
  model: gemini-2.5-flash
  bounds:                         # All adjustable parameters and their safe ranges
    risk.hard_stop_pct: [0.5, 3.0]
    risk.crypto.hard_stop_pct: [1.5, 6.0]
    strategy.params.min_hold_minutes: [5, 30]
    # ... see config.yaml for full list
```

### Monitoring

```yaml
monitoring:
  prometheus_port: 8001
  audit:
    enabled: false                # JSONL audit log for each trade decision
  compliance:
    enabled: false                # JSONL + CSV compliance log
```

### Checkpointing

```yaml
checkpointing:
  enabled: true
  interval_seconds: 60
  dir: /data/checkpoints
  retention:
    max_age_hours: 72
    max_files: 200
```

---

## Health Checks and Monitoring

### Healthwatch

The `healthwatch` container probes all services every 30 seconds and manages market-hours sleep/wake cycles. In `partial` mode (current default), the trader stays running 24/7 for crypto; only `ollama` is stopped outside equity hours to save GPU.

Service states are written to `/data/system_state.json` for cross-service coordination.

### Grafana

Access at http://localhost:3002 (default credentials: admin / admin or `GRAFANA_PASSWORD`).

Dashboards are auto-generated from `.env` and placed in `grafana/provisioning/dashboards/`. Each Alpaca account gets its own dashboard. The meta-orchestrator dashboard (`meta_orchestrator.json`) shows tactical changes and health grade.

Key metrics to monitor:
- `trades_total{broker, side}` — fill counts
- `position_value_by_broker{broker, symbol}` — current exposure
- `skipped_orders_by_broker{broker, reason}` — rejections by reason
- `meta_orch_health_grade` — 0=green, 1=yellow, 2=red
- `meta_orch_strategy_weight{strategy}` — live weights (for when portfolio mode is enabled)

### Prometheus Alerts

Alert rules are in `prometheus/alerts.yml`. Alertmanager config is generated from `.env` into `alertmanager/alertmanager.generated.yml`.

Key alerts:
- `TraderLoopStale`: trading loop checkpoint > 15 min old
- `HighDailyLoss`: daily loss exceeds threshold
- `OrderRejected` (excluding `min_order_notional`)

---

## Grafana Dashboard Setup

Per-account dashboards are generated at startup by `./scripts/compose_up.sh`. To manually regenerate:

```bash
python3 scripts/generate_grafana_dashboards.py --config config/config.yaml
```

Import custom dashboards via Grafana UI: Settings → Dashboards → Import → upload JSON from `grafana/provisioning/dashboards/`.

---

## GPU Configuration

The GTX 1060 (6 GB VRAM) is allocated as:

| Workload | VRAM | Notes |
|----------|------|-------|
| Ollama (llama3.2:3b) | ~2.7 GB | Primary inference; 15–25× faster than CPU |
| RL inference/training | ~50–100 MB | Secondary; runs when market closed |
| AI filter (PPO) | ~15–20 MB | Tiny model |

Ollama starts only after trader is healthy (declared as `depends_on: trader: condition: service_healthy`) to avoid VRAM contention during initialization.

GPU state persists in `/data/gpu_state.json`. After CUDA OOM:
```bash
./scripts/enable_gpu.sh
docker compose restart trader
```

---

## Common Deployment Issues

### Trader loop stale (checkpoint age > 15 min)
- Check `docker compose logs trader` for errors
- Common cause: Redis connection failure → market cache unavailable → trader stalls on data fetch
- Fix: `docker compose restart redis market-cache trader`

### Alpaca authentication errors
- Verify `ALPACA_API_KEY` and `ALPACA_API_SECRET` in `.env`
- For paper trading, ensure keys are from the paper dashboard, not live

### Binance -2015 error (auth failure)
- The Binance demo API requires IP whitelisting
- Trader logs the server IP automatically on connection for whitelisting

### Ollama not starting / CUDA crash
- Check `docker compose logs ollama`
- If CUDA crash: `./scripts/enable_gpu.sh` then `docker compose restart ollama`
- Ollama depends on trader being healthy first; if trader is unhealthy, ollama won't start

### LLM budget exceeded
- `llm.cost.daily_budget_usd: 5.00` blocks all LLM calls when hit
- Check Grafana or logs for LLM cost metrics
- Budget resets at midnight UTC

### `insufficient_stablecoin` rejections for crypto
- Crypto market orders execute at market price; 1-3% drift can push fill above available balance
- `trading_limits.crypto_order_margin: 0.97` applies a 3% buffer at sizing time
- `execution.insufficient_stablecoin_cooldown_minutes: 5` prevents same-cycle retry

### PDT restriction blocking exits
- `risk.pdt.force_swing: true` holds positions overnight instead of triggering PDT
- Accounts above `equity_threshold: 2500` are not restricted
- PDT count clears daily at midnight ET

### Position marked as dust (< 1e-6 qty)
- These positions will never be traded again automatically
- To clear: manually close via Alpaca dashboard or API
- Restart will reset backoff state; first cycle may attempt and fail once before suppression kicks in
