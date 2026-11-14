<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Developer Guide

This guide covers repository structure, code conventions, how to add new strategies and brokers, testing, and how to extend the LLM subsystem.

---

## Repository Structure

```
Fricktrade/
├── app/
│   ├── agents/
│   │   ├── trader.py           # Main TradingAgent class (~3000 lines; core loop)
│   │   ├── symbol_manager.py   # Symbol selection, AI filter, venue mapping
│   │   ├── performance.py      # Trade recording, P&L, kill switch logic
│   │   ├── open_orders.py      # Open order cache and pending-order checks
│   │   ├── account_metrics.py  # Equity tracking, drawdown, VaR/CVaR
│   │   ├── orchestrator.py     # SimpleOrchestrator (vote/weights); RL orchestrator stub
│   │   ├── strategy_config.py  # Per-strategy enable/disable and config helpers
│   │   ├── decision_context.py # Decision trace record building
│   │   └── trace_helpers.py    # Trace serialization helpers
│   ├── strategies/
│   │   ├── base.py             # Strategy ABC
│   │   ├── trend_following.py
│   │   ├── factor_model.py
│   │   ├── pattern_trading.py
│   │   ├── stat_arb_pairs.py
│   │   ├── top_movers_rf.py
│   │   ├── crypto_momentum.py
│   │   ├── crypto_mean_reversion.py
│   │   ├── gap_reversal.py
│   │   ├── earnings_drift.py
│   │   ├── rl_policy.py        # PPO policy inference (disabled)
│   │   ├── intraday_momentum.py  # Inactive
│   │   └── market_maker.py     # Inactive
│   ├── brokers/
│   │   ├── base.py             # Broker abstract base class
│   │   ├── alpaca.py           # Alpaca broker (equities + crypto)
│   │   ├── binance.py          # Binance broker (Spot + Futures)
│   │   ├── ibkr.py             # IBKR broker (disabled)
│   │   ├── router.py           # BrokerRouter multi-broker dispatch
│   │   └── config_utils.py     # iter_alpaca_accounts, iter_binance_accounts
│   ├── execution/
│   │   ├── executor.py         # ExecutionEngine (limit order upgrade, slippage)
│   │   ├── order_queue.py      # FIFO order heap + feedback loop
│   │   ├── algos.py            # TWAP, VWAP, POV, adaptive_slices, fractional_slice
│   │   ├── smart_router.py     # SmartOrderRouter (Almgren-Chriss impact model)
│   │   ├── tca.py              # TCA feedback (slippage EWMA)
│   │   ├── impact.py           # Market impact estimation
│   │   ├── routing.py          # Symbol-to-broker routing utilities
│   │   └── config.py           # Execution config helpers
│   ├── risk/
│   │   ├── manager.py          # RiskManager (daily loss, leverage, circuit breaker)
│   │   ├── config.py           # Risk config helpers
│   │   └── haircut.py          # Liquidity haircut calculations
│   ├── llm/
│   │   ├── client.py           # LLMClient with ClaudeBackend + GeminiBackend
│   │   ├── macro_regime.py     # MacroRegimeAnalyzer (FRED + Gemini, 4h TTL)
│   │   ├── meta_orchestrator.py  # StrategicOrchestrator (weekly)
│   │   ├── tactical_meta_orchestrator.py  # TacticalMetaOrchestrator (15 min)
│   │   ├── portfolio_orchestrator.py  # LLMPortfolioOrchestrator (disabled)
│   │   ├── post_session.py     # PostSessionAnalyst (daily)
│   │   ├── risk_interpreter.py # RiskEventInterpreter
│   │   ├── sentiment.py        # Per-symbol sentiment (disabled in vote mode)
│   │   ├── ollama_sentiment.py # Aggregate market sentiment via local Ollama
│   │   └── symbols_filter.py   # LLM symbol selection (disabled)
│   ├── data/
│   │   ├── ai_filter.py        # PPO-based AI symbol filter
│   │   ├── alt_data.py         # Fear&Greed, CoinGlass OI, SEC EDGAR
│   │   ├── binance_market_data.py  # Binance bar fetcher + BinanceMarketDataProvider
│   │   ├── downloader.py       # yfinance data download
│   │   ├── earnings_calendar.py  # Alpha Vantage earnings calendar
│   │   ├── ingestion.py        # Alpaca multi-year bar ingestion
│   │   ├── market_cache.py     # Redis + file cache interface
│   │   ├── market_cache_service.py  # Cache service daemon (runs in market-cache container)
│   │   ├── news.py             # Alpaca news API + Ollama catalyst extraction
│   │   ├── news_rss.py         # Multi-source RSS news (CoinDesk, Reuters, etc.)
│   │   ├── quality.py          # Data quality checks
│   │   ├── quote_stream.py     # Real-time bid/ask WebSocket (disabled)
│   │   ├── scanner.py          # Universe scanning from Alpaca
│   │   └── yfinance_utils.py   # yfinance helpers
│   ├── backtest/
│   │   ├── agent_engine.py     # Runs TradingAgent on historical CSV (agent backtest)
│   │   ├── engine.py           # Simple SMA backtest engine (legacy)
│   │   └── sampling.py         # Walk-forward window builder
│   ├── learning/
│   │   ├── env.py              # RL gym environment
│   │   ├── train_rl.py         # PPO training entry point
│   │   ├── online_train.py     # Online update loop
│   │   ├── live_rewards.py     # Live reward recording for online training
│   │   ├── registry.py         # Model registry (snapshot + drift detection)
│   │   ├── drift.py            # DriftMonitor
│   │   └── evaluate.py         # Model evaluation
│   ├── monitoring/
│   │   ├── metrics.py          # All Prometheus metrics definitions
│   │   ├── broker_metrics.py   # Broker call latency recording
│   │   ├── audit.py            # AuditLogger + ComplianceLogger
│   │   └── healthwatch.py      # Market-hours scheduler daemon
│   ├── api/
│   │   └── server.py           # FastAPI server (/health, /config, /ui, /restart)
│   ├── utils/
│   │   ├── market.py           # is_market_open, is_venue_open, next_market_open
│   │   ├── config.py           # load_config (YAML + env interpolation)
│   │   ├── checkpoint.py       # load/save checkpoint
│   │   ├── structured_log.py   # StructuredLogger (JSON trade events)
│   │   ├── signal_features.py  # compute_all_indicators
│   │   └── volatility.py       # realized_volatility_pct
│   └── main.py                 # CLI entrypoint (trade, backtest, download, api, train…)
├── config/
│   └── config.yaml             # All runtime configuration
├── tests/                      # pytest test suite (~42 test files, ~182 tests)
├── scripts/                    # Operational scripts
├── docker/
│   ├── Dockerfile              # CPU image
│   ├── Dockerfile.gpu          # GPU image (CUDA)
│   └── Dockerfile.light        # Lightweight image for daily-report
├── grafana/                    # Grafana provisioning + dashboard JSON
├── prometheus/                 # Prometheus scrape + alert rules
├── alertmanager/               # Alertmanager config template
├── docs/                       # Documentation (this directory)
└── docker-compose.yml          # All services
```

---

## Code Conventions

### Timezone Awareness

All datetime operations use timezone-aware objects. `datetime.utcnow()` has been fully migrated out (zero occurrences remain). Always use `datetime.now(timezone.utc)` or `datetime.now(tz)`.

### Decimal Arithmetic

For sell quantity calculations, always use `math.floor()`, never `round()`. Round can round up to a value marginally above the held quantity (due to float64 precision of broker decimal strings), causing `insufficient_quantity` rejections.

Pattern for safe sell qty:
```python
factor = 10 ** decimal_places
qty = math.floor(raw_qty * factor) / factor
if qty >= current_qty:
    qty = max(0.0, (math.floor(current_qty * factor) - 1) / factor)
```

### Config Access

Never hard-code config values. Read from the `cfg` dict. Use `cfg.get("key", default)` for optional values. Use `merge_cfg()` from `app/brokers/config_utils.py` to merge account-level overrides with global config.

### Threading

Shared mutable state that is accessed from multiple threads must be protected by a lock. Use `threading.Lock()` for simple guards. Never hold a lock across I/O (broker calls, LLM calls, etc.).

The order queue uses `threading.Lock()` for its heap and all `_pending_*` dicts. Each `_pending_*` dict has its own lock.

### Error Handling

Use narrow exception handling. Catch only the specific exceptions you expect and can handle. Never `except Exception` at the top level without logging and re-raising or taking specific recovery action.

### Logging

Use `logging.getLogger(__name__)` at module level. Use `_slog.event()` for structured trade events that should appear in decision traces. Use `logger.warning()` / `logger.error()` for operational issues.

---

## Adding a New Strategy

### Step 1: Create the strategy file

Create `app/strategies/my_strategy.py`:

```python
from app.strategies.base import Strategy

class MyStrategy(Strategy):
    def __init__(self, params: dict):
        cfg = params.get("my_strategy", {}) if isinstance(params, dict) else {}
        self.threshold = float(cfg.get("threshold", 0.5))

    def generate_signal(self, market_state: dict) -> dict:
        symbol = market_state.get("symbol", "")
        prices = market_state.get("prices", []) or []

        if len(prices) < 10:
            return {"action": "hold", "confidence": 0.0, "name": "my_strategy"}

        # Your signal logic here
        action = "buy"  # or "sell" or "hold"
        confidence = 0.7

        return {"action": action, "confidence": confidence, "name": "my_strategy"}
```

Rules:
- Return `name` matching the strategy key used in config
- Confidence should be in [0, 1]; will be calibrated by `ConfidenceCalibrator`
- Return `hold` when data is insufficient rather than erroring
- For crypto-only strategies: check `"/" in symbol` and return hold for non-crypto
- For equity-only strategies: check `"/" not in symbol` and return hold for crypto

### Step 2: Register in trader.py

In `app/agents/trader.py`, import and instantiate:

```python
from app.strategies.my_strategy import MyStrategy

# In TradingAgent.__init__, in the strategy initialization block:
strategies.append(MyStrategy(strategy_params))
```

### Step 3: Add config

In `config/config.yaml`:

```yaml
strategy:
  names:
    - my_strategy   # Add to active list
  params:
    my_strategy:
      threshold: 0.5   # Strategy-specific params
      enabled: true    # Optional: per-strategy kill switch
```

### Step 4: Write a test

Create `tests/test_my_strategy.py`:

```python
from app.strategies.my_strategy import MyStrategy

def test_hold_on_insufficient_data():
    s = MyStrategy({})
    result = s.generate_signal({"prices": [1.0]})
    assert result["action"] == "hold"

def test_buy_signal():
    s = MyStrategy({"my_strategy": {"threshold": 0.5}})
    market_state = {"symbol": "AAPL", "prices": list(range(100, 120)), ...}
    result = s.generate_signal(market_state)
    assert result["action"] in ("buy", "sell", "hold")
    assert 0.0 <= result["confidence"] <= 1.0
```

---

## Adding a New Broker

### Step 1: Implement the Broker interface

Create `app/brokers/my_broker.py`. Implement all methods from `app/brokers/base.py`:

```python
from app.brokers.base import Broker

class MyBroker(Broker):
    def __init__(self, api_key: str, api_secret: str, name: str = "mybroker"):
        self._name = name
        # Initialize your client

    def get_account(self) -> dict: ...
    def get_positions(self) -> dict: ...
    def get_open_orders(self) -> list: ...
    def place_order(self, symbol, qty, side, order_type, **kwargs) -> str: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def get_order_status(self, order_id: str) -> dict: ...
    def close_position(self, symbol: str) -> bool: ...
    def is_connected(self) -> bool: ...
    def get_asset_class(self, symbol: str) -> str: ...
    def supports_short(self) -> bool: ...
    def get_today_deposits(self) -> float: ...
```

### Step 2: Wire into config_utils.py

Add an iterator function in `app/brokers/config_utils.py`:

```python
def iter_mybroker_accounts(cfg: dict):
    broker_cfg = cfg.get("brokers", {}).get("mybroker", {})
    if not broker_cfg.get("enabled", False):
        return
    yield "mybroker", broker_cfg
```

### Step 3: Wire into main.py

In `app/main.py`, in the `_build_broker()` function, add:

```python
from app.brokers.my_broker import MyBroker
from app.brokers.config_utils import iter_mybroker_accounts

for name, broker_cfg in iter_mybroker_accounts(cfg):
    broker = MyBroker(
        api_key=broker_cfg["api_key"],
        api_secret=broker_cfg["api_secret"],
        name=name,
    )
    brokers[name] = broker
```

### Step 4: Add config section

```yaml
brokers:
  mybroker:
    enabled: true
    api_key: ${MYBROKER_API_KEY}
    api_secret: ${MYBROKER_API_SECRET}
    accounts: []
    fees:
      commission_pct: 0.05
```

---

## Testing

### Running Tests

```bash
# Run all tests
docker compose run --rm trader pytest tests/ -v

# Run specific test file
docker compose run --rm trader pytest tests/test_risk_manager.py -v

# Run with coverage
docker compose run --rm trader pytest tests/ --cov=app --cov-report=html

# On host (requires virtualenv with requirements installed)
pytest tests/ -v
```

### Test Categories

182 tests, 17 skipped without tensorflow/prometheus:

| Category | Files | What's Tested |
|----------|-------|---------------|
| Unit: risk | `test_risk_manager.py`, `test_risk_governance.py`, `test_risk_haircut.py` | RiskManager state, daily loss, exposure caps |
| Unit: execution | `test_execution_algos.py`, `test_order_queue.py`, `test_order_queue_ext.py` | TWAP slicing, queue ordering, pending guards |
| Unit: strategies | `test_strategy_models.py`, `test_strategy_upgrades.py`, `test_trader_signals.py` | Signal generation, vote counting |
| Unit: brokers | `test_broker_router.py`, `test_routing.py` | Broker dispatch, symbol routing |
| Integration | `test_two_phase_dispatch.py`, `test_session_fixes.py` | Full cycle behavior |
| Market hours | `test_market_hours_extended.py`, `test_trading_venues_filter.py` | Venue gating |
| Data | `test_data_quality.py`, `test_market_cache.py` | Cache, quality checks |
| LLM | (via mocks) | LLM client backends |

### What Needs Manual Testing

- Live broker connections (Alpaca, Binance)
- LLM API calls (Gemini, Claude)
- GPU inference (AI filter, RL policy)
- End-to-end trading cycle with real market data
- Multi-day P&L accumulation
- PDT restriction behavior near the $2500 threshold

### Test Conventions

- Use `unittest.mock.patch` for broker and LLM calls
- Use `pytest.mark.skip(reason="requires tensorflow")` for GPU-dependent tests
- Build minimal market state dicts (see `test_strategy_models.py` for examples)
- Test edge cases: empty prices list, None values, zero qty, dust positions

---

## Running Backtests

### Agent Backtest (recommended)

Runs the real `TradingAgent` loop on historical CSV data:

```bash
docker compose run --rm trader python3 -m app.main backtest \
  --config /app/config/config.yaml
```

Config controls: `backtest.start`, `backtest.end`, `backtest.symbols`, `backtest.initial_cash`.

### Walk-Forward Backtest

```bash
python3 scripts/benchmark_runner.py --config config/config.yaml --walk-forward
```

Produces report at `data/reports/benchmark_report.json` and plots at `data/reports/benchmark_plots/`.

### Download Historical Data

```bash
docker compose run --rm trader python3 -m app.main download \
  --config /app/config/config.yaml --symbols AAPL MSFT NVDA
```

Data is stored in `/data/` as CSV files.

---

## LLM Subsystem Extension

### Switching Backends

Each LLM module has its own `backend:` config key. Supported values: `gemini`, `claude`, `ollama`.

```yaml
llm:
  macro_regime:
    backend: claude   # Switch this module to Claude
```

The `LLMClient` in `app/llm/client.py` routes based on backend name. `complete()` accepts an optional `model` kwarg to override per-call.

### Adding a New LLM Module

1. Create `app/llm/my_module.py`
2. Use `LLMClient` from `app/llm/client.py`:

```python
from app.llm.client import LLMClient

class MyLLMModule:
    def __init__(self, cfg: dict, llm_client: LLMClient):
        self._cfg = cfg
        self._client = llm_client

    def run(self, context: dict) -> dict:
        prompt = self._build_prompt(context)
        response = self._client.complete(
            backend=self._cfg.get("backend", "gemini"),
            system_prompt="...",
            user_prompt=prompt,
            max_tokens=1024,
        )
        return self._parse(response)
```

3. Wire into `trader.py:_init_llm()` and call from the appropriate place in the trading loop.
4. Add config section under `llm:`.

### Budget Awareness

All LLM calls go through `LLMClient.complete()` which tracks cost against `llm.cost.daily_budget_usd`. When the budget is exceeded, `complete()` raises `LLMBudgetExceeded`. Handle this by logging and returning a safe default.

---

## Common Pitfalls

### Sell Quantity Rounding

**Never use `round()` for sell qty.** Use `math.floor()`. Float64 representation of broker decimal quantities can be marginally larger than the decimal string, causing `round()` to round up and then the broker to reject with `insufficient_quantity`.

### Update vs. Accumulate P&L

`record_pnl()` accumulates deltas. `update_daily_loss()` sets the absolute day P&L. Using the wrong one causes the daily loss tracker to compound errors (e.g., reaching -543% while real loss is -1.2%). See commit `7fb438b`.

### Order Queue Sentinel Return Value

`order_queue.enqueue()` returns `"queued"` (truthy string) when the order is accepted to the heap but not started (because an active order is already running for that broker). Previously it returned `None`, which caused callers to incorrectly record `order_failed` and skip `_reserve_pending_buy`. Always check `if result == "queued"` explicitly.

### Two-Phase Symbol Batches

`_build_symbol_batches()` MUST intersect with the provided `symbols` set. Without this, caller-side filters (e.g., crypto-only when equity market is closed) are silently bypassed. See commit `5baad58`.

### Crypto vs. Equity Stops

The `_check_position_exit()` function reads stops from `risk.crypto.*` for symbols containing `/`, falling back to `risk.*` for equities. Old code always used flat values. If you add a new stop condition, make sure to include the crypto/equity branch.

### `is_market_open()` vs. `is_venue_open()`

`is_market_open()` with `open_mode: any` returns True 24/7 when Crypto is in `trading_venues`. To check equity-only hours, use `is_venue_open(cfg, 'NYSE')` explicitly.

### Binance Asset Class Check

`str(AssetClass.CRYPTO).lower()` returns `"assetclass.crypto"`, not `"crypto"`. Always check with `"crypto" in asset_class` rather than `== "crypto"`.

### Stablecoin Symbols

`/USDT`-quoted symbols (e.g., `BTC/USDT`) must not appear in Alpaca batches. The filter in `build_symbols_by_broker()` prevents this. If adding new data paths, apply the same filter.

### Meta-Orchestrator Prompt Kwargs

TacticalMetaOrchestrator calls use `system_prompt=` and `user_prompt=` kwargs. These must match the `LLMClient.complete()` signature exactly. Wrong kwarg names cause silent failures (no LLM call made, no error logged). See commit fixing Mar 11 breakage.
