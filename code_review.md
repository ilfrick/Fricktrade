# Code Review — Fricktrade 3.0

**Progetto**: Automated multi-broker trading system con RL strategy orchestration  
**LOC**: ~24.000 righe Python | ~1.100 righe di test | 12 servizi Docker  
**Data review**: 6 Febbraio 2026

---

## Valutazione complessiva

Il progetto è ambizioso e ben strutturato architetturalmente: un sistema di trading live multi-broker con reinforcement learning, HMM regime detection, portfolio optimization, audit trail e monitoring Prometheus/Grafana. Il livello di feature-completeness è impressionante per un progetto personale. Detto questo, ci sono alcuni problemi strutturali che meritano attenzione, soprattutto considerando che questo codice gestisce denaro reale.

---

## 🔴 Problemi critici

### 1. ~~`trader.py` è un God Object (3.584 righe)~~ ✅ Risolto

Decomposizione completata. Estratte le seguenti classi dedicate:
- `SymbolManager` (`app/agents/symbol_manager.py`) — dynamic symbols, AI filter, universe resolution
- `PerformanceTracker` (`app/agents/performance.py`) — strategy performance, trade records, kill switch
- `AccountMetricsUpdater` (`app/agents/account_metrics.py`) — equity tracking, VaR/CVaR, drawdown
- `OpenOrderManager` (`app/agents/open_orders.py`) — order cache, pending checks, metrics
- `MarketStateBuilder` (`app/agents/market_state.py`) — market state construction

`trader.py` ridotto da ~3.584 a ~2.437 righe.

### 2. Concorrenza fragile: `self._lock` usato inconsistentemente

`run_once()` acquisisce e rilascia il lock **tre volte** nello stesso metodo (righe ~700-900), con blocchi di codice "UNLOCKED" nel mezzo che accedono a `self._regime_returns_buffer`, `market_state` mutato, e `self._orchestrator`. Il pattern lock/unlock/lock è rischioso:

```python
with self._lock:
    # prepare strategies
    ...

# UNLOCKED: Run strategy inference
signals = []
for name, strategy in strategies_to_run:
    signal = strategy.generate_signal(market_state)  # market_state è condiviso!

with self._lock:
    self._update_orchestrator(...)

# UNLOCKED: Orchestrator inference
names, weights = self._orchestrator.select(...)

with self._lock:
    # execution logic...
```

`market_state` è un dizionario mutato da più parti senza protezione. `self._regime_returns_buffer` è letto/scritto fuori dal lock. In `_run_symbol_batch`, un `ThreadPoolExecutor(max_workers=4)` esegue `_process_single_symbol` in parallelo, dove ciascun thread passa per questi stessi percorsi.

**Suggerimento**: O si fa tutto single-threaded (più sicuro dato il costo I/O reale del broker), oppure si passa a un pattern actor/message-queue dove ogni simbolo ha il suo stato isolato.

### 3. Broad `except Exception` in 195 punti

Quasi tutti gli errori sono catturati con `except Exception` e loggati con un warning. Questo nasconde bug reali — un `KeyError` in un calcolo di risk viene trattato come una condizione normale:

```python
except Exception as exc:
    logging.warning("Portfolio position scale failed: %s", exc)
    return 1.0  # Silently returns default, risk miscalculation hidden
```

**Suggerimento**: Usare eccezioni specifiche (`ConnectionError`, `TimeoutError`, `ValueError`) e lasciare che errori di programmazione (`KeyError`, `AttributeError`, `TypeError`) propaghino fino al livello superiore. Almeno nei path critici di risk e sizing.

---

## 🟡 Problemi importanti

### 4. Rapporto test/codice insufficiente per un sistema finanziario

1.143 righe di test per 23.657 righe di codice. Il file più critico (`trader.py`, 3.584 righe) non ha test dedicati. `risk/manager.py` ha solo 18 righe di test. L'intero modulo `execution/` (order queue, smart router, TCA) ha 23 righe di test per gli algoritmi.

Moduli senza test: `orchestrator.py`, `pipeline.py`, `brokers/alpaca.py`, `brokers/ibkr.py`, `portfolio/optimizer.py`, `portfolio/rebalance.py`, `learning/train_rl.py`, `main.py`.

**Suggerimento**: Prioritizzare test per:
1. `_size_order()` — è il calcolo più critico, determina quanti soldi vengono messi a rischio
2. `_combine_signals()` — decide se comprare o vendere
3. `run_once()` — il flow completo con mock del broker
4. `RiskManager.can_open_trade()` — edge cases con daily loss reset

### 5. ~~Codice duplicato~~ ✅ Parzialmente risolto

`_realized_volatility_pct()` consolidata in `app/utils/volatility.py`, usata sia da `trader.py` che da `execution/impact.py`. Structured logging estratto in `app/utils/structured_log.py`. Rimangono duplicazioni minori nei pattern di estrazione equity/cash.

### 6. Nessuna validazione dei dati di mercato in ingresso

`market_state` è un dizionario non tipizzato che viene passato ovunque. Non c'è validazione che `prices` sia una lista non vuota, che `last_price` sia positivo, o che i volumi abbiano senso. Se il data provider restituisce dati corrotti, il sistema li processa silenziosamente:

```python
last_price = market_state.get("last_price")
# last_price could be None, 0, negative, NaN...
qty = int(allowed_value // last_price)  # ZeroDivisionError if 0, crash if None
```

**Suggerimento**: Creare un `@dataclass` o Pydantic model per `MarketState` con validazione. Almeno aggiungere un check esplicito `if not last_price or last_price <= 0` prima di ogni calcolo di sizing.

### 7. ~~`datetime.utcnow()` è deprecato~~ ✅ Risolto

Migrato a `datetime.now(timezone.utc)` in tutto il progetto.

---

## 🟢 Aspetti positivi

### Architettura ben stratificata
La separazione broker/strategy/execution/risk/learning è chiara e ben pensata. Il pattern `Broker` ABC con `BrokerRouter` per il multi-broker è pulito e estensibile. Le strategie sono pluggabili con un naming convention chiaro.

### Monitoring e observability eccellenti
L'integrazione Prometheus è capillare con metriche per ogni aspetto del sistema: PnL, drawdown, latenza decisionale, ordini aperti per broker, VaR/CVaR. Le Grafana dashboards sono pre-provisionate. L'audit trail con HMAC signing è un livello di compliance serio.

### Risk management multi-livello
Circuit breaker, daily loss limit, exposure caps per venue/settore, VaR limits, liquidity haircuts, vol targeting, cooldown, kill switch con interlock code. Sono molti layer di protezione, appropriati per un sistema live.

### GPU fallback graceful
Il pattern di fallback CUDA → CPU con `disable_gpu_until_restart()` è ben implementato, con retry loop e aggiornamento del config at runtime. Gestisce cleanly gli OOM errors sia di TensorFlow che di PyTorch.

### Docker-compose production-ready
Servizi separati per trader, learner, API, market-cache, healthwatch, daily-report. Autoheal configurato. Health checks su ogni servizio. GPU isolation tra trader (GPU) e ollama (CPU-only).

### Checkpoint e recovery
Il sistema può riprendere da dove si era fermato dopo un restart, salvando dynamic symbols, equity state, disabled strategies e trade timestamps.

---

## Suggerimenti aggiuntivi

### ~~Config come typed objects~~ ✅ Risolto
Sezioni critiche convertite in dataclass tipizzati:
- `RiskConfig` (`app/risk/config.py`)
- `StrategyConfig` (`app/agents/strategy_config.py`)
- `ExecutionConfig` (`app/execution/config.py`)

### ~~Logging strutturato~~ ✅ Risolto
Implementato `StructuredLogger` (`app/utils/structured_log.py`) con output JSON per eventi di trading e risk.

### ~~Secret management~~ ✅ Risolto
Campi `pretrain_alpaca_api_key` e `pretrain_alpaca_api_secret` in `RLOrchestratorConfig` marcati con `repr=False` per evitare esposizione in log/tracebacks. Verificato che i checkpoint non serializzano questo config.

### ~~`_build_strategy` fallback silenzioso~~ ✅ Risolto
Strategia sconosciuta ora restituisce `None` con `logging.warning()` invece di un fallback silenzioso a `IntradayMomentumStrategy`. Il caller già filtra i `None`.

---

## Riepilogo priorità

| Priorità | Issue | Stato |
|----------|-------|-------|
| 🔴 P0 | Refactor `trader.py` — estrarre classi | ✅ Completato |
| 🔴 P0 | Fix concorrenza su `market_state` / lock pattern | Aperto |
| 🔴 P0 | Exception handling specifico nei path di risk | Aperto |
| 🟡 P1 | Test per `_size_order`, `_combine_signals`, `run_once` | Aperto |
| 🟡 P1 | Validazione `MarketState` tipizzata | Aperto |
| 🟡 P1 | Eliminare codice duplicato | ✅ Parziale |
| 🟡 P1 | Migrare da `datetime.utcnow()` | ✅ Completato |
| 🟢 P2 | Config as typed objects | ✅ Completato |
| 🟢 P2 | Structured logging | ✅ Completato |
