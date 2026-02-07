# Fricktrade 3.0 — Code Review

**Data:** 7 febbraio 2026  
**Scope:** Intero codebase (110 file Python, 24.306 LOC app + 2.224 LOC test)

---

## Panoramica Architetturale

Fricktrade 3.0 è un sistema di trading automatizzato multi-broker con orchestrazione RL, composto da 14 servizi Docker. L'architettura è organizzata in layer chiari: `agents/` (decisionale), `brokers/` (astrazione broker), `execution/` (ordini), `risk/` (gestione rischio), `learning/` (RL/HMM/ensemble), `portfolio/` (ottimizzazione), `monitoring/` (Prometheus/Grafana/audit), `data/` (market cache, AI filter, news).

Il progetto rispetto alla versione precedentemente rivista mostra miglioramenti strutturali significativi: `trader.py` è sceso da ~3.584 a 2.434 righe grazie all'estrazione di `SymbolManager` (776), `PerformanceTracker` (287), `AccountMetricsUpdater` (248), `OpenOrderManager` (116), `StrategyConfig`, `ExecutionConfig`, e `BrokerState` in dataclass. Esiste un `MarketState` dataclass (market_state.py), `realized_volatility_pct` è stato estratto in `utils/volatility.py`, e `_build_strategy` non ritorna più un fallback silenzioso (ritorna `None` con warning). I test sono passati da ~1.143 a 2.224 LOC (163 funzioni test).

---

## Issue Critiche — P0

### 1. Concorrenza: `_regime_returns_buffer` e `_regime_state` senza lock

`_detect_regime()` viene chiamato nella sezione UNLOCKED di `run_once()` (linea ~873), ma muta stato condiviso:

```python
# UNLOCKED — chiamato da thread paralleli via ThreadPoolExecutor(max_workers=4)
self._regime_returns_buffer.append(ret)                          # L684
self._regime_returns_buffer = self._regime_returns_buffer[-N:]   # L687
self._regime_state = self._regime_hmm.get_state(returns)         # L693
```

Con `_run_symbol_batch` che lancia 4 thread paralleli (L1715), ogni thread chiama `_process_single_symbol` → `run_once` → `_detect_regime`. Due thread possono contemporaneamente: appendere al buffer, troncarlo, e leggere una versione inconsistente del buffer per l'HMM. La riassegnazione `self._regime_returns_buffer = ...` non è atomica e un thread potrebbe leggere un buffer parzialmente troncato.

**Impatto:** Crash silenzioso dell'HMM con array corrotto, o regime detection basata su dati inconsistenti che influenza le decisioni di trading.

**Fix:** O spostare `_detect_regime` dentro il blocco `with self._lock` (è un calcolo veloce), oppure usare un `threading.Lock` dedicato per il regime buffer. La soluzione più semplice:

```python
with self._lock:
    regime_state = self._detect_regime(market_state)
```

### 2. Concorrenza: `_enrich_market_state` fuori dal lock legge stato condiviso

In `_process_single_symbol` (L1670), `_enrich_market_state` è chiamato SENZA lock:

```python
self._enrich_market_state(market_state, portfolio, sym)  # L1670 — NO LOCK
```

Ma all'interno legge:

```python
market_state["catalyst"] = self._news_cache.get(symbol, False)   # dict condiviso
market_state["open_orders"] = self._open_order_mgr.cache         # list condivisa
```

`_news_cache` è riassegnato in `_refresh_news_cache` (L2114: `self._news_cache = result`), e `_open_order_mgr.cache` è aggiornato da `refresh()` nel main loop. Mentre la riassegnazione di un dict in Python è atomica a livello GIL, la lettura di `self._open_order_mgr.cache` durante un aggiornamento potrebbe dare risultati parziali se il cache è una lista mutata in-place.

**Impatto:** Basso ma non zero — potrebbe leggere ordini aperti stale o parzialmente aggiornati, causando un ordine duplicato o un mancato skip.

**Fix:** Leggere `_news_cache` e `_open_order_mgr.cache` sotto lock, o fare snapshot immutabili (tuple/frozenset) prima dell'uso.

### 3. 191 `except Exception` — silenziano bug reali nei path critici

Il conteggio è sceso da 195 a 191, ma il pattern rimane invariato nei percorsi finanziari critici:

```python
# app/agents/trader.py L1580 — _flush_order_responses
except Exception as exc:
    logging.warning("Orchestrator order feedback failed: %s", exc)

# app/agents/trader.py L1591
except Exception as exc:
    logging.warning("Live reward update failed: %s", exc)
```

`_flush_order_responses` è il ciclo che processa le risposte degli ordini eseguiti e aggiorna l'orchestrator RL. Un `KeyError` o `TypeError` qui viene swallowed, il che significa che l'orchestrator non riceve feedback su ordini reali e le sue decisioni future degradano silenziosamente.

**Pattern peggiore:**

```python
# _portfolio_position_scale — determina quanto capitale allocare
except (ValueError, TypeError, KeyError) as exc:
    logging.debug("Portfolio position scale failed: %s", exc)  # DEBUG, non WARNING
    return 1.0  # Fallback a scala piena — rischia overallocation
```

Se il calcolo della posizione fallisce, il sistema scala a 1.0 (nessuna riduzione), potenzialmente allocando il doppio del desiderato.

**Fix prioritario:** Nei path di sizing (`_size_order`, `_portfolio_position_scale`), risk check (`can_open_trade`, `check_exposure_caps`), e order feedback (`_flush_order_responses`): usare eccezioni specifiche, far propagare `AttributeError`/`TypeError`, e per i fallback di sizing tornare a `0` (skip) non a `1.0` (full allocation).

---

## Issue Importanti — P1

### 4. `MarketState` dataclass esiste ma non è usata

`app/agents/market_state.py` definisce un dataclass `MarketState` con 30+ campi tipizzati, `from_dict()`, e `to_dict()`. Ma non è importato né usato da nessuna parte nel codebase:

```bash
$ grep -rn "MarketState" app/ --include="*.py"
app/agents/market_state.py:8:class MarketState:   # solo qui
```

`market_state` rimane un `dict` non tipizzato passato attraverso l'intero pipeline. Ogni accesso è via `.get()` senza validazione:

```python
last_price = market_state.get("last_price")     # Potrebbe essere None, 0, NaN
qty = int(allowed_value // last_price)           # ZeroDivisionError se 0, crash se None
```

**Impatto:** Il dataclass è dead code. Le protezioni contro dati invalidi (prezzo zero, volume negativo, NaN da feed corrotti) non esistono.

**Fix:** Integrare `MarketState.from_dict()` nel punto d'ingresso (dove `market_data_provider(sym)` ritorna un dict), validare `last_price > 0` e `len(prices) > 0` come precondizioni. Non serve convertire tutto in una volta — basta validare all'ingresso.

### 5. Duplicazione `_enrich_market_state` / `_recalculate_exposure`

`_recalculate_exposure` (L2176-2191) è identico byte-per-byte alle prime 15 righe di `_enrich_market_state` (L2142-2156). Entrambi calcolano `exposure_pct`, `short_exposure_pct`, e `leverage` con la stessa formula.

```python
# _enrich_market_state (L2142) e _recalculate_exposure (L2176):
# IDENTICO per le prime 15 righe:
equity = float(portfolio.get("equity", 0.0) or 0.0)
# ... stessa logica ...
market_state["leverage"] = (gross_exposure / equity) if equity else 1.0
```

`_recalculate_exposure` è chiamato dentro `run_once` (L916) dopo un potenziale cambio di broker, e `_enrich_market_state` da `_process_single_symbol` (L1670).

**Fix:** Estrarre una funzione `_calc_exposure(market_state, portfolio, symbol) -> dict` e chiamarla da entrambi i punti.

### 6. Duplicazione dell'estrazione equity/cash

L'estrazione di equity/cash da account Alpaca e IBKR è implementata in 3 posti:

- `trader.py:_get_portfolio_snapshot()` (L2204-2230) — gestisce `equity`, `NetLiquidation`, e multi-broker
- `account_metrics.py:update()` (L110-142) — stessa logica con chiavi identiche
- `brokers/router.py:_extract_equity_cash()` (L161) — stessa logica come funzione standalone

Ogni implementazione gestisce gli stessi casi (formato Alpaca vs IBKR) ma con sottili variazioni. Se un nuovo broker viene aggiunto, tre file devono essere aggiornati.

**Fix:** Consolidare in `brokers/config_utils.py` o `utils/portfolio.py` con una singola `extract_equity_cash(account: dict) -> tuple[float, float, float]`.

### 7. 35 occorrenze di `datetime.utcnow()` — deprecato in Python 3.12+

`trader.py` usa correttamente `datetime.now(timezone.utc)` ovunque (buon lavoro), ma altri 35 usi di `utcnow()` rimangono sparsi in moduli chiave:

- `app/data/ai_filter.py` (4 occorrenze) — usato per decidere quando retrainare il modello
- `app/data/market_cache_service.py` (2) — timestamp del cache
- `app/learning/registry.py` (4) — timestamp nel model registry
- `app/portfolio/rebalance.py` (2) — timing del rebalance
- `app/strategies/stat_arb_pairs.py` (1)
- `app/data/ingestion.py` (1)

**Impatto:** `utcnow()` ritorna un datetime naive (senza timezone). Confrontarlo con un datetime aware causa `TypeError` in Python 3.12+. Visto che Fricktrade opera cross-timezone (NYSE ET, BorsaItaliana CET, server UTC), confronti temporali fra moduli diversi sono fragili.

**Fix:** Grep-and-replace globale: `datetime.utcnow()` → `datetime.now(timezone.utc)`, `pd.Timestamp.utcnow()` → `pd.Timestamp.now(tz="UTC")`.

### 8. Test coverage ancora insufficiente per un sistema finanziario

I test sono migliorati (2.224 LOC, 163 funzioni, 30 file), ma i moduli più grandi e critici restano senza test dedicati:

| Modulo | LOC | Test | Note |
|--------|-----|------|------|
| `orchestrator.py` | 1.552 | 0 | Core decisionale RL |
| `ai_filter.py` | 1.094 | Solo mapping (95 LOC) | Filtra quali simboli tradare |
| `main.py` | 901 | 0 | Wiring e startup |
| `portfolio/optimizer.py` | 413 | 0 | Ottimizzazione portafoglio |
| `portfolio/risk_model.py` | 401 | 0 | Modello di rischio |
| `execution/tca.py` | 366 | 0 | Transaction Cost Analysis |
| `execution/smart_router.py` | 317 | 0 | Routing ordini |
| `learning/train_rl.py` | 314 | 0 | Training RL |
| `brokers/alpaca.py` | 169 | 0 | Broker reale |
| `brokers/ibkr.py` | 168 | 0 | Broker reale |
| `brokers/router.py` | 171 | 0 | Multi-broker routing |

**Totale non testato: ~5.866 LOC** (24% del codebase), concentrato nei moduli che toccano soldi reali.

**Positivo:** `test_trader_sizing.py` (158 LOC) e `test_trader_signals.py` (144 LOC) esistono ora, e il `RiskManager` ha 201 LOC di test. Queste sono le aree più critiche e coprono i path principali.

**Fix prioritario:** Test per `orchestrator.select()` (decide quale strategia pesa di più), `_flush_order_responses()` (feedback loop RL), e integration test per `BrokerRouter.get_account()` con mock di broker multipli.

### 9. `_build_strategy` duplicazione RL policy / RL policy fees

I blocchi per `rl_policy` (L274-351) e `rl_policy_fees` (L352-398) sono ~95% identici. Entrambi fanno: select model path → loop over devices → try build → handle OOM → fallback CPU → handle errors. L'unica differenza è la classe istanziata e 2 parametri extra.

```python
# rl_policy: ~77 righe
if name == "rl_policy":
    # ... device loop, OOM handling, fallback ...
    return RLPolicyStrategy(model_path, ...)

# rl_policy_fees: ~47 righe — stessa struttura
if name == "rl_policy_fees":
    # ... device loop, OOM handling, fallback ...
    return FeeAwareRLPolicyStrategy(model_path, ...)
```

**Fix:** Estrarre `_build_rl_strategy(cls, extra_kwargs)` che gestisce il device loop e OOM una volta sola.

### 10. `RiskManager._maybe_reset_daily()` usa `date.today()` locale

```python
def _maybe_reset_daily(self) -> None:
    today = date.today()                    # ← timezone del server
    if self._last_reset_date != today:
        self._daily_loss = 0.0
```

`date.today()` ritorna la data nel fuso orario locale del server (tipicamente UTC in Docker). Se il server è UTC e si tradano azioni NYSE (ET = UTC−5), la daily loss si resetta alle 19:00 ET (mezzanotte UTC) — durante la sessione after-hours. Il loss limit giornaliero perde efficacia nelle ultime 5 ore di trading.

**Impatto:** Il sistema potrebbe accumulare perdite oltre il limite giornaliero perché il contatore si azzera prima della fine della sessione di trading effettiva.

**Fix:** Accettare un `timezone` configurabile nel `RiskManager.__init__`:

```python
def _maybe_reset_daily(self) -> None:
    today = datetime.now(self._tz).date()   # tz dalla config per venue
```

### 11. `ThreadPoolExecutor` ricreato ad ogni ciclo di trading

```python
# L1715 — dentro _run_symbol_batch, chiamato ogni ciclo
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(self._process_single_symbol, ...) for sym in symbols]
    wait(futures)
```

`_run_symbol_batch` è chiamato per ogni batch di simboli ad ogni iterazione del main loop (L1956). Con `with`, l'executor viene creato e distrutto ad ogni invocazione: allocazione del thread pool, startup dei thread, e shutdown con join. Su un universo di 200 simboli con batch da 50, sono 4 creazioni/distruzioni per ciclo, ogni 30-60 secondi.

Per confronto, `_news_executor` (L165) è giustamente persistente.

**Impatto:** Latenza non deterministica nel hot path. In media ~1-5ms per creazione/shutdown, ma sotto carico di sistema il thread startup può prendere decine di ms, aggiungendo jitter alla decisione di trading.

**Fix:** Creare l'executor in `__init__` come `_news_executor` e riusarlo:

```python
self._symbol_executor = ThreadPoolExecutor(max_workers=4)
```

### 12. OrderQueue: FIFO non stabile a parità di timestamp

```python
def __lt__(self, other: "OrderRequest") -> bool:
    return self.earliest_at < other.earliest_at    # unico criterio

def __eq__(self, other: object) -> bool:
    return self.earliest_at == other.earliest_at
```

`heapq` non è stabile: se due ordini hanno lo stesso `earliest_at` (es. due ordini market creati nello stesso microsecondo), l'ordine di estrazione è indefinito. In Python, `heapq` confronta il secondo elemento della tupla se il primo è uguale, ma qui non c'è tupla — confronta direttamente gli `OrderRequest`, e con `__eq__` basato solo su timestamp, due ordini "uguali" possono uscire in ordine arbitrario.

**Impatto:** Basso in condizioni normali (timestamp diversi), ma in burst di ordini (es. dopo un segnale forte su più simboli) l'ordine di esecuzione diventa non deterministico.

**Fix:** Aggiungere un sequence counter monotono come tie-breaker:

```python
_seq_counter: int = 0

def __lt__(self, other):
    return (self.earliest_at, self._seq) < (other.earliest_at, other._seq)
```

### 13. `_retry_notional_used` mai resettato — retry si esauriscono permanentemente

```python
# __init__ (L82)
self._retry_notional_used = 0.0

# _enqueue_retry (L309) — solo incremento, mai reset
self._retry_notional_used += notional

# _should_retry (L298) — controlla il budget
if max_notional > 0 and (self._retry_notional_used + notional) > max_notional:
    return False
```

In un run continuo (il sistema è sempre attivo durante le ore di mercato), `_retry_notional_used` cresce monotonicamente. Dopo un numero sufficiente di retry, il budget si esaurisce e nessun ordine può più essere ritentato, anche se il problema originale (es. rate limit temporaneo del broker) è risolto.

**Impatto:** Dopo giorni/settimane di operatività, i retry sono silenziosamente disabilitati. Ordini che falliscono per errori transitori vengono persi.

**Fix:** Reset giornaliero (o per sessione) del budget, analogamente al `_maybe_reset_daily` del `RiskManager`:

```python
def _maybe_reset_retry_budget(self):
    today = datetime.now(timezone.utc).date()
    if self._last_retry_reset != today:
        self._retry_notional_used = 0.0
        self._last_retry_reset = today
```

---

## Issue Minori — P2

### 14. Config ancora prevalentemente `dict` non tipizzato

`StrategyConfig` e `ExecutionConfig` sono un buon inizio, ma i loro campi interni sono ancora `dict`:

```python
@dataclass
class ExecutionConfig:
    routing: dict = field(default_factory=dict)   # Nessun tipo sui sotto-campi
    algos: dict = field(default_factory=dict)      # Typo in "algo_nme" è silenzioso
```

Il config `risk` (il più critico) è ancora acceduto via `self.cfg["risk"]["max_position_size_pct"]` con key stringa. Un typo come `max_position_sizee_pct` ritorna `None` e il calcolo di sizing produce risultati errati.

**Fix:** Creare `@dataclass RiskConfig` con validazione, almeno per i campi che determinano quanti soldi rischiare: `max_position_size_pct`, `max_short_exposure_pct`, `max_daily_loss_pct`, `circuit_breaker_drawdown_pct`.

### 15. Cardinalità Prometheus potenzialmente esplosiva

`ORDER_REJECTS` ha 5 label: `broker × symbol × side × code × reason`. Con 200 simboli, 2 broker, 2 side, e 10 possibili codici/reason, si generano fino a ~8.000 time series per un singolo Counter. `SKIPPED_ORDERS_BY_BROKER` ha 4 label con dinamica simile. `ORCHESTRATOR_STRATEGY_ACTIVE` ha `symbol × strategy`.

Prometheus alloca memoria per ogni combinazione unica di label. Con un universo di simboli dinamico (l'AI filter può aggiungere/rimuovere simboli), le time series crescono monotonicamente perché Prometheus non le dealloca.

**Impatto:** Consumo di memoria crescente in Prometheus. Con un universo ampio e rotazione frequente di simboli, si possono superare i limiti di Prometheus in settimane.

**Fix:** Per le metriche ad alta cardinalità, spostare il dettaglio nei log strutturati e usare metriche aggregate:

```python
# Invece di: ORDER_REJECTS.labels(broker=..., symbol=..., side=..., code=..., reason=...).inc()
# Usare: ORDER_REJECTS.labels(broker=..., reason=...).inc()  # solo dimensioni a bassa cardinalità
# E loggare il dettaglio: _slog.event("info", "order_rejected", symbol=..., code=..., ...)
```

### 16. `DecisionPipeline` è un pass-through

```python
class DecisionPipeline:
    def run(self, symbol, market_state):
        _ = DecisionContext(...)  # Crea un oggetto e lo scarta
        return self._agent.run_once(symbol, market_state)
```

Il `DecisionContext` creato non viene usato. L'intera pipeline è un wrapper che aggiunge un'allocazione inutile.

**Fix:** O rimuovere la pipeline e chiamare `run_once` direttamente, oppure usare effettivamente `DecisionContext` per accumulare informazioni di tracing.

### 17. Import di numpy dentro funzione hot-path

```python
def _detect_regime(self, market_state):
    # ... calcoli ...
    import numpy as np                    # L691 — import ad ogni chiamata
    returns = np.array(self._regime_returns_buffer)
```

`import numpy` è chiamato ad ogni invocazione di `_detect_regime`, cioè per ogni simbolo ad ogni ciclo. Python caches i moduli dopo il primo import, quindi non è catastrofico, ma è un code smell e una lookup nel `sys.modules` per ogni simbolo.

**Fix:** Spostare `import numpy as np` a livello di modulo o almeno a livello di `__init__`.

### 18. Docker: Grafana con credenziali default

```yaml
grafana:
  environment:
    - GF_SECURITY_ADMIN_USER=admin
    - GF_SECURITY_ADMIN_PASSWORD=admin
```

Anche se è un deploy interno, le credenziali hardcoded nel compose sono un rischio se il docker-compose viene committato o i port vengono esposti.

**Fix:** Usare variabili d'ambiente: `GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}`.

### 19. Healthwatch con Docker socket mount

```yaml
healthwatch:
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock
```

Il Docker socket dà accesso root all'host. Se il container healthwatch è compromesso, l'attaccante controlla la macchina.

**Fix:** Valutare se healthwatch ha realmente bisogno del socket Docker, oppure usare un proxy (docker-socket-proxy) con permessi limitati.

---

## Punti di Forza

**Architettura migliorata rispetto alla versione precedente.** L'estrazione di `SymbolManager`, `PerformanceTracker`, `AccountMetricsUpdater`, `OpenOrderManager`, `StrategyConfig`, `ExecutionConfig`, e `BrokerState` ha ridotto `trader.py` del 32% (da 3.584 a 2.434 righe). La separazione delle responsabilità è ora molto più chiara.

**Broker ABC + BrokerRouter.** Il pattern è pulito e estensibile. Aggiungere un nuovo broker richiede solo implementare 7 metodi astratti. Il router gestisce account aggregation, position merging con weighted average entry, e fallback automatico.

**Risk management multi-layer.** Circuit breaker, daily loss limit, exposure caps per venue/sector, VaR/CVaR limits, liquidity haircuts, vol targeting, cooldown, kill switch con interlock code e profili adattivi. Ogni layer ha una funzione dedicata nel `RiskManager`.

**Monitoring production-ready.** Prometheus metrics granulari (PnL, drawdown, decision latency, position value per broker, signal metrics). Grafana dashboards pre-provisioned. Audit trail con HMAC signing. Compliance logger con formati multipli.

**GPU fallback robusto.** Graceful CUDA → CPU con `disable_gpu_until_restart()`. Gestisce OOM sia TensorFlow che PyTorch. Persiste lo stato su disco per sopravvivere ai restart.

**Structured logging introdotto.** `StructuredLogger` emette JSON per eventi critici (trade_executed, risk_blocked, kill_switch_liquidation). Buona base per query in produzione.

**Checkpoint/recovery.** Il sistema salva e ripristina: simboli dinamici, equity state, strategie disabilitate, timestamp dell'ultimo trade, news cache, venue mapping. La migrazione da checkpoint legacy (single-broker) a multi-broker è gestita con backward compat.

**Test significativamente migliorati.** Da ~1.143 a 2.224 LOC. In particolare `test_risk_manager.py` (201 LOC), `test_trader_sizing.py` (158 LOC), e `test_trader_signals.py` (144 LOC) coprono ora i path finanziari più critici.

**`_build_strategy` fixato.** Non ritorna più un `IntradayMomentumStrategy` di fallback per strategie sconosciute — ritorna `None` con warning esplicito. Questo elimina il rischio di eseguire una strategia diversa da quella richiesta.

**Deep copy prima dei thread.** `copy.deepcopy(market_state)` prima di passare al pipeline (L1683) previene la mutazione cross-thread del market state. Buon fix rispetto alla versione precedente.

---

## Riepilogo Priorità

| # | Priorità | Issue | Rischio |
|---|----------|-------|---------|
| 1 | **P0** | `_regime_returns_buffer` mutato senza lock da 4 thread | Race condition → regime detection corrotta |
| 2 | **P0** | `_enrich_market_state` legge stato condiviso senza lock | Ordini duplicati o skip errati |
| 3 | **P0** | 191 `except Exception` nei path finanziari | Bug mascherati → overallocation silenziosa |
| 4 | P1 | `MarketState` dataclass non usato, nessuna validazione dati | Crash su dati invalidi (price=0, NaN) |
| 5 | P1 | `_enrich` / `_recalculate` duplicazione | Manutenzione fragile |
| 6 | P1 | Equity extraction triplicata | Inconsistenza fra broker |
| 7 | P1 | 35× `datetime.utcnow()` deprecato | TypeError in Python 3.12+ |
| 8 | P1 | 5.866 LOC non testati (orchestrator, brokers, portfolio) | Regressioni non detectate |
| 9 | P1 | `_build_strategy` duplicazione RL/RL-fees | 120+ righe duplicate |
| 10 | P1 | RiskManager reset con `date.today()` locale | Daily loss limit inefficace cross-timezone |
| 11 | P1 | ThreadPoolExecutor ricreato ogni ciclo | Latenza jitter nel hot path |
| 12 | P1 | OrderQueue FIFO non stabile | Ordine esecuzione non deterministico in burst |
| 13 | P1 | `_retry_notional_used` mai resettato | Retry silenziosamente disabilitati dopo giorni |
| 14 | P2 | Risk config non tipizzato | Typo = sizing errato silenzioso |
| 15 | P2 | Cardinalità Prometheus esplosiva | Memoria crescente con simboli dinamici |
| 16 | P2 | DecisionPipeline pass-through | Dead code |
| 17 | P2 | Import numpy nel hot path | Micro-inefficienza |
| 18 | P2 | Grafana credenziali default | Security hygiene |
| 19 | P2 | Docker socket in healthwatch | Escalation risk |
