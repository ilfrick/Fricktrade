# Fricktrade 3.0 — Code Review

**Data:** 7 febbraio 2026
**Scope:** Intero codebase (110 file Python, 24.306 LOC app + 2.224 LOC test)
**Stato:** Tutti i 19 issue risolti — v3.0 baseline

---

## Panoramica Architetturale

Fricktrade 3.0 è un sistema di trading automatizzato multi-broker con orchestrazione RL, composto da 15 servizi Docker (incluso docker-socket-proxy). L'architettura è organizzata in layer chiari: `agents/` (decisionale), `brokers/` (astrazione broker), `execution/` (ordini), `risk/` (gestione rischio), `learning/` (RL/HMM/ensemble), `portfolio/` (ottimizzazione), `monitoring/` (Prometheus/Grafana/audit), `data/` (market cache, AI filter, news).

Il progetto rispetto alla versione precedentemente rivista mostra miglioramenti strutturali significativi: `trader.py` è sceso da ~3.584 a ~2.350 righe grazie all'estrazione di `SymbolManager`, `PerformanceTracker`, `AccountMetricsUpdater`, `OpenOrderManager`, `StrategyConfig`, `ExecutionConfig`, `BrokerState`, `_build_rl_strategy`, `_calc_exposure_metrics`, e `extract_equity_cash` (in `app/utils/account.py`). Dead code rimosso: `pipeline.py`, `market_state.py`. I test sono 123 (13 skip senza tensorflow/prometheus), con nuovi test per orchestrator, flush responses, broker router, e order queue.

---

## Issue Risolti

### P0 — Safety (3 issue)

| # | Issue | Fix | File |
|---|-------|-----|------|
| 1 | `_regime_returns_buffer` mutato senza lock da 4 thread | Regime detection spostata dentro `with self._lock` insieme a `_update_orchestrator` | `trader.py` |
| 2 | `_enrich_market_state` legge stato condiviso senza lock | Snapshot `_news_cache` e `_open_order_mgr.cache` sotto lock prima di `ThreadPoolExecutor`; passati come parametri | `trader.py` |
| 3 | `except Exception` nei path finanziari | Narrowed a `(AttributeError, TypeError, ValueError, KeyError)` in `_flush_order_responses`; `_portfolio_position_scale` fallback cambiato da `1.0` a `0.0` | `trader.py` |

### P1 — Important (10 issue)

| # | Issue | Fix | File |
|---|-------|-----|------|
| 4 | `MarketState` dead code, nessuna validazione | Aggiunta validazione `last_price`/`prices` in `_process_single_symbol`; eliminati `market_state.py` e `test_market_state.py` | `trader.py` |
| 5 | `_enrich` / `_recalculate` duplicazione | Estratto `_calc_exposure_metrics()` come static method, chiamato da entrambi | `trader.py` |
| 6 | Equity extraction triplicata | Creato `app/utils/account.py` con `extract_equity_cash()`; usato in `trader.py`, `account_metrics.py`, `brokers/router.py` | 4 file |
| 7 | 39× `datetime.utcnow()` deprecato | Migrati tutti a `datetime.now(timezone.utc)` / `pd.Timestamp.now(tz="UTC")`; guard `.replace(tzinfo=timezone.utc)` per datetime naive da JSON; fix anche `utcfromtimestamp` | 17 file + 1 test |
| 8 | Test coverage insufficiente | Aggiunti `test_orchestrator_select.py`, `test_flush_responses.py`, `test_broker_router.py`, `test_order_queue_ext.py`; totale 123 test | 4 nuovi file test |
| 9 | `_build_strategy` duplicazione RL | Estratto `_build_rl_strategy(cls, name, extra_kwargs)` con device-fallback loop + OOM handling condiviso | `trader.py` |
| 10 | RiskManager reset con `date.today()` locale | Aggiunto parametro `tz` a `RiskManager.__init__`; `_maybe_reset_daily` usa `datetime.now(self._tz).date()`; default `US/Eastern` da config | `risk/manager.py`, `trader.py` |
| 11 | ThreadPoolExecutor ricreato ogni ciclo | `self._symbol_executor = ThreadPoolExecutor(max_workers=4)` persistente in `__init__` | `trader.py` |
| 12 | OrderQueue FIFO non stabile | Aggiunto `_seq` counter monotono a `OrderRequest`; `__lt__` usa `(earliest_at, _seq)` | `order_queue.py` |
| 13 | `_retry_notional_used` mai resettato | Aggiunto `_maybe_reset_retry_budget()` con reset giornaliero | `order_queue.py` |

### P2 — Cleanup (6 issue)

| # | Issue | Fix | File |
|---|-------|-----|------|
| 14 | Config dict non tipizzato | Documentato come tech debt futuro; `RiskConfig` dataclass già esiste | — |
| 15 | Cardinalità Prometheus esplosiva | `ORDER_REJECTS` labels ridotte da `[broker, symbol, side, code, reason]` a `[broker, side, reason]` | `metrics.py`, `order_queue.py` |
| 16 | DecisionPipeline pass-through | Rimosso; `run_once` chiamato direttamente; eliminato `pipeline.py` | `trader.py` |
| 17 | Import numpy nel hot path | Spostato `import numpy as np` a livello modulo | `trader.py` |
| 18 | Grafana credenziali default | `GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}` | `docker-compose.yml` |
| 19 | Docker socket in healthwatch | Aggiunto `docker-socket-proxy` (tecnativa) con `POST=1`, `CONTAINERS=1`; healthwatch e autoheal puntano al proxy via `DOCKER_HOST` | `docker-compose.yml`, `healthwatch.py` |

---

## Punti di Forza

**Architettura migliorata.** L'estrazione di `SymbolManager`, `PerformanceTracker`, `AccountMetricsUpdater`, `OpenOrderManager`, `StrategyConfig`, `ExecutionConfig`, e `BrokerState` ha ridotto `trader.py` del 35%+. La separazione delle responsabilità è chiara.

**Broker ABC + BrokerRouter.** Pattern pulito e estensibile. Aggiungere un nuovo broker richiede solo 7 metodi astratti. Il router gestisce account aggregation, position merging, e fallback automatico.

**Risk management multi-layer.** Circuit breaker, daily loss limit (con timezone configurabile), exposure caps per venue/sector, VaR/CVaR limits, liquidity haircuts, vol targeting, cooldown, kill switch con interlock code e profili adattivi.

**Monitoring production-ready.** Prometheus metrics con cardinalità controllata. Grafana dashboards pre-provisioned. Audit trail con HMAC signing. Docker socket isolato via proxy.

**GPU fallback robusto.** Graceful CUDA → CPU con `disable_gpu_until_restart()`. Gestisce OOM TensorFlow e PyTorch. Persiste stato su disco.

**Concurrency safety.** Regime detection, news cache, e open orders snapshot tutti sotto lock. ThreadPoolExecutor persistente. OrderQueue FIFO stabile con sequence counter.

**Zero `datetime.utcnow()`.** Tutto il codebase usa `datetime.now(timezone.utc)` con guard per datetime naive da JSON/file.

**Test migliorati.** 123 test (13 skip senza tensorflow/prometheus), con copertura per risk manager, trader sizing, signals, orchestrator, flush responses, broker router, e order queue.

---

## Tech Debt Residuo

| Area | Note |
|------|------|
| Config typing | Risk config è `RiskConfig` dataclass, ma molti sotto-config (`routing`, `algos`) sono ancora `dict` |
| Test coverage | `orchestrator.py` (1.552 LOC), `ai_filter.py` (1.094 LOC), `portfolio/` (~800 LOC) ancora poco testati |
| `except Exception` | 19 occorrenze rimangono in `trader.py` (ridotte da 22); ~169 nel resto del codebase (data/monitoring, basso rischio) |
| Broad catch in data paths | `app/data/`, `app/monitoring/` usano `except Exception` liberalmente — accettabile per resilienza in non-financial paths |
