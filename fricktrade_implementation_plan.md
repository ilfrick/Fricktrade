# Fricktrade 3.0 — Piano di Implementazione Completo

**Data**: 28 Febbraio 2026
**Obiettivo**: Rendere Fricktrade profittevole attraverso espansione crypto, strategie migliorate, execution ottimizzata e attivazione dei componenti esistenti.

**Convenzioni**: ogni task ha una stima di effort (ore), priorità (P0-P3), e dipendenze.

---

## FASE 0 — FONDAMENTA (Settimana 1-2)

> Senza queste basi, ogni miglioramento successivo è inutile perché non puoi misurarne l'impatto.

### 0.1 Walk-Forward Backtesting Rigoroso

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P0 |
| **Effort** | 12-16 ore |
| **File coinvolti** | `app/backtest/`, `scripts/benchmark_runner.py` |
| **Dipendenze** | Nessuna |

**Cosa fare:**

1. Modificare `benchmark_runner.py` per supportare purged walk-forward:
   - Train window: {t-365d, t-30d}
   - Embargo gap: 5 giorni (evita leakage)
   - Test window: {t-25d, t}
   - Rolling: avanza di 30 giorni e ripeti
2. Aggiungere costi di transazione realistici nel backtest engine:
   - Equities: 5 bps per trade (spread + slippage)
   - Crypto: 8 bps per trade (spread + fee Alpaca crypto)
3. Aggiungere benchmark comparison automatico:
   - Ogni strategia vs buy-and-hold SPY (equities) o buy-and-hold BTC (crypto)
   - Calcolare Sharpe, Sortino, max drawdown, profit factor, win rate
4. Generare report per ogni walk-forward fold con metriche aggregate

**Criterio di successo**: ogni strategia che non batte il benchmark dopo costi viene disabilitata o riscritta.

---

### 0.2 Metriche di Performance Standardizzate

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P0 |
| **Effort** | 4-6 ore |
| **File coinvolti** | `app/monitoring/metrics.py`, `grafana/` |
| **Dipendenze** | Nessuna |

**Cosa fare:**

1. Aggiungere Prometheus gauges per:
   - `strategy_sharpe_ratio{strategy, asset_class}` — rolling 30 giorni
   - `strategy_profit_factor{strategy, asset_class}`
   - `strategy_win_rate{strategy, asset_class}`
   - `trade_cost_bps{broker, asset_class}` — costo medio effettivo per trade
2. Creare dashboard Grafana "Strategy P&L Scorecard" con comparazione side-by-side
3. Alert quando Sharpe < 0 per 5 sessioni consecutive → auto-disable strategia

---

## FASE 1 — INTEGRAZIONE CRYPTO SU ALPACA (Settimana 2-4)

> Questa è la singola azione con il maggior impatto sulla profittabilità. Il broker adapter esiste già; servono adattamenti per gestire le differenze tra equities e crypto.

### 1.1 Estensione Config per Crypto

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P0 |
| **Effort** | 3-4 ore |
| **File coinvolti** | `config/config.yaml`, `app/brokers/config_utils.py` |
| **Dipendenze** | Nessuna |

**Modifiche a `config.yaml`:**

```yaml
market:
  venue: Multi
  trading_venues:
    - NYSE
    - Nasdaq
    - Crypto          # ← NUOVO

  venues:
    # ... NYSE, Nasdaq, BorsaItaliana esistenti ...
    - name: Crypto
      timezone: UTC
      trading_hours:
        open: "00:00"
        close: "23:59"   # 24/7
      holidays: []        # Mai chiuso

  asset_classes:           # ← NUOVA SEZIONE
    equities:
      venues: [NYSE, Nasdaq]
      min_price: 2.0
      order_types: [market, limit, stop, stop_limit]
      time_in_force: [day, gtc, ioc]
      shortable: true
      marginable: true
    crypto:
      venues: [Crypto]
      min_price: 0.0
      order_types: [market, limit, stop_limit]
      time_in_force: [gtc, ioc]    # NO 'day' per crypto
      shortable: false
      marginable: false

data:
  symbols:
    - AAPL
    - MSFT
    # ...equities...
  crypto_symbols:           # ← NUOVA SEZIONE
    - BTC/USD
    - ETH/USD
    - SOL/USD
    - AVAX/USD
    - LINK/USD
    - DOGE/USD
    - MATIC/USD
    - DOT/USD
```

**Modifiche a `config_utils.py`:**

- Aggiungere helper `is_crypto_symbol(symbol: str) -> bool` (controlla se contiene `/`)
- Aggiungere `get_asset_class(symbol: str) -> str` che ritorna `"crypto"` o `"equities"`
- Modificare `get_venue_for_symbol()` per ritornare `"Crypto"` per crypto symbols

---

### 1.2 Adattamento Alpaca Broker Adapter

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P0 |
| **Effort** | 6-8 ore |
| **File coinvolti** | `app/brokers/alpaca.py`, `app/brokers/base.py` |
| **Dipendenze** | 1.1 |

**Modifiche a `app/brokers/alpaca.py`:**

1. **`place_order()`**: cambiare `time_in_force` default a `gtc` per crypto:
   ```python
   def place_order(self, symbol, side, qty, order_type="market", limit_price=None):
       is_crypto = "/" in symbol
       tif = "gtc" if is_crypto else "day"
       # ...resto invariato...
   ```

2. **`get_universe()`**: aggiungere query per asset crypto:
   ```python
   def get_crypto_assets(self):
       assets = self.api.list_assets(asset_class="crypto", status="active")
       return [a for a in assets if a.tradable]
   ```

3. **`get_bars()`**: usare il crypto data endpoint (stessa API, diverso symbol format):
   ```python
   # Per crypto: symbol = "BTC/USD" → Alpaca lo accetta direttamente
   # Il data feed per crypto è sempre "us" (non "iex" o "sip")
   ```

4. **`get_position()`**: gestire il caso crypto dove qty è frazionaria (es. 0.001 BTC)

5. **`close_position()`**: per crypto serve specificare qty esatta (no "close all" shortcut per crypto su Alpaca)

**Modifiche a `app/brokers/base.py`:**

Aggiungere al `Broker` base class:
```python
def get_asset_class(self, symbol: str) -> str:
    """Return 'crypto' or 'equities'."""
    return "crypto" if "/" in symbol else "equities"

def supports_short(self, symbol: str) -> bool:
    return self.get_asset_class(symbol) != "crypto"
```

---

### 1.3 Market Hours e Venue Gating per 24/7

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P0 |
| **Effort** | 4-5 ore |
| **File coinvolti** | `app/agents/trader.py`, `app/data/market_cache.py` |
| **Dipendenze** | 1.1 |

**Cosa fare:**

1. In `trader.py`, il check `_is_market_open()` deve ritornare `True` sempre per venue `Crypto`
2. Il trading loop deve gestire due cicli separati o un ciclo unificato con batching:
   - **Opzione consigliata**: ciclo unificato che processa tutti i simboli (equities + crypto) ma skippa equities fuori orario
3. In `market_cache.py`, il TTL per crypto può essere più aggressivo (2x interval vs 6x) perché il mercato è sempre aperto e non ci sono gap notturni
4. Il `healthwatch` market shutdown non deve fermare i container quando solo i mercati equity chiudono — deve controllare se ci sono asset class ancora attive

---

### 1.4 Healthwatch e Market Shutdown: Non Spegnere Mai il Trader

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P0 |
| **Effort** | 6-8 ore |
| **File coinvolti** | `app/agents/trader.py`, `scripts/run_tests_when_closed.py`, `config/config.yaml`, `docker-compose.yml`, healthwatch logic |
| **Dipendenze** | 1.1 |

**Il problema attuale:**

Il sistema `healthwatch.market_shutdown` controlla se i mercati equity (NYSE/Nasdaq) sono aperti. Quando chiudono (16:00 ET), il healthwatch:
1. Scrive `state: sleeping` in `/data/system_state.json`
2. Ferma i container nel `stop_services` list (e tutti quelli non in `keep_services`)
3. Il trader si spegne e resta spento fino al prossimo `start_before_minutes` (15 min) prima dell'open successivo

Con crypto attivo 24/7, il trader **non deve mai essere spento**. Ma serve comunque un modo per gestire il ciclo equity (certe operazioni, come il daily report equities o il backtest, devono continuare a eseguirsi a mercato equity chiuso).

**Soluzione: modalità "always-on" con equity-aware scheduling interno**

#### A. Modificare `config/config.yaml`

```yaml
healthwatch:
  market_shutdown:
    enabled: true
    mode: partial              # ← NUOVO: 'full' (attuale) | 'partial' | 'disabled'
    # 'partial' = non spegne il trader, ma segnala quando equity è chiuso
    # Il trader usa questo stato per decidere cosa fare internamente
    project_name: fricktrade
    check_interval_seconds: 60
    start_before_minutes: 15
    heartbeat_minutes: 15
    write_state: true
    state_path: /data/system_state.json
    keep_services:
      - healthwatch
      - autoheal
      - docker-socket-proxy
      - daily-report
      - prometheus
      - tests-when-closed
      - trader              # ← NUOVO: il trader resta SEMPRE attivo
      - api                 # ← NUOVO: l'API resta attiva per monitoring
      - market-cache        # ← NUOVO: la cache serve per crypto data
      - redis               # ← NUOVO: serve alla cache
      - learner             # ← NUOVO: il learner può allenare su dati crypto 24/7
      - grafana             # ← NUOVO: monitoring 24/7
      - alertmanager        # ← NUOVO: alert 24/7
    stop_services:
      - ollama              # Ollama può spegnersi fuori orario equity (consuma GPU)
      - calendar-updater    # Solo per equity holidays
```

In pratica con crypto attivo, quasi tutto deve restare acceso. L'unico servizio che ha senso spegnere è `ollama` per risparmiare risorse GPU quando non ci sono news equity da processare.

#### B. Modificare la logica di `system_state.json`

Attualmente lo state è binario: `running` o `sleeping`. Cambiare a:

```json
{
  "state": "running",
  "equity_market_open": false,
  "crypto_market_open": true,
  "active_asset_classes": ["crypto"],
  "next_equity_open": "2026-03-02T14:30:00Z",
  "last_updated": "2026-02-28T22:00:00Z"
}
```

Il healthwatch scrive lo stato di ogni asset class. Il trader legge questo file e decide quali simboli processare.

#### C. Modificare `app/agents/trader.py` — Trading Loop

Il trading loop attualmente ha questa logica (semplificata):

```python
# ATTUALE:
if not self._is_market_open():
    sleep(60)
    continue
# processa tutti i simboli
```

Deve diventare:

```python
# NUOVO:
ops_state = self._read_ops_state()
active_classes = set()

if ops_state.get("equity_market_open", False):
    active_classes.add("equities")

if ops_state.get("crypto_market_open", True):  # default True per crypto
    active_classes.add("crypto")

if not active_classes:
    # Nessun mercato aperto (non dovrebbe mai succedere con crypto)
    sleep(60)
    continue

# Filtra simboli per asset class attive
active_symbols = [
    s for s in all_symbols
    if get_asset_class(s) in active_classes
]
# Processa solo i simboli attivi
for symbol in active_symbols:
    self._process_symbol(symbol, market_state)
```

Questo approccio è il meno invasivo: il loop gira sempre, ma processa solo i simboli delle asset class con mercato aperto.

#### D. Modificare `scripts/run_tests_when_closed.py`

Attualmente questo script gira i test e i backtest quando il mercato è chiuso. Con crypto 24/7, "mercato chiuso" si riferisce solo a equities. Modificare:

```python
# ATTUALE:
if is_market_closed():
    run_tests()
    run_backtests()

# NUOVO:
if is_equity_market_closed():
    run_equity_tests()
    run_equity_backtests()
# I test crypto possono girare in qualsiasi momento in un thread separato,
# oppure schedulati a orari fissi (es. 04:00 UTC quando il volume crypto è minimo)
if is_low_volume_crypto_window():  # 00:00-04:00 UTC
    run_crypto_backtests()
```

#### E. Modificare il `daily-report` service

Il daily report attualmente gira dopo la chiusura equity (16:05 ET). Con crypto serve un report separato o un report unificato multi-asset:

```yaml
reports:
  daily_top_movers:
    # Report equities: gira dopo close NYSE (invariato)
    equity_report_time: "16:05 ET"
    # Report crypto: gira ogni 24h a mezzanotte UTC
    crypto_report_time: "00:05 UTC"
    # Oppure: report unificato ogni 24h
    unified: true
    unified_report_time: "00:05 UTC"
```

#### F. Docker Compose — Resource Management

Con il trader attivo 24/7, serve attenzione alle risorse:

```yaml
# docker-compose.yml
services:
  trader:
    # Aggiungere limiti di memoria per evitare memory leak su run 24/7
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 512M
    # Health check più frequente
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/metrics"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  market-cache:
    deploy:
      resources:
        limits:
          memory: 1G

  redis:
    # Redis deve stare sempre su per crypto data cache
    restart: always
    deploy:
      resources:
        limits:
          memory: 512M
```

#### G. Checkpointing più frequente

Con il trader che gira 24/7, il checkpoint diventa più critico (crash = perdita stato):

```yaml
checkpointing:
  interval_seconds: 30        # ← Da 60 a 30 per 24/7
  retention:
    max_age_hours: 168         # ← Da 72 a 168 (1 settimana di storia)
    max_files: 500             # ← Da 200 a 500
```

#### H. Log Rotation per 24/7

I log cresceranno molto di più con operatività continua:

```yaml
logging:
  max_bytes: 10000000          # ← Da 5MB a 10MB
  backup_count: 10             # ← Da 5 a 10
```

**Riepilogo delle modifiche per il 24/7:**

| Componente | Modifica | Effort |
|------------|----------|--------|
| `config.yaml` healthwatch | `mode: partial`, trader in `keep_services` | 30 min |
| `system_state.json` schema | Multi-asset-class state | 1-2 ore |
| Healthwatch logic | Scrivere stato per-asset-class | 2-3 ore |
| `trader.py` trading loop | Filtrare simboli per asset class attiva | 1-2 ore |
| `run_tests_when_closed.py` | Separare test equity vs crypto | 30 min |
| `daily-report` | Report crypto separato/unificato | 1 ora |
| `docker-compose.yml` | Resource limits, restart policy | 30 min |
| Checkpointing + logging | Parametri per 24/7 | 15 min |

---

### 1.5 Strategie Adattate per Crypto

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P0 |
| **Effort** | 10-14 ore |
| **File coinvolti** | `app/strategies/`, nuovi file |
| **Dipendenze** | 1.1, 1.2, 1.3, 1.4 |

#### A. Adattare `trend_following.py` per crypto

La trend following con MA crossover funziona decisamente meglio su crypto che su equities perché i trend crypto durano più a lungo e sono più pronunciati. Modifiche:

- Allargare le finestre: `fast_window: 20`, `slow_window: 60` (su barre 5min = 100-300 min)
- Abbassare il RSI overbought da 70 a 65 (crypto trending overshoota regolarmente)
- Aggiungere filtro di volatilità: non tradare quando la volatility realizzata 1h è sotto il 10° percentile (mercato morto, falsi segnali)

#### B. Nuovo: `crypto_momentum.py`

```python
class CryptoMomentumStrategy(Strategy):
    """Momentum a breve termine ottimizzato per crypto.

    Logica:
    1. Calcola il rendimento degli ultimi 5, 15, 60 minuti
    2. Se tutti e tre sono positivi E il volume è > 2x media → BUY
    3. Se tutti e tre sono negativi E il volume è > 2x media → SELL (exit)
    4. Confidence proporzionale alla concordanza degli indicatori

    Parametri crypto-specifici:
    - Soglia volume più alta (crypto ha spike di volume più violenti)
    - Trailing stop più largo (volatilità 3-5x equities)
    - No short (Alpaca crypto non supporta short)
    """
```

**Parametri config:**
```yaml
strategy:
  params:
    crypto_momentum:
      fast_window: 5          # 5 barre = 25 min
      medium_window: 15       # 15 barre = 75 min
      slow_window: 60         # 60 barre = 300 min (5 ore)
      volume_mult: 2.0
      min_return_pct: 0.3     # Soglia più alta che equities
      trailing_stop_pct: 2.0  # Più largo che equities (1.5%)
```

#### C. Nuovo: `crypto_mean_reversion.py`

```python
class CryptoMeanReversionStrategy(Strategy):
    """Mean reversion su crypto usando Bollinger Bands + RSI.

    Le crypto tendono a revertire dopo spike violenti causati da
    liquidazioni forzate. Questa strategia compra quando:
    1. Prezzo sotto la lower Bollinger Band (2 std, 20 periodi)
    2. RSI < 25
    3. Il drop è avvenuto in < 15 minuti (liquidation cascade signature)

    Exit: quando il prezzo torna alla media mobile (centro delle BB)
    Stop: 3% sotto l'entry (le crypto possono continuare a scendere)
    """
```

#### D. Adattare `top_movers_rf.py` per crypto

- Il training data deve includere i crypto top movers (serve raccogliere dati 1m per crypto)
- Le feature sono le stesse (prezzo, volume, VWAP) ma i parametri cambiano:
  - `rebound_target_pct: 3.0` (vs 2.0 per equities — crypto rimbalza di più)
  - `low_zone_tol_pct: 0.5` (vs 0.35)
  - `cutoff_minutes: 60` (vs 30 — crypto non ha un "open" definito, serve più warmup)

#### E. Disabilitare `stat_arb_pairs` per crypto

Non funziona senza short selling. Tenerla solo per equities.

---

### 1.6 Risk Management Crypto-Specific

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P1 |
| **Effort** | 4-6 ore |
| **File coinvolti** | `app/risk/manager.py`, `app/risk/config.py` |
| **Dipendenze** | 1.1 |

**Modifiche:**

```yaml
risk:
  crypto:                          # ← NUOVA SEZIONE
    max_daily_loss_pct: 5.0        # Più largo che equities (3%) per la volatilità
    max_position_size_pct: 15.0    # Max 15% in un singolo crypto asset
    max_crypto_exposure_pct: 40.0  # Max 40% del portafoglio in crypto totale
    hard_stop_pct: 3.0             # Stop loss più largo
    trailing_stop_pct: 5.0         # Trailing più largo
    take_profit_pct: 4.0           # Target più alto
    circuit_breaker_drawdown_pct: 8.0
    vol_targeting:
      target_vol_pct: 4.0          # 2x equities
```

In `manager.py`:
1. Aggiungere check `max_crypto_exposure_pct` che somma tutte le posizioni crypto
2. Scalare dinamicamente gli stop in base alla volatilità realizzata (ATR-based):
   ```python
   if is_crypto:
       effective_stop = max(config.hard_stop_pct, atr_pct * 1.5)
   ```
3. Aggiungere check per concentrazione BTC: se > 50% del crypto allocation è in BTC, warning

---

### 1.7 Data Collection e Training Crypto

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P1 |
| **Effort** | 6-8 ore |
| **File coinvolti** | `app/data/`, `app/reporting/`, `scripts/` |
| **Dipendenze** | 1.2 |

**Cosa fare:**

1. Aggiungere download di dati storici crypto da Alpaca (1m e 5m bars)
   - Alpaca fornisce dati crypto storici gratis via la stessa API dei bars
   - Periodo: almeno 6 mesi di storia per training
2. Estendere il daily top movers report per includere crypto:
   ```yaml
   reports:
     daily_top_movers:
       universe: alpaca_active
       crypto_universe: alpaca_crypto   # ← NUOVO
   ```
3. Aggiungere script `scripts/collect_crypto_training_data.py` che raccoglie 1m bars per tutti i crypto asset disponibili su Alpaca
4. Estendere `return_ranker_train.py` per includere crypto features

---

## FASE 2 — ATTIVAZIONE COMPONENTI ESISTENTI (Settimana 3-5)

> Codice già scritto e testato che è attualmente disabilitato o non collegato al flusso principale.

### 2.1 Wiring dello Smart Router

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P1 |
| **Effort** | 4-6 ore |
| **File coinvolti** | `app/agents/trader.py`, `app/execution/smart_router.py`, `app/execution/executor.py` |
| **Dipendenze** | Nessuna |

**Cosa fare:**

1. In `trader.py`, sostituire la chiamata diretta a `execution_engine.execute()` con `smart_router.route_order()`:
   ```python
   # PRIMA (attuale):
   order_id = self.execution_engine.execute(symbol, action, qty)

   # DOPO:
   order_id = self.smart_router.route_order(
       symbol=symbol,
       side=action,
       qty=qty,
       urgency=self._urgency_from_confidence(confidence),
       market_state=market_state,
   )
   ```
2. Implementare `_urgency_from_confidence()`:
   - confidence > 0.8 → urgency "high" → market order
   - confidence 0.5-0.8 → urgency "medium" → limit order al midpoint
   - confidence < 0.5 → urgency "low" → limit order al bid (buy) / ask (sell)
3. Attivare l'Almgren-Chriss trajectory per ordini > 5% dell'ADV
4. Attivare TCA (Transaction Cost Analysis) logging per misurare lo slippage effettivo

---

### 2.2 Attivazione Meta-Orchestrator LLM

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P1 |
| **Effort** | 2-3 ore |
| **File coinvolti** | `config/config.yaml`, `app/llm/meta_orchestrator.py` |
| **Dipendenze** | Nessuna (il codice è già scritto) |

**Modifiche config:**
```yaml
llm:
  meta_orchestrator:
    enabled: true              # ← Era false
    run_day: sunday
    auto_apply: false          # Tieni manuale all'inizio
    max_weight_change: 0.10    # Riduci da 0.15 a 0.10 per safety
    min_sessions_required: 5
```

**Azione aggiuntiva:** dopo 2 settimane di paper trading con crypto, esaminare i report generati dal meta-orchestrator e valutare se attivare `auto_apply: true`.

---

### 2.3 Attivazione Risk Interpreter LLM

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P1 |
| **Effort** | 1-2 ore |
| **File coinvolti** | `config/config.yaml` |
| **Dipendenze** | Nessuna |

```yaml
llm:
  risk_interpreter:
    enabled: true              # ← Era false
    min_severity: medium       # ← Da 'high' a 'medium'
    timeout_seconds: 15        # ← Da 10 a 15 per Claude
```

---

### 2.4 Attivazione LLM Symbol Filter Pre-Market

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P2 |
| **Effort** | 3-4 ore |
| **File coinvolti** | `config/config.yaml`, `app/llm/symbols_filter.py` |
| **Dipendenze** | 2.2 |

```yaml
llm:
  symbols_filter:
    enabled: true              # ← Era false
    run_time: pre_market
    select_count: 30           # ← Da 20 a 30
    fallback_to_rules: true
```

**Adattamento per crypto:** il symbol filter deve girare anche per crypto (che non ha "pre-market"). Aggiungere un trigger periodico ogni 4 ore per rinfrescare la crypto watchlist.

---

### 2.5 Wiring del Portfolio Optimizer

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P2 |
| **Effort** | 8-10 ore |
| **File coinvolti** | `app/agents/trader.py`, `app/portfolio/optimizer.py`, `app/portfolio/rebalance.py` |
| **Dipendenze** | 0.1 |

**Cosa fare:**

1. Dopo che tutte le strategie generano segnali per tutti i simboli, raccogliere i target weights in un dict `{symbol: target_weight}`
2. Chiamare `PortfolioOptimizer.optimize(method="risk_parity")` con i vincoli:
   ```python
   constraints = PortfolioConstraints(
       max_position_pct=0.15,
       max_sector_pct={"crypto": 0.40, "tech": 0.25},
       long_only=True,
       max_turnover_pct=0.30,  # Max 30% turnover per ciclo
   )
   ```
3. Confrontare target allocation con posizioni correnti
4. Generare ordini solo per il delta > soglia minima (evita micro-rebalancing)
5. Il `rebalance.py` ha già la logica di delta calculation — collegarla

**Impatto**: riduce drasticamente l'over-trading e migliora la diversificazione.

---

## FASE 3 — NUOVE STRATEGIE CON ALPHA DOCUMENTATO (Settimana 4-8)

### 3.1 Gap Reversal Strategy

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P1 |
| **Effort** | 8-10 ore |
| **File coinvolti** | Nuovo: `app/strategies/gap_reversal.py` |
| **Dipendenze** | 0.1, 0.2 |

```python
class GapReversalStrategy(Strategy):
    """
    I gap di apertura > 2% su equities US tendono a revertire
    nei primi 30-60 minuti. Sharpe storico: 1.0-1.5.

    Segnale BUY (gap down reversal):
    - Gap down > 2% rispetto al close precedente
    - Volume nei primi 5 min > 2x media
    - RSI(5 min) < 30
    → Buy. Target: 50% del gap. Stop: 0.5% sotto l'open.

    Segnale SELL/EXIT (gap up reversal):
    - Gap up > 2% rispetto al close precedente
    - Volume nei primi 5 min > 2x media
    - RSI(5 min) > 70
    → Sell se hai posizione. Target: 50% del gap.

    Finestra operativa: 9:35 - 10:30 ET (dopo i primi 5 min)
    Solo equities (non crypto — le crypto non hanno un 'open' definito)
    """
```

**Parametri config:**
```yaml
strategy:
  names:
    - gap_reversal        # ← NUOVO
  params:
    gap_reversal:
      min_gap_pct: 2.0
      target_fill_pct: 50.0
      stop_pct: 0.5
      volume_confirm_mult: 2.0
      max_trade_window_minutes: 60
      rsi_period: 5
      rsi_oversold: 30
      rsi_overbought: 70
```

---

### 3.2 Earnings Drift Strategy (PEAD)

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P2 |
| **Effort** | 12-16 ore |
| **File coinvolti** | Nuovo: `app/strategies/earnings_drift.py`, `app/data/earnings_calendar.py` |
| **Dipendenze** | 2.2 (LLM sentiment) |

```python
class EarningsDriftStrategy(Strategy):
    """
    Post-Earnings Announcement Drift (PEAD).

    Dopo un earnings surprise, il titolo continua a muoversi nella
    direzione della surprise per 30-60 giorni.

    Flusso:
    1. Pre-earnings: nessun segnale (non speculare sull'earnings)
    2. Post-earnings (giorno dopo):
       a. Scarica il transcript via API
       b. LLM (Claude) analizza: surprise vs consensus, guidance, tone
       c. Se surprise > 10% e LLM confidence > 0.7 → BUY
       d. Se surprise < -10% e LLM confidence > 0.7 → SELL/EXIT
    3. Hold per 20-40 giorni con trailing stop

    Integra con il NewsSentimentAnalyzer esistente.
    """
```

**Dati necessari:**
- Earnings calendar: gratuito via Alpaca corporate actions API
- Earnings consensus: Alpha Vantage (free tier) o scraping Earnings Whispers
- Earnings transcript: Alpaca news + LLM summary

---

### 3.3 LLM Macro Regime Detector

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P2 |
| **Effort** | 6-8 ore |
| **File coinvolti** | Nuovo: `app/llm/macro_regime.py` |
| **Dipendenze** | 2.2 |

```python
class MacroRegimeDetector:
    """
    Usa Claude per classificare il regime macro corrente.

    Ogni mattina (equities) e ogni 4 ore (crypto):
    1. Fetch dati macro: VIX, DXY, UST 10Y yield, oil, gold, SPX
    2. Fetch top 10 headlines da news feed
    3. Prompt Claude per classificare:
       - risk_on → boost momentum e trend following
       - risk_off → boost mean reversion, ridurre posizioni
       - rotation → focus su sector signals
       - range_bound → boost stat arb, ridurre trend
       - crisis → circuit breaker, cash only

    Output: regime weights che modificano l'orchestratore.
    """
```

**Integrazione con orchestratore:**
```python
# In trader.py, prima del ciclo di segnali:
regime = self.macro_detector.get_current_regime()
adjusted_weights = self.orchestrator.apply_regime_overlay(
    base_weights=config.strategy_weights,
    regime=regime,
)
```

---

## FASE 4 — EXECUTION OPTIMIZATION (Settimana 6-9)

### 4.1 Limit Orders di Default

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P1 |
| **Effort** | 4-6 ore |
| **File coinvolti** | `app/execution/executor.py`, `app/execution/smart_router.py` |
| **Dipendenze** | 2.1 |

**Cosa fare:**

1. Default a limit order al midpoint per tutti gli ordini (equities e crypto):
   ```python
   if urgency == "high":
       order_type = "market"
   else:
       mid = (bid + ask) / 2
       if side == "buy":
           limit_price = mid  # o bid + spread * 0.3 per crossing minimo
       else:
           limit_price = mid  # o ask - spread * 0.3
       order_type = "limit"
   ```
2. Aggiungere timeout per limit orders non fillati:
   - Equities: 60 secondi → cancel e re-submit come market se urgency > "low"
   - Crypto: 30 secondi (mercato più veloce)
3. Logging del fill rate e slippage per analisi TCA

**Impatto stimato**: 3-8 bps di saving per trade. Su 20 trade/giorno = 60-160 bps/giorno.

---

### 4.2 Time-of-Day Filter

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P2 |
| **Effort** | 2-3 ore |
| **File coinvolti** | `app/agents/trader.py` |
| **Dipendenze** | Nessuna |

**Regole per equities:**
- 9:30-9:35 ET: no trading (spread amplissimi, volatilità noise)
- 12:00-13:00 ET: ridurre position size del 50% (lunch hour, bassa liquidità)
- 15:55-16:00 ET: solo exit, no nuove posizioni

**Regole per crypto:**
- 00:00-04:00 UTC (Asia night): ridurre position size del 30% (volume basso)
- Intorno a settlement times (00:00, 08:00, 16:00 UTC): attenzione a funding-related moves

---

### 4.3 Adaptive Execution Algos

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P3 |
| **Effort** | 6-8 ore |
| **File coinvolti** | `app/execution/algos.py`, `app/execution/smart_router.py` |
| **Dipendenze** | 2.1 |

Modificare TWAP/VWAP per adattarsi al regime:
- Alta volatilità → slices più piccoli e più frequenti
- Volume spike → accelerare l'esecuzione (ride the momentum)
- Bassa liquidità → allargare la durata, ridurre partecipazione

---

## FASE 5 — POSITION SIZING INTELLIGENTE (Settimana 7-9)

### 5.1 Kelly Criterion Sizing

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P1 |
| **Effort** | 4-6 ore |
| **File coinvolti** | `app/risk/manager.py`, `app/strategies/confidence_calibrator.py` |
| **Dipendenze** | 0.2 |

**Cosa fare:**

1. Collegare il `ConfidenceCalibrator` (già esistente e funzionante) al sizing:
   ```python
   calibrated_confidence = calibrator.calibrate(strategy_name, raw_confidence)
   kelly_f = kelly_fraction(
       win_rate=calibrated_confidence,
       avg_win=strategy_avg_win,
       avg_loss=strategy_avg_loss,
   )
   position_size = equity * kelly_f * 0.5  # half-Kelly per prudenza
   position_size = min(position_size, equity * max_position_size_pct)
   ```
2. Tracciare per-strategy avg_win e avg_loss con rolling window di 100 trades
3. Se i dati sono insufficienti (< 30 trades), usare un sizing conservativo fisso

---

### 5.2 ATR-Based Stop Loss

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P2 |
| **Effort** | 3-4 ore |
| **File coinvolti** | `app/risk/manager.py` |
| **Dipendenze** | Nessuna |

Sostituire gli stop fissi con stop basati su ATR:
```python
atr = compute_atr(prices, highs, lows, period=14)
if is_crypto:
    stop_distance = atr * 2.5   # Più largo per crypto
else:
    stop_distance = atr * 1.5   # Equities
trailing_stop = max_high_since_entry - stop_distance
```

---

## FASE 6 — INFRASTRUTTURA DATI (Settimana 8-12)

### 6.1 Quote Data per Spread Modeling

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P2 |
| **Effort** | 8-10 ore |
| **File coinvolti** | Nuovo: `app/data/quote_stream.py` |
| **Dipendenze** | 1.2 |

1. Sottoscrivere Alpaca websocket per quote data (bid/ask/size) — incluso nel piano gratuito
2. Calcolare in real-time:
   - Spread medio per simbolo (utile per limit order pricing)
   - Quote imbalance (bid_size - ask_size) / (bid_size + ask_size) → feature per strategie
   - Microprice: (bid * ask_size + ask * bid_size) / (bid_size + ask_size) → prezzo fair più preciso del mid

### 6.2 Passaggio da yfinance ad Alpaca Data

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P2 |
| **Effort** | 4-6 ore |
| **File coinvolti** | `config/config.yaml`, `app/data/market_cache.py` |
| **Dipendenze** | Nessuna |

```yaml
data:
  provider: alpaca              # ← Era 'yfinance' o 'brokers'
  # Rimuovere yfinance come source primaria per live data
  sources:
    - provider: alpaca
      enabled: true
      # Alpaca data è incluso, più veloce e più affidabile di yfinance

market_cache:
  max_age_multiplier: 3         # ← Da 6 a 3 (15 min max staleness vs 30)
```

---

### 6.3 Alternative Data (Opzionale)

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P3 |
| **Effort** | 8-12 ore |
| **File coinvolti** | Nuovi file in `app/data/` |
| **Dipendenze** | Varie |

Per quando le fasi precedenti sono completate:

| Fonte | Costo | Effort | Integrazione |
|-------|-------|--------|--------------|
| Alpaca News (già integrato) | Gratis | 0 ore | Già fatto |
| Fear & Greed Index (crypto) | Gratis | 2 ore | REST → feature |
| CoinGlass (open interest, liquidations) | Gratis tier | 4 ore | REST → feature per crypto strategies |
| Unusual Whales (options flow) | $30/mo | 4 ore | REST → feature per equities |
| SEC EDGAR (insider trades) | Gratis | 4 ore | REST → earnings drift strategy |

---

## FASE 7 — PORTFOLIO CROSS-ASSET (Settimana 10-14)

### 7.1 Allocazione Cross-Asset

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P2 |
| **Effort** | 6-8 ore |
| **File coinvolti** | `app/portfolio/optimizer.py`, `app/portfolio/risk_model.py` |
| **Dipendenze** | 2.5, 1.6 |

Estendere il portfolio optimizer per gestire equities + crypto insieme:
- Calcolare la matrice di correlazione cross-asset
- Risk parity che bilancia il rischio tra equities (vol ~15% annuo) e crypto (vol ~60% annuo)
- Il risultato sarà ~80% equities e ~20% crypto in termini di notional, ma ~50/50 in termini di risk contribution

### 7.2 Sector/Theme Concentration Limits

| Campo | Dettaglio |
|-------|-----------|
| **Priorità** | P2 |
| **Effort** | 3-4 ore |
| **File coinvolti** | `app/risk/manager.py` |
| **Dipendenze** | Nessuna |

Aggiungere:
```yaml
risk:
  exposure_caps:
    enabled: true             # ← Era false
    sectors:
      Technology: 25.0
      Healthcare: 20.0
      Financials: 20.0
      crypto: 40.0
    themes:
      meme_stocks: 10.0
      ai_stocks: 15.0
```

---

## TIMELINE RIASSUNTIVA

```
Settimana  1-2:  FASE 0 (backtesting, metriche)
Settimana  2-4:  FASE 1 (crypto integration — IL CORE)
Settimana  3-5:  FASE 2 (attivazione componenti esistenti)
Settimana  4-8:  FASE 3 (nuove strategie)
Settimana  6-9:  FASE 4 (execution optimization)
Settimana  7-9:  FASE 5 (position sizing)
Settimana  8-12: FASE 6 (infrastruttura dati)
Settimana 10-14: FASE 7 (portfolio cross-asset)
```

Le fasi si sovrappongono perché molti task sono paralleli.

---

## EFFORT TOTALE STIMATO

| Fase | Ore Stimate | Priorità |
|------|-------------|----------|
| Fase 0 — Fondamenta | 16-22 | P0 |
| Fase 1 — Crypto Integration | 39-53 | P0 |
| Fase 2 — Attivazione Esistenti | 18-25 | P1 |
| Fase 3 — Nuove Strategie | 26-34 | P1-P2 |
| Fase 4 — Execution | 12-17 | P1-P3 |
| Fase 5 — Position Sizing | 7-10 | P1-P2 |
| Fase 6 — Infrastruttura Dati | 20-28 | P2-P3 |
| Fase 7 — Portfolio Cross-Asset | 9-12 | P2 |
| **TOTALE** | **147-201 ore** | |

Con 2-3 ore/giorno di lavoro = 2-3 mesi per il piano completo.
Con focus solo su P0 + P1 = ~85-110 ore = 6-8 settimane.

---

## MILESTONES E GO/NO-GO

| Milestone | Criterio | Azione se non raggiunto |
|-----------|----------|------------------------|
| **M1** (Fine sett. 2): Backtest walk-forward operativo | Report automatico con Sharpe, profit factor, benchmark comparison | Non procedere finché il framework non è solido |
| **M2** (Fine sett. 4): Crypto paper trading attivo | Almeno 50 trade crypto simulati, nessun bug di execution | Fix bugs prima di estendere le strategie |
| **M3** (Fine sett. 6): Sharpe > 0.5 su paper | Almeno 2 strategie con Sharpe > 0.5 dopo costi su 30 giorni di paper | Riesaminare le strategie, consultare i report del meta-orchestrator |
| **M4** (Fine sett. 10): Sharpe > 1.0 su paper | Portfolio complessivo con Sharpe > 1.0 su 60 giorni | Non andare live. Iterare su strategie e sizing |
| **M5** (Fine sett. 14): Ready for live | Sharpe > 1.0 stabile, max drawdown < 10%, tutte le strategie profittevoli dopo costi | Go live con 10-20% del capitale pianificato |

---

## NOTE IMPORTANTI

1. **Non andare live prima di M4.** Ogni giorno di paper trading è un giorno di dati gratuiti per il training. Non c'è fretta.

2. **Il meta-orchestrator LLM è il tuo "portfolio manager" automatico.** Una volta attivato e alimentato con dati sufficienti, adatterà i pesi delle strategie ai regimi di mercato meglio di qualsiasi regola statica.

3. **Il vantaggio competitivo di Fricktrade** non è nelle strategie individuali (che sono standard) ma nella combinazione di: multi-asset (equities + crypto), LLM intelligence (sentiment + regime detection + meta-orchestration), RL-based adaptation, e infrastruttura robusta. Nessuno di questi da solo fa profitto, ma insieme creano un edge composito.

4. **Costi LLM**: con il budget attuale di $5/giorno, hai ~100 chiamate Claude/giorno. Allocale: 30% sentiment analysis, 20% meta-orchestrator, 20% risk interpreter, 30% macro regime. Monitora con il `cost.log_all_costs: true` già configurato.
