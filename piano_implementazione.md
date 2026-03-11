# Fricktrade 3.0 — Piano di Implementazione

**Obiettivo: Massimizzare il profitto nel minor tempo possibile attraverso decisioni buy/sell ottimali.**

L'unica cosa che puoi controllare sono gli ordini. Tutto il resto — infrastruttura, monitoring, architettura — ha valore solo nella misura in cui migliora la qualità di quegli ordini. Questo piano è ordinato per **impatto atteso sul PnL**, non per eleganza tecnica.

---

## Diagnosi: Perché il Sistema Attuale Non Genera Alpha

Il problema centrale non è la mancanza di componenti — è la **disconnessione** tra componenti avanzati e il percorso critico delle decisioni. Ecco cosa succede oggi quando il sistema decide un ordine:

```
Dati (yfinance, possibilmente stale) 
  → 4 strategie rule-based (MA crossover, factor composito, pattern breakout, spread diff)
    → Vote/weight con confidence non calibrate
      → Risk manager (OK)
        → executor.py: broker.place_order(symbol, "buy", qty, "market")  ← SEMPRE market order
```

I componenti avanzati **esistono** ma **non sono nel percorso critico**:
- SmartOrderRouter con Almgren-Chriss → mai chiamato dall'executor
- Portfolio optimizer (Markowitz, risk parity, Black-Litterman) → non collegato al sizing attivo
- TCN feature extractor → collegato al training RL, ma RL produce output uniforme ~33/33/33
- Ensemble (LightGBM + meta-learning) → wired nell'orchestratore ma RL è disabilitato
- RegimeHMM → wired nel trader ma impatto non verificato
- 22 indicatori tecnici estesi → usati solo nelle features RL, non dalle 4 strategie attive
- TCA → calcola metriche ma non retroagisce sulle decisioni

**Risultato**: il sistema prende decisioni con 4 strategie semplici su dati potenzialmente vecchi, e le esegue nel modo più costoso possibile (market order).

---

## Fase 0: Smettere di Perdere Soldi (Giorni 1-3)

**Impatto PnL atteso: immediato, evita perdite sistematiche**

Queste modifiche non generano alpha ma eliminano drag che erode qualsiasi edge.

### 0.1 Disabilitare `cache_only` e `ignore_staleness`

**File**: `config/config.yaml`

```yaml
# PRIMA (pericoloso)
market_cache:
  cache_only: true       # ← legge solo dalla cache, mai dal mercato
  ignore_staleness: true  # ← usa dati anche se vecchi di ore

# DOPO
market_cache:
  cache_only: false
  ignore_staleness: false
  max_age_multiplier: 6   # 5m × 6 = 30 minuti TTL max
```

**Perché**: Con target di profitto dello 0.3-1.5% su intraday, tradare su dati stale di anche solo 10 minuti può invertire completamente la direzione del segnale. È come guidare guardando lo specchietto retrovisore.

### 0.2 Limit Order come Default nell'Executor

**File**: `app/execution/executor.py`

Il broker Alpaca supporta già limit order (linee 102-123 di `alpaca.py`). L'executor deve usarli.

```python
# ATTUALE: 30 righe, solo market order
def execute(self, symbol, action, qty):
    return self.broker.place_order(symbol, action, qty, "market")

# NUOVO: limit order al mid-price con fallback a market
def execute(self, symbol, action, qty, market_state=None):
    bid = market_state.get("bid") if market_state else None
    ask = market_state.get("ask") if market_state else None
    
    if bid and ask and ask > bid:
        mid = (bid + ask) / 2
        # Post al mid: cattura metà dello spread
        limit_price = mid if action == "buy" else mid
        order_id = self.broker.place_order(
            symbol, action, qty, "limit", limit_price=limit_price
        )
        # Schedule check: se non filled in 30s, convert a market
        return order_id
    else:
        return self.broker.place_order(symbol, action, qty, "market")
```

**Impatto**: Su uno spread medio di 0.05-0.15%, postare al mid anziché crossare lo spread risparmia ~0.025-0.075% per trade. Su centinaia di trade/mese con margini dello 0.3-1.5%, questo è enorme — può fare la differenza tra profitto e perdita netta.

**Collegamento con smart router**: Successivamente (Fase 2), il `SmartOrderRouter` esistente sostituirà questa logica con decisioni più sofisticate. Per ora, il limit al mid è un miglioramento a costo zero.

### 0.3 Propagare `market_state` all'Executor

**File**: `app/agents/trader.py`

Attualmente il trader chiama l'executor senza passare bid/ask. Bisogna propagare lo stato di mercato fino all'esecuzione, perché senza bid/ask il limit order non è possibile.

Cercare dove `execute()` viene chiamato e aggiungere il `market_state` disponibile nel contesto.

---

## Fase 1: Segnali che Funzionano (Giorni 4-14)

**Impatto PnL atteso: alto — è qui che si genera o si perde alpha**

### 1.1 Iniettare le Feature Avanzate nelle Strategie Attive

**Problema**: Le 4 strategie calcolano tutto internamente con codice duplicato. I 22 indicatori in `indicators.py`, le feature regime, i dati multi-timeframe **non vengono mai visti** dalle strategie attive.

**Soluzione**: Arricchire `market_state` prima che arrivi alle strategie.

**File**: `app/agents/trader.py` (nel loop per-symbol, prima di chiamare le strategie)

```python
from app.learning.indicators import compute_extended_indicators
from app.learning.regime import compute_regime_features
from app.learning.multi_timeframe import compute_mtf_features

# Prima di passare market_state alle strategie:
if len(prices) >= 50:
    market_state["indicators"] = compute_extended_indicators(
        highs, lows, closes, volumes
    )
    market_state["regime"] = compute_regime_features(returns)
    market_state["mtf"] = compute_mtf_features(ohlcv_df)
```

### 1.2 Riscrivere Trend Following con Feature Avanzate

**File**: `app/strategies/trend_following.py`

Il trend following attuale usa MA(10/30) crossover — un segnale che i mercati hanno ampiamente arbitraggiato. Con le feature avanzate disponibili, può diventare molto più sofisticato.

**Nuova logica**:
```
ENTRY (buy):
  - ADX > 25 (trend forte, non proxy semplificato)
  - +DI > -DI (direzione up)  
  - Supertrend bullish
  - RSI 40-70 (non ipercomprato, non oversold)
  - Volume > 1.5× media
  - Regime != high_volatility (da RegimeHMM)
  - VWAP deviation: prezzo sopra VWAP session

EXIT (sell):
  - ADX < 20 (trend morto) OPPURE
  - +DI < -DI (inversione) OPPURE  
  - RSI > 75 (ipercomprato) OPPURE
  - Supertrend flip bearish
```

**Perché è meglio**: ADX reale (non il proxy `net_move/total_move`), Supertrend, regime awareness e VWAP deviation sono tutti segnali con decenni di letteratura quantitativa a supporto. La combinazione di più conferme riduce i falsi segnali.

### 1.3 Riscrivere Stat Arb Pairs: Ratio e Cointegrazione

**File**: `app/strategies/stat_arb_pairs.py`

Due problemi critici nella versione attuale:

**Problema 1 — Spread come differenza**: `spread = a - b` (linea 56). Se AAPL è a $180 e MSFT a $420, lo spread è -$240 e il suo z-score non ha senso economico. Deve essere un ratio: `spread = log(a) - log(b)` oppure il residuo di una regressione.

**Problema 2 — Correlazione ≠ Cointegrazione**: La selezione dei pair usa `np.corrcoef` (linea 87). Due asset possono essere altamente correlati ma divergere nel lungo termine (es. due tech stock in settori diversi). La cointegrazione (test di Engle-Granger o Johansen) è il requisito corretto per il mean-reversion.

**Nuova logica**:
```python
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

# Selezione pair: test di cointegrazione
def _refresh_pairs(self):
    pairs = []
    for i, j in combinations(symbols, 2):
        a, b = price_series[i], price_series[j]
        score, pvalue, _ = coint(a, b)
        if pvalue < 0.05:  # cointegrati al 95%
            # Regressione OLS per hedge ratio
            model = sm.OLS(a, sm.add_constant(b)).fit()
            hedge_ratio = model.params[1]
            pairs.append((symbols[i], symbols[j], hedge_ratio, pvalue))
    pairs.sort(key=lambda p: p[3])  # ordina per p-value
    self._pairs = pairs[:self.params.max_pairs]

# Segnale: spread come residuo della regressione
def generate_signal(self, market_state):
    spread = log(price_a) - hedge_ratio * log(price_b)
    z = (spread - spread_mean) / spread_std
    # Entry: z > 2.0 o z < -2.0
    # Exit: |z| < 0.5
```

**Dipendenze**: `statsmodels` (probabilmente già installato).

### 1.4 Calibrare la Confidence Attraverso le Strategie

**Problema**: Ogni strategia calcola `confidence` in modo diverso e su scale diverse. Trend following usa `trend_strength / (breakout_pct * 3)`, factor model usa `|score| / (threshold * 3)`, pattern trading non emette confidence. Quando il combiner fa `confidence * strategy_weight`, sta sommando mele e arance.

**Soluzione**: Normalizzare le confidence a probabilità calibrate.

**File**: Nuovo file `app/strategies/confidence_calibrator.py`

```python
class ConfidenceCalibrator:
    """Normalizza confidence su [0,1] con significato probabilistico."""
    
    def __init__(self, window=200):
        self.history = {}  # {strategy_name: deque of (confidence, was_profitable)}
    
    def record(self, strategy, confidence, profitable):
        """Registra outcome per calibrazione."""
        self.history.setdefault(strategy, deque(maxlen=200))
        self.history[strategy].append((confidence, profitable))
    
    def calibrate(self, strategy, raw_confidence):
        """Restituisce P(profitto | confidence_raw, strategy)."""
        hist = self.history.get(strategy, [])
        if len(hist) < 30:
            return raw_confidence * 0.5  # conservativo finché non calibrato
        
        # Bin the confidences e calcola win rate per bin
        # Isotonic regression per monotonicity
        from sklearn.isotonic import IsotonicRegression
        confs = [h[0] for h in hist]
        outcomes = [1.0 if h[1] else 0.0 for h in hist]
        ir = IsotonicRegression(y_min=0, y_max=1)
        ir.fit(confs, outcomes)
        return float(ir.predict([raw_confidence])[0])
```

**Integrazione**: Dopo ogni trade chiuso, registrare `(confidence_al_momento_del_segnale, trade_profitable)`. Usare la confidence calibrata nel combiner.

### 1.5 Arricchire il Signal Combiner

**File**: `app/agents/trader.py`, metodo `_combine_signals()`

La logica attuale (linee 1881-1945) fa una somma pesata di `confidence * strategy_weight` e confronta buy_score vs sell_score. È un buon framework ma manca di:

1. **Correlation discount**: Se trend_following e factor_model dicono entrambi "buy" sullo stesso momentum signal, il loro contributo combinato dovrebbe essere meno di 2× perché sono correlati. Aggiungere un fattore di discount per segnali basati su feature simili.

2. **Regime weighting**: In regime trending, dare più peso a trend_following e pattern_trading. In regime mean-reverting, dare più peso a stat_arb e factor_model (componente mr). Il RegimeHMM è già wired — usarne l'output per modulare i pesi.

```python
def _adjust_weights_for_regime(self, weights, regime):
    """Modula pesi strategia in base al regime di mercato."""
    if regime == 0:  # low vol / trending
        weights["trend_following"] *= 1.3
        weights["pattern_trading"] *= 1.2
        weights["stat_arb_pairs"] *= 0.8
    elif regime == 2:  # high vol / mean-reverting
        weights["trend_following"] *= 0.7
        weights["stat_arb_pairs"] *= 1.4
        weights["factor_model"] *= 1.2
    return weights
```

---

## Fase 2: Esecuzione Intelligente (Giorni 15-25)

**Impatto PnL atteso: medio-alto — recupero di 0.05-0.15% per trade**

### 2.1 Collegare lo Smart Router all'Executor

Lo `SmartOrderRouter` esiste e funziona (260 righe, Almgren-Chriss implementato). L'executor deve usarlo al posto della logica diretta.

**File**: `app/execution/executor.py`

```python
from app.execution.smart_router import SmartOrderRouter, OrderContext

class ExecutionEngine:
    def __init__(self, broker, config=None):
        self.broker = broker
        self.router = SmartOrderRouter(config)
    
    def execute(self, symbol, action, qty, market_state=None):
        if market_state and qty > 0:
            ctx = OrderContext(
                symbol=symbol, side=action, qty=int(qty),
                price=market_state.get("last_price", 0),
                urgency=market_state.get("urgency", 0.5),
                market_state=market_state
            )
            decision = self.router.route(ctx)
            # decision.algo = "twap" | "vwap" | "pov" | "market"
            # decision.slices = lista di (qty, time) per esecuzione frazionata
            
            if decision.algo == "market" or len(decision.slices) <= 1:
                # Ordine piccolo: limit al mid con timeout
                return self._execute_limit_with_timeout(symbol, action, qty, market_state)
            else:
                # Ordine grande: esecuzione frazionata
                return self._execute_sliced(symbol, action, decision.slices, market_state)
```

### 2.2 TCA Feedback Loop

Il modulo TCA esiste e calcola slippage, impact, timing cost. Attualmente non retroagisce sulle decisioni. Collegarlo.

**Meccanismo**: Dopo ogni fill, calcolare il costo reale di esecuzione. Se una strategia genera sistematicamente trade con alto slippage (es. symbol illiquidi), penalizzarne la confidence futura.

**File**: Aggiungere in `app/agents/trader.py` dopo la conferma dell'ordine:

```python
# Dopo il fill
from app.execution.tca import TCAAnalyzer

tca = TCAAnalyzer()
slippage = tca.compute_slippage(fill_price, decision_price)
# Feedback alla strategia: se slippage > expected, ridurre conviction futura per quel symbol
if slippage_bps > 10:
    self._symbol_slippage_penalty[symbol] = min(
        self._symbol_slippage_penalty.get(symbol, 0) + 0.1, 0.5
    )
```

---

## Fase 3: Portfolio-Level Optimization (Giorni 20-35)

**Impatto PnL atteso: medio — allocazione ottimale del capitale**

### 3.1 Collegare il Portfolio Optimizer al Trading Loop

Il `PortfolioOptimizer` implementa Markowitz, risk parity e Black-Litterman. Il `RebalanceEngine` calcola i trade necessari. Nessuno dei due è nel percorso attivo degli ordini.

**Stato attuale del sizing**: Il risk manager calcola `position_size = equity * max_position_size_pct` (10%) per ogni trade, indipendentemente da quante posizioni sono aperte o da come sono correlate. È capital allocation first-come-first-served.

**Nuovo flusso**:
```
Segnali delle 4 strategie (con confidence calibrate)
  → Raccogliere tutti i "buy" candidati del ciclo
    → Portfolio optimizer: dato il portafoglio attuale + candidati,
       calcolare i target weight ottimali (risk parity o max Sharpe)
      → Per ogni candidato: qty = target_weight * equity / price
        → Risk manager: verifica limiti
          → Executor: esegui
```

**File**: `app/agents/trader.py` — sostituire il sizing fisso con una chiamata all'optimizer.

```python
from app.portfolio.optimizer import PortfolioOptimizer, PortfolioConstraints
from app.portfolio.risk_model import RiskModel

# All'inizio del ciclo, dopo aver raccolto tutti i segnali:
candidates = {sym: signal for sym, signal in all_signals.items() if signal["action"] == "buy"}
if candidates:
    # Stima rendimenti attesi dalle confidence calibrate
    expected_returns = {sym: signal["calibrated_confidence"] * 0.01 for sym, signal in candidates.items()}
    
    # Stima covarianza dai rendimenti storici
    risk_model = RiskModel()
    cov = risk_model.compute_covariance(historical_returns[list(candidates.keys())])
    
    # Ottimizzazione
    optimizer = PortfolioOptimizer()
    result = optimizer.max_sharpe(expected_returns, cov, constraints)
    
    # result.weights = {"AAPL": 0.15, "MSFT": 0.10, ...}
    for sym, weight in result.weights.items():
        qty = int(weight * equity / prices[sym])
        # ... esegui
```

### 3.2 Gestione Correlazione e Concentrazione

Aggiungere un check pre-trade: se il portafoglio ha già esposizione al settore/fattore del candidato, ridurre il sizing. Il modulo `risk_model.py` ha già `factor_risk()` per questo.

---

## Fase 4: Far Funzionare l'RL (Giorni 25-45)

**Impatto PnL atteso: potenzialmente alto, ma il più rischioso**

### 4.1 Diagnosi del Problema di Convergenza

L'RL attualmente produce output ~33/33/33 (buy/hold/sell uniformi). Le cause probabili sono:

1. **Action space troppo discreto**: Buy/Hold/Sell con quantità fissa non cattura la sfumatura del sizing.
2. **Credit assignment estremo**: Su bar da 5 minuti, l'effetto di un buy si vede dopo 10-60 bar. PPO con γ=0.99 e horizon corto non riesce ad attribuire il reward.
3. **Training insufficiente**: 2000 timestep per il filtro AI, 50000 per il policy — PPO su dati finanziari tipicamente ne richiede 500k-2M.
4. **Reward signal troppo debole**: NAV differenziale normalizzato su 100k + time penalty crea un segnale reward molto piccolo per singolo step.

### 4.2 Piano di Fix

**Step 1 — Continuous action space** (alto impatto):
```python
# ATTUALE: Discrete(3) → buy=0, hold=1, sell=2
# NUOVO: Box(-1, 1) → -1=sell max, 0=hold, +1=buy max
# Il sizing è proporzionale all'azione
action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
```
Questo è un cambio fondamentale: il policy deve imparare solo "quanto" e "in che direzione", non una scelta categorica.

**Step 2 — Reward shaping migliorato**:
```python
# Aggiungere reward immediato per posizioni in profitto (mark-to-market)
# Ogni bar, se in posizione:
unrealized_pnl_pct = (current_price - entry_price) / entry_price
immediate_reward = unrealized_pnl_pct * 10.0  # amplifica il segnale
# Questo dà al policy feedback continuo, non solo al close della posizione
```

**Step 3 — Training su più dati** (necessario):
```yaml
learning:
  training:
    timesteps: 500000    # era 50000
    n_envs: 8            # parallelizzare con SubprocVecEnv
  online:
    steps: 50            # era 10
    timesteps: 2048      # era 256
```

**Step 4 — Curriculum learning**: Iniziare il training su dati facili (trend chiari, alta volatilità) e progressivamente introdurre condizioni difficili (choppy, bassa vol). Questo è implementabile con un wrapper dell'environment che filtra i dati.

### 4.3 Validazione: RL Solo se Batte le Strategie Rule-Based

Prima di attivare l'RL in produzione, deve battere la combinazione delle 4 strategie in backtest walk-forward su almeno 1 anno di dati out-of-sample. Se non ci riesce, le strategie rule-based migliorate (Fase 1) sono la scelta migliore.

---

## Fase 5: Validazione Rigorosa (Parallela, Giorni 1-45)

**Questa fase corre in parallelo a tutte le altre. Non deployare nulla in live senza validazione.**

### 5.1 Benchmark Basale (Giorno 1)

Prima di qualsiasi modifica, catturare la performance attuale:
```bash
docker compose run --rm trader python3 -m app.main backtest \
  --config /app/config/backtest_case1.yaml
```

Registrare: Sharpe, Sortino, max drawdown, win rate, numero trade, PnL netto dopo costi, confronto con buy-and-hold SPY.

### 5.2 Walk-Forward per Ogni Modifica

Per ogni fase completata, eseguire walk-forward validation:
- Training: 6 mesi rolling
- Validation: 1 mese
- Test: 1 mese (mai toccato fino al test finale)
- Almeno 12 finestre rolling

Il benchmark runner esistente supporta questo. **Usarlo.**

### 5.3 Criteri Go/No-Go per Live

| Metrica | Minimo per Go-Live | Target |
|---------|-------------------|--------|
| Sharpe annualizzato (netto costi) | > 1.0 | > 1.5 |
| Max drawdown | < 15% | < 10% |
| Win rate | > 50% | > 55% |
| Profit factor | > 1.3 | > 1.5 |
| Avg trade netto spread+slippage | > 0 | > 0.1% |
| N. trade nel test | > 50 | > 200 |
| Confronto vs buy-hold SPY | Positivo | > 2× |

---

## Priorità e Timeline

```
Giorno  1-3:   FASE 0 — Stop bleeding (cache, limit orders)
Giorno  1-45:  FASE 5 — Validazione (parallela a tutto)
Giorno  4-14:  FASE 1 — Segnali che funzionano (strategie, confidence, combiner)
Giorno 15-25:  FASE 2 — Esecuzione intelligente (smart router, TCA feedback)
Giorno 20-35:  FASE 3 — Portfolio optimization (sizing, correlazione)
Giorno 25-45:  FASE 4 — RL (se le fasi 1-3 non bastano)
```

### Perché questo ordine

1. **Fase 0 prima di tutto**: Ogni trade eseguito come market order su dati stale perde soldi inutilmente. Bloccare l'emorragia prima.

2. **Fase 1 è il core**: Le strategie generano i segnali buy/sell. Se i segnali sono sbagliati, nient'altro importa. Feature avanzate + cointegrazione + calibrazione sono le modifiche a più alto impatto.

3. **Fase 2 dopo la 1**: Ha senso ottimizzare l'esecuzione solo quando i segnali sono buoni. Se i segnali sono random, eseguire perfettamente produce comunque perdite.

4. **Fase 3 dopo la 2**: Il portfolio optimization ha impatto quando ci sono abbastanza posizioni simultanee da beneficiare della diversificazione. Prima bisogna avere segnali e esecuzione funzionanti.

5. **Fase 4 è opzionale**: Se le fasi 1-3 producono Sharpe > 1.0, l'RL potrebbe non essere necessario. Se le strategie rule-based con feature avanzate non bastano, allora l'RL con le fix proposte diventa il piano B.

---

## Modifiche ai File — Riepilogo

| File | Fase | Tipo | Descrizione |
|------|------|------|-------------|
| `config/config.yaml` | 0 | Modifica | cache_only=false, ignore_staleness=false |
| `app/execution/executor.py` | 0, 2 | Riscrittura | Limit order default → Smart router |
| `app/agents/trader.py` | 0, 1, 3 | Modifica | Propagare market_state; arricchire con features; portfolio sizing |
| `app/strategies/trend_following.py` | 1 | Riscrittura | ADX reale, Supertrend, VWAP, regime-aware |
| `app/strategies/stat_arb_pairs.py` | 1 | Riscrittura | Log-ratio spread, cointegrazione, hedge ratio |
| `app/strategies/factor_model.py` | 1 | Modifica | Usare indicatori estesi dal market_state |
| `app/strategies/confidence_calibrator.py` | 1 | Nuovo | Calibrazione probabilistica confidence |
| `app/agents/trader.py` (`_combine_signals`) | 1 | Modifica | Regime weighting, correlation discount |
| `app/execution/executor.py` | 2 | Modifica | Wire SmartOrderRouter |
| `app/agents/trader.py` (post-fill) | 2 | Modifica | TCA feedback loop |
| `app/agents/trader.py` (sizing) | 3 | Modifica | Wire PortfolioOptimizer per target weights |
| `app/learning/env.py` | 4 | Modifica | Continuous action space, reward shaping |
| `config/config.yaml` (learning) | 4 | Modifica | timesteps 500k, n_envs 8 |

---

## Note Finali

Questo piano si concentra deliberatamente sulle **strategie e l'esecuzione** perché sono i due punti dove il denaro entra o esce. L'infrastruttura di Fricktrade (Docker, monitoring, risk management, checkpoint) è già a livello professionale e non ha bisogno di lavoro.

Il rischio principale è l'overfitting: ogni miglioramento delle strategie deve essere validato out-of-sample. La Fase 5 esiste per questo, e deve essere seguita con disciplina.

Se dopo le Fasi 0-3 il sistema produce Sharpe > 1.0 netto di costi in walk-forward validation, hai un edge tradabile. Se non ci riesce, la Fase 4 (RL con le fix proposte) è il tentativo successivo. Se neanche quello funziona, il problema non è implementativo ma di alpha — e la risposta onesta è che servono feature o dati diversi (order flow, options flow, alternative data).
