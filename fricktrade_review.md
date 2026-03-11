# Fricktrade 3.0 — Valutazione come Strumento di Trading Finanziario

**Data:** 15 febbraio 2026  
**Scope:** Branch v3.0 (542 commit, ~24.300 LOC applicativo)  
**Tipo:** Valutazione di efficacia operativa, non code review

---

## Verdetto Sintetico

**Fricktrade 3.0 è un'infrastruttura di trading ambiziosa e strutturalmente matura, ma la cui capacità di generare alpha reale in mercati live rimane non dimostrata e presenta lacune strutturali significative nella qualità dei segnali, nella validazione statistica, e nella pipeline di esecuzione.**

Il sistema eccelle come framework operativo (risk management, monitoraggio, resilienza), ma le strategie di trading sono troppo semplici per competere con il rumore di mercato intraday, e i risultati di backtest riportati (+25.76%, +56.20%) non sono accompagnati da metriche di validazione sufficienti per essere considerati affidabili.

---

## Scorecard Complessiva

| Area | Voto | Giudizio |
|------|------|----------|
| Generazione di Alpha / Qualità Segnali | 4/10 | Strategie troppo semplici per edge reale |
| Risk Management | 8/10 | Multi-layer, solido, ben progettato |
| Qualità di Esecuzione | 5/10 | Infrastruttura presente ma sottoutilizzata |
| Data Pipeline / Selezione Universo | 6/10 | Innovativa (AI filter) ma fragile |
| Apprendimento / Adattabilità | 4/10 | RL ambizioso ma non funzionante in produzione |
| Infrastruttura Operativa | 8/10 | Production-ready, ottima osservabilità |
| Portfolio Construction | 3/10 | Implementata ma non collegata al loop |
| Validazione / Backtesting | 3/10 | Framework presente, risultati non verificabili |
| **COMPLESSIVO** | **5/10** | **Framework solido, motore di alpha debole** |

---

## 1. Generazione di Alpha — Qualità dei Segnali (4/10)

Questa è l'area più critica e quella dove il sistema mostra le debolezze più significative.

### 1.1 Trend Following — Troppo Naive per l'Intraday

La strategia usa un crossover MA(10)/MA(30) con filtro RSI(14) e conferma volume. Questo approccio ha tre problemi fondamentali:

- **Su barre 5-minuti, MA(10) = 50 minuti e MA(30) = 2.5 ore.** Queste medie mobili sono troppo lente per catturare momentum intraday genuino, e troppo veloci per filtrare il rumore. Il risultato è un numero elevato di falsi segnali in mercati laterali.
- **Il filtro RSI(14) su 5m è praticamente inutile.** RSI(14) su barre a 5 minuti = 70 minuti di dati. In un contesto intraday, l'RSI oscilla troppo rapidamente per funzionare come filtro di overbought/oversold affidabile. Studi quantitativi mostrano che RSI funziona meglio su timeframe giornalieri o superiori.
- **Breakout threshold dello 0.3% è troppo stretto.** Su azioni liquide US, lo spread bid-ask + slippage può facilmente mangiare uno 0.3%. La strategia rischia di tradare dentro lo spread senza generare profitto netto dopo costi.

### 1.2 Factor Model — Composito Ma Senza Robustezza

Il factor model combina momentum (10 bar), liquidità, volatilità e mean-reversion con una "trend quality gate" (proxy ADX). Le criticità:

- **I pesi dei fattori sono statici e arbitrari** (0.5 momentum, 0.2 liquidity, 0.1 volatility, 0.15 mean-reversion). Non c'è evidenza che questi pesi siano stati ottimizzati o che riflettano la realtà statistica dei mercati. In quant finance, i pesi dei fattori sono tipicamente stimati da regressione cross-sezionale o PCA, non fissati manualmente.
- **Mean-reversion su 20 barre (100 minuti) è problematico.** Il mean-reversion intraday funziona su scale temporali molto più brevi (1-5 minuti) o molto più lunghe (multi-giorno). 100 minuti è una "terra di nessuno" dove né il momentum né il mean-reversion hanno edge statistico documentato.
- **Il "trend quality gate" è una semplificazione grossolana.** Il rapporto tra net move e total absolute moves è una proxy di direzionalità, non di trend quality nel senso di ADX. Non cattura la persistenza del trend, solo la sua efficienza su una finestra fissa.

### 1.3 Pattern Trading — Entry Robusta, Exit Problematico

È probabilmente la strategia più solida del set. L'entry richiede breakout di range, trend OK (9/20 MA), higher highs/lows, pullback controllato, e volume confirmation. Il problema:

- **Lo stato della posizione è interno alla strategia** (`_PositionState`), ma la gestione exit è al livello dell'agente (take-profit, trailing stop, hard stop nel risk manager). Questo crea potenziali conflitti: la strategia crede di essere in posizione, l'agente ha già chiuso.
- **L'ATR-based stop è un buon approccio**, ma 2x ATR come moltiplicatore è aggressivo per intraday. Su barre a 5 minuti, l'ATR tende ad essere piccolo e lo stop risulta troppo stretto, causando uscite premature su normale noise di mercato.

### 1.4 Stat Arb Pairs — Concettualmente Valido, Implementazione Problematica

L'idea di pairs trading basato su correlazione rolling è legittima, ma:

- **Lo spread è calcolato come differenza di prezzo** (`a - b`), non come rapporto o residuo di regressione. Questo significa che coppie con prezzi molto diversi (es. AAPL a $200 vs MSFT a $400) generano spread dominati dalla scala dei prezzi, non dalla relazione statistica.
- **Non c'è test di cointegrazione.** Correlazione ≠ cointegrazione. Due titoli possono essere altamente correlati ma non cointegrati (cioè lo spread non è mean-reverting). Il pairs trading classico richiede il test di Engle-Granger o Johansen per validare la coppia.
- **La cache dei prezzi è append-only**, il che in un ambiente multi-thread con centinaia di simboli può diventare un problema di memoria e di freshness dei dati.

### 1.5 Combinazione Segnali — Il Cuore del Problema

L'orchestratore in modalità `weight` somma `confidence × strategy_weight` per ogni strategia. Ma:

- **La confidence non è calibrata.** Ogni strategia calcola la propria confidence in modo diverso (trend_strength / breakout_pct×3 per trend following, score / threshold×3 per factor model). Queste scale non sono confrontabili. Una confidence di 0.7 dal trend following non equivale a una confidence di 0.7 dal factor model.
- **Il min_conviction threshold (0.3) è una soglia su una somma pesata di valori non calibrati.** Non ha significato statistico.
- **Le strategie possono essere in conflitto.** Trend following potrebbe dire "buy" (momentum positivo) mentre mean-reversion dice "sell" (prezzo troppo alto vs media). Il vote mode li fa annullare a vicenda, ma non cattura il fatto che in certi regimi uno è più affidabile dell'altro.

---

## 2. Risk Management (8/10)

Il risk management è l'area più forte del sistema. La catena di guardie è ben stratificata:

- **Daily loss cap (3%)** con reset timezone-aware — corretto
- **Circuit breaker (5% drawdown)** — ragionevole
- **Position sizing cap (10%)** — prudente
- **VaR/CVaR gating** con finestra rolling di 60 barre — approccio istituzionale
- **Vol targeting** che scala le posizioni per volatilità target — best practice
- **Cooldown** tra trade — previene overtrading
- **Take-profit a due livelli** (partial + full) — gestione professionale
- **Trailing stop** — protegge i profitti
- **Kill switch con interlock code** — safety critica operativa
- **Exposure caps per venue/sector** — diversificazione forzata
- **Exit backoff esponenziale** — resilienza operativa

Le uniche debolezze del risk management:

- **La liquidity haircut è disabilitata di default.** Per un sistema che opera su fino a 50.000 simboli, non avere haircut di liquidità attivo è pericoloso. Un ordine su un titolo illiquido potrebbe soffrire di slippage significativo.
- **Il hard stop (0.8%) e il trailing stop (0.35%) sono molto stretti per intraday.** Su barre 5m con azioni US large-cap, i movimenti del ±0.5% sono normali rumore. Questi stop rischiano di essere triggerati costantemente, generando un alto numero di trade perdenti piccoli (death by a thousand cuts).

---

## 3. Qualità di Esecuzione (5/10)

### Punti di Forza

- **Infrastruttura execution algo completa**: TWAP, VWAP, POV sono implementati correttamente
- **Market impact model** che stima l'impatto in bps
- **Order queue FIFO** con retry budget e backoff
- **Smart router** con Almgren-Chriss implementato (ma non wired)

### Criticità

- **L'executor usa solo ordini market** (`self.broker.place_order(symbol, "buy", qty, "market")`). Questo è il tipo di ordine più costoso possibile. Per trading intraday dove i margini sono dell'ordine di 0.3-0.5%, pagare market order ogni volta erode significativamente i profitti.
- **L'Almgren-Chriss optimal trajectory è implementato ma mai chiamato**, come nota lo stesso improvement roadmap. Questo è un pezzo chiave per l'esecuzione su ordini di dimensioni significative.
- **Il TWAP con 4 slice su 120 secondi** è un approccio da retail, non da quant. Un TWAP serio dovrebbe adattare il numero di slice al volume disponibile, non usare parametri fissi.
- **Non c'è limit order support effettivo.** Il codice menziona "limit-order support for brokers", ma l'ExecutionEngine invia sempre market orders.

---

## 4. Data Pipeline e Selezione Universo (6/10)

### Innovazione: AI Symbol Filter

L'uso di un modello PPO per filtrare l'universo di simboli è un'idea interessante e originale. Il filtro combina:
- Features di prezzo/volume
- News catalysts via Ollama LLM
- Keras return prediction overlay
- Online updates ogni minuto

Questo è probabilmente l'elemento più innovativo del sistema.

### Problemi

- **yfinance come data source primario è inadeguato per trading live.** yfinance ha rate limits, ritardi, e non è progettato per uso real-time. I dati possono avere gap, timestamp inconsistenti, e problemi di adjustments. Per trading intraday serio, serve un data feed market data professionale (Polygon, Alpaca streaming, IEX Cloud real-time).
- **La configurazione `cache_only: true` + `ignore_staleness: true`** è un red flag. Questo significa che il sistema potrebbe tradare su dati vecchi senza saperlo. In un contesto intraday dove i movimenti sono piccoli, tradare su dati stale è equivalente a tradare alla cieca.
- **L'universo di 50.000 simboli** è esagerato per un sistema con le risorse compute di un singolo server. Processare 50.000 simboli ogni ciclo su barre 5m crea un collo di bottiglia che degrada la latenza del loop di trading.

---

## 5. Apprendimento e Adattabilità (4/10)

### RL Policy — Ambizioso Ma Non Funzionante

La documentazione stessa ammette che le RL policies sono disabilitate perché producono "near-uniform output (~33/33/33)". Questo è il sintomo classico di un agente RL che non ha imparato nulla di utile. Le ragioni probabili:

- **Il reward function è ben progettato** (differential NAV + time penalty + profit bonus + velocity), ma la funzione di reward da sola non garantisce apprendimento se l'ambiente è troppo rumoroso o lo spazio delle azioni troppo semplice.
- **Lo spazio delle azioni (buy/hold/sell) è troppo discreto** per catturare la complessità del trading. Un sistema RL efficace dovrebbe decidere anche sizing, timing, e tipo di ordine.
- **Il credit assignment problem nel trading è estremo.** L'effetto di un buy oggi potrebbe manifestarsi tra ore o giorni. PPO con i parametri attuali (2000 timesteps, batch 64) è probabilmente insufficiente per apprendere pattern significativi.
- **Online updates con 10 steps e 256 timesteps** sono insufficienti per adaptation significativa. Il modello cambia così poco che è effettivamente statico.

### Drift Detection — Concetto Corretto

Il monitoraggio della distribuzione delle features e del PnL rolling con auto-rollback al best model è un approccio corretto e professionale. Ma se il modello base non funziona, il rollback non risolve il problema.

---

## 6. Infrastruttura Operativa (8/10)

Questa è un'area di eccellenza:

- **14 servizi Docker** orchestrati con compose, GPU detection automatico
- **Prometheus + Grafana** con dashboard pre-provisionate per ordini, posizioni, PnL, latenza
- **Alertmanager** con notifiche email
- **Healthwatch** con market-based sleep/wake
- **Autoheal** per restart automatico di container falliti
- **Checkpointing** con retention pruning
- **Audit trail** con HMAC signing per compliance
- **FastAPI** con config UI per tuning real-time
- **Docker socket proxy** per isolamento di sicurezza
- **Daily report** automatizzato con top movers e decision trace

Questo livello di infrastruttura operativa è professionale e superiore a molti sistemi di trading retail. Il sistema è pensato per girare in produzione autonomamente.

---

## 7. Portfolio Construction (3/10)

Il modulo `app/portfolio/optimizer.py` implementa:
- Mean-variance (Markowitz)
- Risk parity
- Maximum Sharpe
- Black-Litterman

Tuttavia, **nessuno di questi è collegato al loop di trading attivo.** Il sizing viene fatto dal risk manager come percentuale del portfolio (max_position_size_pct), non dall'ottimizzatore. L'intero modulo portfolio è effettivamente codice morto dal punto di vista operativo.

In un sistema multi-strategy multi-symbol, l'allocazione del capitale tra strategie e tra posizioni è critica. Senza un portfolio optimizer attivo, il sistema alloca capitale in modo "first come, first served", il che è subottimale.

---

## 8. Validazione e Backtesting (3/10)

### Risultati Riportati

Il README riporta:
- RL-only strategy: **+25.76%**, 12 trade su full-year run
- Dual RL strategies: **+56.20%** su short-window dynamic-symbol tests

### Problemi di Credibilità

Questi risultati non sono accompagnati da:
- **Sharpe ratio, Sortino ratio, Calmar ratio** — metriche standard per valutare risk-adjusted returns
- **Maximum drawdown** — quanto il sistema ha perso nel peggior momento
- **Numero di trade totali vs win rate** — 12 trade in un anno suggerisce pochissima attività; il +25.76% potrebbe essere guidato da 1-2 trade fortunati
- **Confronto con buy-and-hold** — il benchmark runner lo supporta ma i risultati non sono riportati
- **Walk-forward validation** — il framework lo supporta, ma non c'è evidenza che sia stato usato per validare i risultati
- **Out-of-sample testing** — non c'è separazione chiara tra train/validation/test periods
- **Costi di transazione realistici** — il broker Alpaca paper ha zero commissioni, ma lo slippage reale non è modelizzato
- **Periodo di test** — "full-year" e "short-window" sono vaghi. In quale anno? Con quale regime di mercato?

Il +56.20% su "short-window dynamic-symbol tests" è particolarmente sospetto: short window + dynamic symbol selection = alto rischio di data snooping e overfitting alla selezione.

### Il Framework di Benchmark è Presente Ma Non Usato

Il sistema ha un benchmark runner con bootstrap CI, Monte Carlo stress, scorecard metrics, e regime tagging. Questi sono tutti gli strumenti giusti. Ma i risultati nel README non li usano, il che suggerisce che la validazione rigorosa non è stata ancora eseguita.

---

## Rischi Chiave per il Deploiement in Live Trading

### Rischio 1: Death by Spread and Slippage

Con stop loss del 0.35-0.8% e entry threshold dello 0.3%, il margine operativo è estremamente sottile. Su azioni con spread bid-ask dello 0.05-0.15% + slippage da market order, il margine netto è quasi zero. In mercati veloci, lo slippage può facilmente superare il margine atteso.

### Rischio 2: Overtrading su Universo Troppo Grande

50.000 simboli con scansione ogni minuto, 4 strategie per simbolo, e stop molto stretti = potenzialmente centinaia di segnali al giorno. Con cooldown di soli 5 secondi, il sistema potrebbe generare un volume di ordini insostenibile che erode il capitale in commissioni e slippage.

### Rischio 3: Dati Stale = Decisioni Errate

`cache_only: true` + `ignore_staleness: true` è una configurazione che accetta consapevolmente dati vecchi. In un contesto dove i profitti target sono <1%, tradare su dati vecchi di anche solo 2-5 minuti può trasformare un trade winning in un trade perdente.

### Rischio 4: RL Non Funzionante Ma Integrato

Anche se le RL policies sono disabilitate, l'AI symbol filter (PPO) è attivo. Se il modello PPO per la selezione simboli ha gli stessi problemi di convergenza delle RL policies (output quasi-uniforme), il filtro potrebbe selezionare simboli essenzialmente a caso, degradando la qualità dell'universo di trading.

### Rischio 5: Assenza di Portfolio Optimization

Senza un allocatore di portafoglio attivo, il sistema potrebbe concentrare il capitale in pochi titoli correlati (es. tutti tech stock), amplificando il rischio settoriale. I caps per venue/sector nel risk manager mitigano parzialmente questo, ma sono disabilitati di default.

---

## Raccomandazioni Prioritarie

### Priorità 1: Validazione Rigorosa Prima di Qualsiasi Trading Reale

Eseguire il benchmark runner completo con walk-forward su almeno 3 anni di dati, riportando Sharpe, Sortino, max drawdown, e confronto con buy-and-hold. Se lo Sharpe ratio annualizzato è < 1.0, il sistema non ha edge sufficiente per il live trading.

### Priorità 2: Upgrade delle Strategie

- Sostituire il crossover MA con un modello di momentum basato su informazione di ordini (order flow, quote imbalance)
- Implementare cointegrazione nel pairs trading
- Calibrare la confidence su base statistica (es. p-value del segnale)
- Usare multi-timeframe features (1m, 5m, 15m, 1h) per distinguere trend da rumore

### Priorità 3: Esecuzione con Limit Order

Passare a limit order come default (post al mid o meglio) eliminerebbe gran parte del costo di crossing the spread. Per un sistema intraday con margini sottili, questa è forse la singola modifica con il maggiore impatto sul P&L netto.

### Priorità 4: Collegare il Portfolio Optimizer

Il modulo esiste già. Collegarlo al loop di trading per allocare il capitale in modo ottimale tra strategie e posizioni migliorebbe significativamente il risk-adjusted return.

### Priorità 5: Disabilitare `cache_only` e `ignore_staleness`

Per trading live, i dati devono essere freschi. Meglio avere latenza extra per un fetch live che tradare su dati stale.

---

## Conclusione

Fricktrade 3.0 è un progetto impressionante dal punto di vista ingegneristico. La quantità di lavoro nell'infrastruttura operativa, nel risk management, e nel framework di monitoring è di livello professionale. Il sistema è chiaramente costruito da qualcuno che capisce i requisiti operativi del trading automatizzato.

Tuttavia, un sistema di trading è buono quanto i suoi segnali, e i segnali di Fricktrade 3.0 sono la parte più debole. Le strategie sono implementazioni naive di concetti classici, senza l'edge statistico necessario per sopravvivere ai costi di transazione intraday. L'RL, che dovrebbe essere il differenziatore, non funziona ancora.

La raccomandazione è chiara: **non deployare in live trading con denaro reale fino a quando la validazione rigorosa non dimostra un edge positivo statisticamente significativo dopo costi.** Il framework c'è, gli strumenti ci sono, ma il motore di alpha ha bisogno di lavoro sostanziale prima di essere affidabile.

Il potenziale è significativo. Se le strategie vengono portate a livello quant-grade (features microstructure, multi-timeframe, portfolio optimization attiva, limit orders), e validate rigorosamente, il sistema ha l'infrastruttura per supportarlo. La road da qui a lì è chiaramente tracciata nei documenti di roadmap del progetto stesso.
