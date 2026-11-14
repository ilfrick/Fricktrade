<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Fricktrade — Letteratura applicabile e spunti trasferibili

**Data:** 2026-11-14

Criterio di selezione: ho cercato solo lavori che risolvano un problema **presente nel codice**
o che propongano un'ipotesi **testabile con i dati già in `data/`**. Ogni voce indica dove si
aggancia, quanto costa, e — soprattutto — che cosa cambierebbe *misurabilmente*.

Avvertenza onesta: la maggior parte di questi spunti migliora la **capacità di misurare**, non
l'alpha. Solo i punti 4.1–4.3 propongono una possibile fonte di rendimento, e la letteratura
suggerisce che a questa dimensione di conto il rendimento più probabile non sia nel
prevedere il prezzo.

---

## 1. Validazione statistica — il difetto più costoso del progetto

Il progetto è stato chiuso perché una strategia non ha superato un test. Il test stesso ha
almeno cinque difetti misurati (doc 02, sezione B). Questo filone è quindi il più importante.

### 1.1 Deflated Sharpe Ratio (DSR)
> **Bailey, D. H. & López de Prado, M. (2014).** *The Deflated Sharpe Ratio: Correcting for
> Selection Bias, Backtest Overfitting, and Non-Normality.* Journal of Portfolio Management
> 40(5), 94–107.

**Idea.** Il massimo Sharpe su N tentativi indipendenti cresce come
`E[max SR] ≈ sqrt(Var[SR]) · ((1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(Ne)))`.
Il DSR è la probabilità che lo Sharpe osservato superi quello atteso dal puro rumore, corretto
per N tentativi, skew e kurtosi.

**Perché serve qui.** AGENTS.md riporta "0 su 36 configurazioni positive out-of-sample" e
"parametro migliore in train +0.241 → −0.654 out-of-sample" **senza** nessuna correzione per
i 36 tentativi. Il DSR renderebbe quel numero una probabilità, non un aneddoto.

**Costo:** basso. `scipy.stats.norm` + registro dei tentativi. Il registro **non esiste** ed è
il vero lavoro (30–50 righe + un file JSONL append-only).

**Cambierebbe:** il verdetto diventerebbe falsificabile in entrambe le direzioni. Oggi "non
passa" non dice *quanto* è improbabile che sia rumore.

**Riferimento operativo:** https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf

### 1.2 Probability of Backtest Overfitting (PBO) via CSCV
> **Bailey, Borwein, López de Prado, Zhu (2014).** *Pseudo-Mathematics and Financial
> Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance.*
> Notices of the AMS 61(5), 458–471.

**Idea.** Si partiziona la matrice (tempo × configurazioni) in S blocchi, si sceglie la
configurazione migliore su metà dei blocchi e se ne misura il rango out-of-sample sull'altra
metà. La frequenza con cui il "vincitore" finisce sotto la mediana OOS **è** il PBO.

**Perché serve qui.** Si applica direttamente alla griglia di 36 configurazioni già eseguita.
Non richiede di cambiare la strategia, solo di conservare i risultati per configurazione e
periodo (oggi vengono buttati).

**Costo:** medio. Serve rifare lo sweep tenendo la matrice completa.

### 1.3 Purged K-Fold CV ed embargo
> **López de Prado, M. (2018).** *Advances in Financial Machine Learning*, cap. 7 (purging,
> embargo) e cap. 12 (Combinatorial Purged CV). Wiley.

**Idea.** Quando le etichette si sovrappongono nel tempo (qui: un trade che dura fino a 72 h
su barre da 1 minuto), il training set contiene informazione del test set. Si *rimuovono* le
osservazioni di training il cui orizzonte di etichetta interseca il test, e si aggiunge un
embargo dopo il test.

**Perché serve qui.** Fricktrade usa split contigui (0–75% / 75–100%) e fold contigui. I trade
della finestra di training si chiudono **dentro** l'hold-out: la contaminazione è strutturale,
non solo quella del B-3. L'hold-out stesso è contaminato dai trade aperti prima del taglio.

**Costo:** basso-medio. Nell'harness: quando si taglia a `t`, scartare i trade aperti prima di
`t` e ancora aperti dopo `t`.

**Cambierebbe:** è probabile che peggiori i numeri già negativi — ed è esattamente il punto.

### 1.4 White's Reality Check e Hansen's SPA
> **White, H. (2000).** *A Reality Check for Data Snooping.* Econometrica 68(5), 1097–1126.
> **Hansen, P. R. (2005).** *A Test for Superior Predictive Ability.* JBES 23(4), 365–380.

**Idea.** Testano l'ipotesi nulla composita "nessun modello batte il benchmark", usando uno
**stationary bootstrap** (Politis & Romano 1994) sulla distribuzione del *massimo* statistico
tra tutti i candidati. L'SPA di Hansen è più potente perché studentizza e recentra.

**Perché serve qui.** È il test corretto per "ho provato 35 o 36 varianti e ho guardato la
migliore". Inoltre lo stationary bootstrap risolve il problema dei t-stat gonfiati da
sovrapposizione temporale, che l'harness oggi si limita a dichiarare in una nota a piè di
pagina.

**Costo:** medio (~150 righe). Esiste implementazione di riferimento:
`github.com/jkbaldwin/datasnooping`.

**Alternativa minima:** block bootstrap sull'expectancy per trade, con blocchi di lunghezza
pari all'holding period mediano. Sostituirebbe il `cv.std()/sqrt(n)` di
`cmd_signal` (riga 423), che assume indipendenza e non è un test di differenza fra medie.

### 1.5 Il benchmark giusto: buy-and-hold e cash
Già fatto (AGENTS.md: "cash ha battuto sia la strategia sia il mercato"). Da mantenere come
**criterio di accettazione**, non come nota: una strategia che non batte il buy-and-hold
aggiustato per il rischio non merita di girare, indipendentemente dal suo Sharpe.

---

## 2. Dimensionamento e rischio

### 2.1 Kelly con payoff asimmetrico
> **Kelly, J. L. (1956).** *A New Interpretation of Information Rate.* Bell System Technical
> Journal 35, 917–926. · **Thorp, E. O. (2006).** *The Kelly Criterion in Blackjack, Sports
> Betting, and the Stock Market.*

**Idea.** `f* = p − q/b`, dove `b` è il rapporto fra vincita media e perdita media. Con
`b < 1` (come qui: 0.12–0.19) il Kelly è **negativo** anche con `p = 0.84`. Il Kelly a frazione
ridotta (½ o ¼) si applica *dopo* aver verificato che `f* > 0`.

**Perché serve qui.** È il difetto A-1: il codice usa `(2p−1)/2`, valido solo per `b = 1`.
Con i numeri reali del sistema il Kelly corretto è −0.47 e −1.75: la dimensione ottima è zero.
Questa singola correzione è il candidato più forte per spiegare come un win rate dell'84%
abbia prodotto perdite.

**Costo:** basso. `PerformanceTracker` calcola già `avg_win` e `avg_loss`.

**Effetto collaterale da affrontare subito:** applicando la formula corretta, il sistema
smette di dimensionare e va **piatto**. È il comportamento giusto, ma va deciso
esplicitamente, non scoperto in produzione.

### 2.2 Win rate vs expectancy
> Corollario operativo della 2.1, formalizzato in **Vince, R. (1990).** *Portfolio Management
> Formulas.*

**Idea.** L'aspettativa matematica per trade è `p·W − q·L`. Calibrare un modello di
"confidenza" sul win rate (come fa `ConfidenceCalibrator`) ottimizza la metrica sbagliata e
**peggiora** il sizing quando `b < 1`.

**Perché serve qui.** A-2: il calibratore oggi è morto, ma se riparato ingenuamente
trasformerebbe un win rate dell'84% in una size più grande su un edge negativo. Va progettato
per stimare l'**expectancy**, non l'hit rate.

### 2.3 Vol targeting: come si fa bene
> Pratica standard post-2008; formalizzazione in **Moreira & Muir (2017).** *Volatility-Managed
> Portfolios.* Journal of Finance 72(4).

**Idea.** Si scala l'esposizione per `σ_target / σ_realizzata`, con la volatilità misurata
sulla **stessa** finestra del target. La volatilità è persistente, quindi la leva è
prevedibile anche quando il rendimento non lo è.

**Perché serve qui.** Il fix del 2026-08-19 (per-bar → daily via `sqrt(bars/day)`) è corretto.
Il problema residuo è che il target è **per-asset** e la scala è limitata a [0.3, 1.5]: in un
portafoglio di 20 altcoin correlate al 0.8, il rischio di portafoglio non è la media dei
rischi individuali. Il vol targeting per-asset dà una falsa sensazione di controllo.

**Costo:** basso (già quasi tutto presente). Il passo successivo vero è il vol targeting **di
portafoglio**, che richiede la matrice di covarianza — che `_portfolio_position_scale` già
calcola, male (30 barre, `np.cov` grezzo).

---

## 3. Esecuzione e microstruttura

### 3.1 Almgren–Chriss
> **Almgren, R. & Chriss, N. (2000).** *Optimal Execution of Portfolio Transactions.*
> Journal of Risk 3(2), 5–39.

**Idea.** Si minimizza `E[costo] + λ·Var[costo]` su una traiettoria di esecuzione; la
soluzione è una traiettoria `sinh` con `κ = sqrt(λσ²/η)` che interpola fra TWAP (λ→0) e
immediato (λ→∞).

**Perché serve qui.** `app/execution/smart_router.py:233-256` **implementa già** la formula, ma
(а) non è usata dal percorso live, (б) `sinh` può andare in `inf` senza cap su `κ`, (в) la
funzione di costo (`estimate_almgren_chriss_cost`) ha formule dimensionalmente sbagliate
(variance in dollari, impatto temporaneo diviso per la quantità totale). È un'implementazione
da correggere, non da scrivere.

**Rilevanza reale:** marginale a 500 $ di nozionale. Diventa rilevante solo sopra ~50k$.

### 3.2 Order Flow Imbalance
> **Cont, R., Kukanov, A., Stoikov, S. (2014).** *The Price Impact of Order Book Events.*
> Journal of Financial Econometrics 12(1), 47–88.

**Idea.** Su orizzonti brevi i prezzi sono guidati dallo sbilanciamento **dell'order flow al
best bid/ask** (arrivi e cancellazioni), non dal volume scambiato: `ΔP = β·OFI + ε`, con R²
del 65% medio e β inversamente proporzionale alla profondità.

**Perché serve qui.** `app/data/order_book.py` calcola già un `order_book_imbalance` dal
WebSocket di profondità Binance, ma (a) non scade mai (D-7), (b) è usato solo come uno dei tre
segnali della peak-detection exit. OFI è la variabile con la migliore evidenza empirica
disponibile **gratis** su Binance: usarlo come feature di timing dell'ingresso (non come
previsione di direzione a 1 minuto) è l'uso appropriato.

**Costo:** basso. Il dato c'è già. Serve: timestamp, scadenza, e una misura normalizzata per
profondità.

### 3.3 Maker vs taker: il conto economico
> Pratica di mercato; quantificazione nel contesto crypto in **Ignashkina, Rinne, Suominen
> (2022).** *Short-term reversals, returns to liquidity provision and the costs of immediacy.*
> Journal of Banking & Finance 138.

**Idea.** Il rendimento della *liquidity provision* è lordo di spread; il costo
dell'immediatezza è ciò che lo separa dal netto. Chi attraversa lo spread paga due volte
(spread + fee taker); chi lo guadagna incassa la fee maker *e* lo spread.

**Perché serve qui.** È il rimedio #1 del post-mortem (0.05% taker vs 0.02% maker su Binance,
~0.15%/round-trip) e la leva **non è implementata**: crypto è forzato a market order in tre
punti (`trader.py:2092`, `:2121`, `alpaca.py:154`), e per le equity il prezzo limite è messo
dal lato sbagliato del mid (C-11). Peggio, `SimBroker` ignora `order_type` (B-13): oggi è
**impossibile misurare** se la leva funzionerebbe.

**Costo:** medio. Richiede (a) ordini post-only, (b) un modello di fill nel backtest,
(c) gestione del rischio di non-esecuzione (adverse selection: chi non fillа è chi avrebbe
perso).

**Attenzione:** questo è l'unico punto della lista dove c'è un guadagno *economicamente*
plausibile e non solo epistemico. Ma il guadagno è al più ~0.15%/round-trip contro un edge
lordo misurato di ~0.17%: avvicina al pareggio, non lo supera.

### 3.4 Perché i 1-minute bar sul top-20 sono il posto peggiore
Il dato è il più arbitraggiato disponibile. La letteratura sul reversal intraday (Nagel 2012,
sotto) mostra che il rendimento della liquidity provision è **compensazione per rischio di
inventario**, e scompare quando il mercato è liquido. I 20 pair più liquidi sono esattamente
dove quel premio è più compresso.

---

## 4. Dove potrebbe esserci un edge (con l'evidenza contraria)

Questa sezione è deliberatamente scettica: ogni ipotesi è accompagnata da ciò che la
letteratura suggerisce sul fatto che possa sopravvivere ai costi.

### 4.1 Liquidity provision / short-term reversal
> **Nagel, S. (2012).** *Evaporating Liquidity.* Review of Financial Studies 25(7), 2005–2039.
> **Farag, Luo, Yarovaya, Zieba (2025).** *Returns from liquidity provision in cryptocurrency
> markets.* Journal of Banking & Finance 175.

**Idea.** I rendimenti delle strategie di reversal a brevissimo termine sono la compensazione
per fornire liquidità; sono prevedibili con misure di volatilità (VIX in Nagel) e crollano
quando la liquidità è abbondante.

**Evidenza crypto.** Il reversal equal-weighted a frequenza giornaliera in crypto è stato
misurato nell'ordine del 2.2% giornaliero lordo (p ≈ 0.000) in un campione ampio; gli stessi
autori riportano però che alcuni sotto-portafogli non mostrano un pattern monotono e che uno
di essi produce uno Sharpe leggermente negativo, e che il rendimento è più alto dove la
liquidità è più bassa. La lettura prudente: **il premio è più grande dove è più difficile
operare**, che è l'opposto del segmento in cui opera Fricktrade.

**Traduzione per Fricktrade.** La strategia MR attuale è *parente* di questa idea ma ne ha
perso i due ingredienti che la rendono pagata: (a) opera solo sul lato long, quindi non è
market-neutral e assorbe il beta del mercato; (b) usa market order, quindi **paga** la
liquidity provision invece di incassarla. Riformularla come market-making passivo su pair
liquidi è un sistema diverso, non un parametro.

### 4.2 Momentum cross-sectional
> **Liu, Y., Tsyvinski, A., Wu, X. (2022).** *Common Risk Factors in Cryptocurrency.*
> Journal of Finance 77(2), 1133–1177. · **Liu & Tsyvinski (2021).** *Risks and Returns of
> Cryptocurrency.* RFS 34, 2689–2727.

**Idea.** Tre fattori — mercato crypto, size e momentum — spiegano il cross-section dei
rendimenti crypto. Il momentum a 1–4 settimane genera ~2.5–4.1% settimanale long-short, e
funziona meglio nel gruppo a capitalizzazione sopra la mediana.

**Evidenza contraria nel repo.** AGENTS.md riporta che momentum/reversal cross-sectional a
24/72/168h e trend-following hanno perso dal −59% al −98% in training. Ma il test è stato fatto
su **20 simboli**: il fattore momentum di Liu-Tsyvinski è costruito su 1.700+ coin e sorting a
quintili. Con 20 simboli non c'è un cross-section, c'è un paniere. **Il test negativo non
confuta la letteratura: misura un'altra cosa.**

**Azione sensata:** se si riapre il tema, serve un universo di almeno 150–200 coin con volume
sufficiente, non i 20 più liquidi. È un esperimento da 1–2 giorni di lavoro sui dati, non mesi.

### 4.3 Time-series momentum
> **Moskowitz, T., Ooi, Y. H., Pedersen, L. H. (2012).** *Time Series Momentum.*
> Journal of Financial Economics 104(2), 228–250. · Per crypto: Liu & Tsyvinski (2021), che
> trovano momentum time-series robusto a orizzonte settimanale.

**Idea.** Il rendimento passato di un asset predice il proprio rendimento futuro a 1–12 mesi;
l'effetto è presente in 58 mercati su 58 studiati, e in crypto a orizzonte settimanale.

**Perché serve qui.** È l'unica ipotesi con un effetto documentato **e** compatibile con un
conto piccolo long-only: si può implementare come filtro di regime ("essere long solo quando
il trend a N giorni è positivo") invece che come strategia di ingresso. Costo: quasi zero,
nessun cambio di infrastruttura.

**Test onesto:** sui dati `history_1h` (3 anni, 20 simboli) è misurabile in un'ora con
l'harness esistente, aggiungendo `--trend-filter-bars` (che **esiste già** come parametro,
riga 86 e 543 di `alpha_validation.py`, e non è mai stato usato nei run documentati).

### 4.4 Funding-rate carry
Già misurato nel repo: 3.9% lordo annuo medio, 2–3% netto, contro un rischio di liquidazione
su due gambe. La conclusione del post-mortem ("respinto per dimensione, non per validità") è
corretta e **confermata dalla letteratura**: è un premio reale che richiede scala.

---

## 5. Architetture di riferimento

### 5.1 NautilusTrader — parità ricerca↔produzione
> https://nautilustrader.io/docs/latest/concepts/overview/

**L'idea da copiare, indipendentemente dal linguaggio.** Il `NautilusKernel` è condiviso fra
`BacktestEngine` e `LiveNode`; l'unica sostituzione avviene al confine
`DataClient`/`ExecClient`. Il `SimulatedExchange` ha `FillModel`, `FeeModel`, `LatencyModel`
separati e configurabili. Conseguenza: **la strategia che passa il backtest è il codice che va
in produzione, senza riscritture.**

**Perché è il singolo spunto architetturale più importante.** Fricktrade ha tre implementazioni
della stessa strategia (live, `agent_engine`, `alpha_validation`) e il gate che decide se
rischiare denaro è agganciato a quella che **non** è il codice di produzione. `agent_engine`
usa già il vero `TradingAgent`: la strada è aperta, basta spostare il gate lì e completare
`SimBroker` (fee model, fill model, vincolo di cassa).

**Costo:** basso per il gate (il motore c'è), medio per un `SimBroker` decente.
**Non serve riscrivere in Rust.** Serve un solo motore con un solo modello di fill.

### 5.2 Hummingbot — market making
> https://hummingbot.org/ | Apache 2.0, ~21k stelle.

**Cosa prendere.** Il framework di market making (inventario, spread, adverse selection,
gestione del rischio di non-fill) è la realizzazione ingegneristica di Avellaneda-Stoikov.
Se mai si esplorasse la §4.1, questo è il punto di partenza, non un riadattamento della
strategia MR.

### 5.3 Freqtrade — FreqAI e dry-run
**Cosa prendere.** (a) Il ciclo `dry-run` come modalità di prima classe, con lo stesso motore
del live; (b) la separazione netta fra strategia e gestione dell'esecuzione; (c) il comando
`lookahead-analysis` che verifica automaticamente il look-ahead bias di una strategia. Quest'
ultimo è un'idea che varrebbe la pena copiare: uno script che, per ogni strategia, confronta
i segnali calcolati su un dataset completo con quelli calcolati barra per barra. Fricktrade ha
il problema descritto in B-8 e non ha nessun test automatico per rilevarlo.

---

## 6. Tabella di trasferibilità

| Spunto | Dove si aggancia | Sforzo | Guadagno atteso | Rischio |
|---|---|---|---|---|
| Kelly asimmetrico (2.1) | `trader._size_order:2827` | 1 h | **elimina la causa meccanica delle perdite** | il sistema va piatto — corretto, ma da accettare |
| Registro dei tentativi + DSR (1.1) | nuovo, `scripts/` | 1 g | rende il gate falsificabile | nessuno |
| Purged split + embargo (1.3) | `alpha_validation.slice_frames` | 2 h | numeri onesti (peggiori) | nessuno |
| Min campioni + block bootstrap (1.4) | `alpha_validation.cmd_holdout` | 3 h | evita PASS da 3 trade | nessuno |
| Gate sul motore vero (5.1) | `agent_engine.py` | 3 g | il gate misura ciò che gira | richiede un `SimBroker` decente |
| Maker orders + fill model (3.3) | `trader:2092`, `SimBroker` | 1 sett | ~0.15%/round-trip | non-esecuzione, adverse selection |
| OFI come timing (3.2) | `order_book.py` + `entry_ranking` | 2 g | ignoto, plausibile | può peggiorare (segnale già usato male) |
| Filtro di trend TS-momentum (4.3) | `emit --trend-filter-bars` (già c'è) | 1 h | ignoto, testabile gratis | nessuno, è già implementato |
| Universo ampio per il cross-section (4.2) | nuovo dataset | 1–2 g | ignoto | richiede dati che non ci sono |
| Market making (5.2) | sistema nuovo | settimane | ignoto | è un altro progetto |

**Le prime cinque righe valgono più delle altre cinque insieme**, e sono tutte misura e
correttezza, non alpha. È coerente con la conclusione del post-mortem: il problema non è che
mancasse una strategia, è che il sistema non era in grado di distinguere una strategia che
funziona da una che non funziona.

---

## 7. Ciò che la letteratura sconsiglia esplicitamente

- **Aumentare il numero di tentativi senza contabilizzarli.** Bailey et al. dimostrano che con
  abbastanza tentativi *qualsiasi* Sharpe è raggiungibile. Il repository ha già 36 + 35 + 30
  tentativi non contati.
- **Ottimizzare i parametri di uscita separatamente dall'ingresso.** Il ladder attuale
  (TP +1%/+5%, stop −5%/−6%) è ottimizzato per massimizzare il win rate, che è la metrica
  sbagliata. Con `b = 0.12`, un TP a +1% e uno stop a −6% richiede un win rate > 85.7% solo
  per andare in pareggio, prima dei costi.
- **Fidarsi di una metrica di attività.** "84% win rate", "trades_total", "137.508 segnali":
  nessuna di queste è una misura di edge. Il `TRADE_COST_BPS` che servirebbe non è nemmeno
  registrato (C-13).
- **Assumere che un risultato negativo su 20 simboli confuti un fattore documentato su 1.700.**
  Vale in entrambe le direzioni: né conferma né confuta.
