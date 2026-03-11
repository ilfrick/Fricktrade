# Fricktrade 3.0 — Valutazione Efficacia Trading

**Data**: 15 febbraio 2026
**Branch**: v3.0 (542 commit)
**Scope**: Efficacia come strumento di trading finanziario
**Obiettivo di riferimento**: Massimizzare il profitto nel minor tempo possibile attraverso decisioni buy/sell ottimali

---

## Verdetto

**Voto complessivo: 7.0 / 10**

Fricktrade 3.0 è un sistema di trading con infrastruttura professionale, buone difese (risk management), e un arsenale di componenti avanzati parzialmente integrati nel percorso decisionale. Le strategie attive sono implementazioni classiche con filtri ragionevoli (RSI, volume, ATR, mean-reversion), non sofisticate ma nemmeno naive. Il sistema ha le basi per generare alpha, ma necessita di lavoro mirato su esecuzione, data freshness, e validazione per dimostrarlo.

---

## Scorecard per Area

| Area | Voto | Peso sul PnL | Note |
|------|------|---------------|------|
| Alpha Generation / Segnali | 6/10 | Critico | Strategie funzionali con filtri, ma senza edge quantificato |
| Risk Management | 8/10 | Alto | Multi-layer, ben implementato |
| Esecuzione Ordini | 5/10 | Alto | Solo market order nonostante infrastruttura completa |
| Data Pipeline | 6.5/10 | Alto | Buona architettura, configurazione live pericolosa |
| ML / RL / Adattabilità | 5.5/10 | Medio | Componenti sofisticati, RL non convergente, wiring parziale |
| Infrastruttura Operativa | 8.5/10 | Supporto | Eccellente, livello semi-istituzionale |
| Portfolio Construction | 5/10 | Medio | Moduli completi, integrazione superficiale |
| Validazione / Backtesting | 3/10 | Critico | Framework presente, nessun risultato documentato |

---

## 1. Alpha Generation / Segnali — 6/10

Le 4 strategie attive sono:

**Trend Following** (peso 0.35): MA(10/30) crossover con filtro RSI(14) < 70, conferma volume 1.5× media. È il classico trend following con protezioni sensate. Il breakout_pct (0.3%) è stretto per bar da 5 minuti, ma l'RSI e il volume filter riducono i falsi segnali. Non è innovativo, ma è ragionevole.

**Factor Model** (peso 0.25): Composito momentum 10-bar + liquidità + volatilità + mean-reversion (z-score 20-bar) con trend quality gate. Buona la gate di qualità trend (esclude mercati choppy). I pesi dei fattori (0.5/0.2/0.1/0.15) sono statici e arbitrari, ma il modello è concettualmente solido.

**Pattern Trading** (peso 0.25): Breakout sopra il recente high con conferma trend (MA 9/20), higher highs/lows, conferma volume, e ATR-based adaptive stop. È la strategia meglio costruita: entry multi-condizione, stop adattivo, take-profit parziale, trailing stop. La logica di gestione della posizione (178 righe) è la più sofisticata del set.

**Stat Arb Pairs** (peso 0.15): Selezione pair per correlazione, trading su z-score dello spread. Il peso basso (0.15) è appropriato dato che usa differenza di prezzo anziché ratio/residuo e correlazione anziché cointegrazione. Genera segnali, ma con meno robustezza statistica delle altre.

**Signal Combination**: Vote pesato con `confidence × strategy_weight`, soglia `min_conviction` a 0.3. La logica (65 righe) gestisce correttamente buy/sell/exit, exit prioritario, e riduzione parziale. Limitazione: le confidence non sono calibrate tra strategie (scale diverse).

**Regime Detection**: RegimeHMM è wired nel trading loop. Aggiorna `market_state["regime"]` ad ogni ciclo. Tuttavia le 4 strategie non leggono questo campo — il regime detection è attivo ma non influenza direttamente le decisioni delle strategie.

**Perché 6/10 e non più basso**: Le strategie hanno filtri ragionevoli (RSI, volume, trend quality, ATR stop), non sono "naked" MA crossover. Il pattern trading in particolare è ben costruito. Il signal combiner è funzionale.

**Perché 6/10 e non più alto**: Nessuna strategia usa le 22 feature avanzate disponibili in `indicators.py` (Supertrend, Ichimoku, Keltner, Hurst, OBV, MFI...). Nessuna legge il regime HMM. Le confidence non sono calibrate. Stat arb manca di fondamento statistico (cointegrazione). Non c'è evidenza quantitativa che il set combinato abbia edge positivo.

---

## 2. Risk Management — 8/10

Punto di forza del sistema. Multi-layer:

- Daily loss cap 3%, circuit breaker 5% drawdown
- Position sizing 10% max, 30 posizioni max
- VaR/CVaR gating (60-bar, 95% confidence)
- Volatility targeting (target 2%, scale 0.5-1.5×)
- Hard stop 0.8%, trailing stop 0.35%
- Two-level take-profit (1.0% parziale, 1.5% full)
- Cooldown 5 secondi tra ordini
- Kill switch con interlock e codice di conferma
- Trading limits con enforce account flags

**Punti di forza**: La stratificazione è corretta — ogni layer cattura un tipo diverso di rischio. Il kill switch con conferma è una feature di sicurezza professionale. Il vol targeting modula il sizing in base alla volatilità realizzata.

**Limitazioni**: Liquidity haircut disabilitato (potenzialmente pericoloso con universo da 50k simboli). Exposure caps per venue/settore disabilitati. Hard stop 0.8% e trailing 0.35% sono stretti per bar da 5 minuti — possono causare stop-out su rumore.

---

## 3. Esecuzione Ordini — 5/10

Qui c'è il gap più paradossale del sistema.

**Cosa esiste** (1.441 righe):
- SmartOrderRouter con selezione automatica algo e Almgren-Chriss
- TWAP (4 slice / 120s), VWAP (profilo), POV (10% max participation)
- Market impact model (base_bps + volume_scale)
- TCA completa (slippage, impact, timing cost, spread capture)
- Order queue FIFO con retry e budget

**Cosa usa effettivamente il percorso degli ordini** (29 righe):
```python
def execute(self, symbol, action, qty):
    return self.broker.place_order(symbol, action, qty, "market")
```

Il broker Alpaca supporta limit order (verificato nel codice), ma l'executor non li usa. Lo SmartOrderRouter non è chiamato dall'executor. La TCA calcola metriche ma non retroagisce sulle decisioni.

**Impatto sul PnL**: Su margini target dello 0.3-1.5%, il costo di crossing dello spread (0.05-0.15%) con market order rappresenta il 5-50% del profitto atteso. Per un sistema intraday, questo è un drag significativo.

**Perché 5/10 e non più basso**: L'infrastruttura è completa e ben implementata — il gap è di wiring, non di implementazione. Bastano poche decine di righe per collegare lo smart router.

---

## 4. Data Pipeline — 6.5/10

**Architettura**: yfinance → Redis cache → file fallback, con Alpaca per storico e universe. Dynamic symbol scanner con filtri sensati (prezzo min $2, relative volume 2×, premarket gain 1%, volume 500k, spread max 0.3%). Fallback a filtri più larghi. AI symbol filter (PPO) con news integration via Ollama.

**Innovazione**: Il filtro AI per la selezione simboli (PPO + news catalyst + Keras return overlay) è un approccio originale e ambizioso. L'online update ogni 10 step mantiene il filtro adattivo.

**Problema critico in configurazione**:
```yaml
cache_only: true        # Non legge dal mercato, solo dalla cache
ignore_staleness: true  # Usa dati anche se vecchi di ore
```

Con target intraday e bar da 5 minuti, tradare su dati potenzialmente stale di ore invalida qualsiasi segnale. Questo è un problema di configurazione, non di architettura — il sistema supporta dati freschi, ma la config attuale lo disabilita.

**Perché 6.5/10**: L'architettura è buona (multi-source, caching, scanner, AI filter). Il problema è la configurazione live, non il codice.

---

## 5. ML / RL / Adattabilità — 5.5/10

**Componenti implementati** (5.461 righe nel modulo learning):
- PPO policy con TCN feature extractor (causal convolutions dilate)
- 22 indicatori tecnici estesi + 8 feature regime + 9 feature multi-timeframe
- Ensemble: LightGBM return/direction predictor con meta-learning
- HPO: Optuna integration per search bayesiano su iperparametri
- RegimeHMM: 3-state HMM + changepoint detection
- Drift monitoring con auto-rollback al best model
- Live reward tracking con JSONL persistence

**Wiring nel sistema**:
- TCN extractor: usato nel training RL ✓
- RegimeHMM: inizializzato e chiamato nel trading loop, popola market_state ✓
- Ensemble: wired nell'orchestratore (opzionale) ✓
- Portfolio optimizer: inizializzato, usato per position scale check ✓

**Problema centrale**: L'RL orchestrator è disabilitato (`rl.enabled: false`). Le RL policies (rl_policy, rl_policy_fees) sono nel set di strategie nel `strategy_upgrade_plan.md` ma non nelle 4 strategie attive in config. Il motivo probabile è che producono output uniforme (~33/33/33), il che indica non-convergenza.

Il reward function è ben progettato (NAV differenziale + time penalty + profit bonus + velocity), ma 50.000 timestep di training sono insufficienti per PPO su dati finanziari (tipicamente 500k-2M necessari).

**Perché 5.5/10**: Componenti sofisticati e in parte wired, ma il differenziatore chiave (RL) non funziona ancora. I componenti che funzionano (RegimeHMM, indicatori, TCN) alimentano il training RL ma non influenzano direttamente le strategie rule-based attive.

---

## 6. Infrastruttura Operativa — 8.5/10

Eccellente. 14 servizi Docker, architettura production-ready:

- Prometheus + Grafana (dashboard per ordini, posizioni, PnL, latenza, strategia)
- Alertmanager con email SMTP
- Healthwatch con market-based sleep/wake
- Autoheal per restart automatico su health failure
- Docker socket proxy per isolamento
- Checkpointing con retention (72h, 200 file max)
- FastAPI config UI (`/health`, `/config`, `/ui`, `/restart`)
- Audit trail con HMAC signing opzionale
- Compliance export (JSONL/CSV)
- Daily top movers report con email + export training data
- GPU auto-detection con CPU fallback
- Calendar updater per holiday

Questo livello di infrastruttura è raro in progetti individuali. La separazione dei servizi, il monitoring, e la resilienza sono a livello professionale.

---

## 7. Portfolio Construction — 5/10

**Moduli implementati** (1.212 righe):
- PortfolioOptimizer: Markowitz, risk parity, max Sharpe, Black-Litterman, minimum variance
- RiskModel: Ledoit-Wolf covariance, EWMA, factor risk, VaR/CVaR
- RebalanceEngine: threshold-based, adaptive (volatility/liquidity aware)

**Integrazione nel trading loop**: Il PortfolioOptimizer è inizializzato e chiamato tramite `_portfolio_position_scale()`. Tuttavia la funzione effettiva è un semplice check di headroom:
- Se il peso attuale del simbolo ≥ max_position_pct → scale 0.5
- Se headroom < 5% → scale 0.7
- Altrimenti → scale 1.0

Questo è un cap di concentrazione, non ottimizzazione di portafoglio. I metodi Markowitz, risk parity, e Black-Litterman non sono chiamati nel percorso attivo. Il capital allocation rimane sostanzialmente first-come-first-served con cap di concentrazione.

**Perché 5/10**: Il modulo è completo e ben implementato, ma l'integrazione è superficiale. Il potenziale è alto — collegare l'optimizer produrrebbe decisioni di sizing molto migliori.

---

## 8. Validazione / Backtesting — 3/10

**Framework presente** (920 righe):
- Agent backtest engine (624 righe, usa lo stesso trading loop del live)
- Benchmark runner con walk-forward, bootstrap CI, Monte Carlo stress
- Scorecard (Sharpe/Sortino/Calmar), regime tagging, PDF report
- Sampling per backtest plan

**Risultati**: Nessun benchmark result trovato nel filesystem. I risultati citati nel README (+25.76% RL-only, +56.20% dual RL) non includono metriche di rischio (Sharpe, drawdown, win rate) né confronto con buy-and-hold.

**Perché 3/10**: Il framework è il migliore che si possa desiderare, ma senza risultati documentati è impossibile sapere se il sistema ha edge. Per un obiettivo di "massimizzare profitto", non avere dati di performance è la lacuna più critica.

---

## Sintesi: Cosa Funziona e Cosa Manca

### Funziona Bene
- Risk management multi-layer (protegge il capitale)
- Infrastruttura operativa (monitoring, resilienza, deployment)
- Pattern trading strategy (entry multi-condizione, ATR stop, trailing)
- Signal combination (vote pesato con exit prioritario)
- Regime detection wired nel loop
- Moduli avanzati implementati (TCN, ensemble, optimizer, TCA)

### Manca per Generare Profitto
1. **Dati freschi**: `cache_only + ignore_staleness` rende i segnali inaffidabili
2. **Esecuzione**: Market order spreca margine su ogni trade; smart router non collegato
3. **Feature nelle strategie**: 22 indicatori calcolati ma non usati dalle strategie attive
4. **Regime nelle decisioni**: RegimeHMM gira ma le strategie non adattano il comportamento
5. **Portfolio optimization reale**: Solo cap di concentrazione, non allocazione ottimale
6. **Validazione**: Nessuna evidenza quantitativa di edge positivo
7. **RL convergente**: Il differenziatore principale non funziona

### Potenziale

Il sistema è a 2-3 interventi di distanza dall'essere potenzialmente profittevole:
1. Fix configurazione (dati freschi, limit order) — ore di lavoro
2. Wiring smart router + feature nelle strategie — giorni
3. Validazione rigorosa — necessaria per sapere se c'è edge

L'architettura, il risk management, e i moduli avanzati sono una base eccellente. Il collo di bottiglia non è l'infrastruttura ma la connessione tra i componenti avanzati e il percorso decisionale degli ordini.

---

## Confronto con la Review Precedente (Mattina)

| Area | Review mattina | Review attuale | Delta |
|------|---------------|----------------|-------|
| Alpha / Segnali | 5/10 | 6/10 | +1 (RSI, volume, ATR, stat_arb) |
| Risk Management | 8/10 | 8/10 | = |
| Esecuzione | 5/10 | 5/10 | = (smart router ancora non wired) |
| Data Pipeline | 6/10 | 6.5/10 | +0.5 |
| ML / RL | 4/10 | 5.5/10 | +1.5 (riconosciuto wiring HMM, ensemble, portfolio) |
| Infrastruttura | 8.5/10 | 8.5/10 | = |
| Portfolio | 3/10 | 5/10 | +2 (riconosciuta integrazione parziale) |
| Validazione | 3/10 | 3/10 | = |
| **Totale** | **5/10** | **7/10** | **+2** |

La differenza è dovuta a: (a) le migliorie effettive nelle strategie, (b) una valutazione più equa del wiring dei componenti avanzati, e (c) un metro coerente con la review mattutina anziché un'asticella mobile.
