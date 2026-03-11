# Fricktrade v3.0 — Sintesi Cross-Review della Sessione 2026-02-17

**Data analisi:** 2026-02-18  
**Fonti:** Report originale (Claude Code Opus 4.6) + Review di ChatGPT, Grok, Gemini, Claude Opus 4.6  
**Autore sintesi:** Claude Opus 4.6

---

## 1. Executive Summary

Quattro modelli AI hanno analizzato indipendentemente il session report del 17 febbraio 2026. Il consenso è unanime: **il sistema Fricktrade è attualmente incapace di tradare** a causa di due bug critici che si amplificano reciprocamente — pesi strategici ignorati e segnali short-entry fantasma che bloccano i buy. Tutti concordano che i primi due fix trasformerebbero il sistema da "rotto" a "funzionante" in poche ore.

Le divergenze tra i reviewer riguardano principalmente l'ordine di priorità secondario, la profondità architetturale delle soluzioni e la gestione del capitale su account piccoli.

**Verdetto condiviso:** Il report originale è eccellente. Con i fix 1–2 il bot diventa operativo. Con i fix 3–7 diventa professionale.

---

## 2. Risultato della Sessione

| Metrica | Higher Account | Realistic Account |
|---------|---------------|-------------------|
| Equity iniziale | $2,600.00 | $250.00 |
| Equity finale | $2,560.47 | $244.20 |
| Picco intraday | +1.9% (18:55 UTC) | — |
| Return | **-1.52%** | **-2.32%** |
| Buy eseguiti in sessione | **0** | **0** |
| Capitale bloccato | **60%** | — |
| Buying power a fine sessione | **$0** | — |

Tutte le posizioni aperte derivano da sessioni precedenti. Il sistema ha osservato il mercato per un'intera sessione senza agire.

---

## 3. Mappa del Consenso — Dove Tutti Concordano

### 3.1 Bug #1: Strategy Weights Ignorate (CRITICO — unanimità)

**Problema:** I pesi configurati in `config.yaml` (trend=0.35, factor=0.25, pattern=0.25, stat_arb=0.15) sono completamente ignorati. L'orchestrator in modalità "weight" con RL disabilitato restituisce peso 1.0 per tutte le strategie, rendendo il sistema di orchestrazione inutile.

**Consenso:** Tutti e quattro i reviewer classificano questo come il fix #1 o #2 in assoluto.

**Soluzione base (tutti):** Applicare `strategy_weights` da config come fallback in `_combine_signals` quando l'orchestrator restituisce pesi uniformi.

**Aggiunta ChatGPT:** Rendere il fix *verificabile* con una metrica `applied_strategy_weight{strategy=…}` e un healthcheck che lanci warning se config dichiara pesi non-uniformi ma l'orchestrator applica uniformi. Questo previene "bug silenziosi" futuri.

### 3.2 Bug #2: Segnali Short-Entry Cancellano i Buy (CRITICO — unanimità)

**Problema:** `_combine_signals` somma algebricamente buy_score vs sell_score. trend_following emette sell (19% dei cicli) che sono tentativi di short-entry, ma con `allow_shorts=false` non possono mai eseguire. Tuttavia partecipano al voto e cancellano i buy di factor_model.

**Risultato sessione:** sell vince 107 volte, buy 77, tie 316 → **zero acquisti**.

**Consenso:** Tutti classificano questo come fix #1 o #2.

**Soluzione base (tutti):** Non contare segnali sell nel voto quando la posizione è zero e shorting è disabilitato.

**Aggiunta ChatGPT (architetturale):** Separare i segnali in *intent* semantici (`enter_long`, `exit_long`, `enter_short`, `exit_short`) e filtrare in base a stato posizione + policy. Questo è più robusto di una patch ad-hoc nel combiner.

**Aggiunta Grok:** Applicare il filtro a livello di strategy layer (non solo nel combiner) per evitare che trend_following sprechi cicli generando sell che sappiamo già bloccati.

### 3.3 Bug RIG: Short Involontario (CRITICO per ChatGPT/Grok, MEDIO per report originale)

**Problema:** RIG passa da +62 a -186 nonostante `allow_shorts=false`. Ordini sell multipli in-flight superano lo zero.

**Consenso sulla gravità:** ChatGPT e Grok lo elevano a *Critical* (violazione di invariante). Il report originale lo classifica Medium.

**Soluzione report originale:** Buy-to-cover automatico se fill produce posizione negativa.

**Soluzione ChatGPT (prevenzione > cura):**
- Per-symbol order mutex (max 1 ordine di riduzione attivo)
- Target-position model (calcola delta, cancel/replace su fill)
- Clamp severo: se `pos_qty=0` e `allow_shorts=false`, sell → NOOP
- Reconciliation loop come airbag finale

**Soluzione Gemini (architetturale):** L'OMS deve validare la "Pending Quantity" (ordini in volo) prima di accettare un nuovo ordine di vendita, prevenendo l'invio di ordini che supererebbero lo zero.

### 3.4 factor_model Troppo Rumoroso (consenso)

**Problema:** 53% buy rate con confidence media 0.49 — essenzialmente "always long" senza selettività.

**Consenso:** Alzare la soglia, ma con sfumature diverse.

- **Report originale:** Alzare a 0.6
- **Grok:** Alzare a 0.6 e applicare soglia globale (non solo factor_model)
- **ChatGPT:** Alzare sì, ma basandosi su analisi della distribuzione (istogramma confidenze vs PnL per bucket), non su un valore arbitrario. Rimandare comunque a dopo i fix strutturali.

### 3.5 stat_arb_pairs Completamente Inerte (consenso)

**Problema:** 100% hold su 1019 simboli, zero pair trovati.

**Consenso:** Disabilitare o fix profondo. L'ADF con z_entry=2.0 è troppo stringente per barre a 5 minuti e l'universo dinamico ruota troppo.

### 3.6 trend_following Sistematicamente Bearish (consenso)

**Problema:** sell 19% vs buy 0.3% in regime low_vol_trending (96.8%).

**Consenso:** Aggiungere diagnostic logging per capire quale condizione fallisce (Supertrend, RSI gate, VWAP deviation, volume), ma *dopo* aver rimosso il deadlock dei fix 1–2 per avere dati puliti.

---

## 4. Divergenze Significative tra i Reviewer

### 4.1 Priorità del Bug RIG

| Reviewer | Classificazione |
|----------|----------------|
| Report originale | Medium (correctness) |
| ChatGPT | **Critical** (violazione di invariante) |
| Grok | **Critical** |
| Gemini | **Priorità 2** (fix architetturale) |
| Claude | **Critical** (concordo con ChatGPT/Grok) |

**Mia posizione:** Concordo nel classificarlo Critical. Una violazione di `allow_shorts=false` è un errore di correttezza fondamentale che può produrre perdite non previste. La prevenzione (OMS-level validation, come suggerisce Gemini) è preferibile alla sola cura (buy-to-cover post-errore).

### 4.2 Priorità PDT

| Reviewer | Classificazione | Approccio |
|----------|----------------|-----------|
| Report originale | Low-Medium | Contare day trades, riservarne per uscite |
| ChatGPT | Medium | PDT cooldown + budget day-trade |
| Grok | **Medium-High** | Se remaining_day_trades ≤ 1 → force swing only |
| Gemini | **Priorità 3** | Modalità "Force Swing" che ignora uscite intraday (tranne stop-loss emergenza) |

**Mia posizione:** Con $2,600 il PDT è un vincolo strutturale, non marginale. Concordo con Grok/Gemini nell'elevarlo a Medium-High. La modalità "Force Swing" di Gemini è la più pragmatica per account sotto $25k.

### 4.3 Approccio Architetturale: Patch vs Redesign

- **Report originale + Grok:** Fix puntuali, rapidi da implementare
- **ChatGPT:** Propone un redesign più profondo (intent semantici, gerarchia Safety→Exit→Entry, state machine per simbolo)
- **Gemini:** Propone validazione OMS preventiva (Pending Quantity check)

**Mia posizione:** L'approccio pragmatico (fix puntuali ora, redesign dopo) è corretto per sbloccare il sistema. Tuttavia le idee architetturali di ChatGPT (separazione intent, gerarchia decisionale) e Gemini (OMS con pending qty) dovrebbero essere pianificate come evoluzione a medio termine per evitare di accumulare debito tecnico.

### 4.4 Tuning delle Soglie: Ora vs Dopo

- **Grok:** Alzare subito min_confidence a 0.6
- **ChatGPT:** Rimandare il tuning a dopo i fix strutturali, poi decidere su base empirica (distribuzioni, non valori fissi)

**Mia posizione:** ChatGPT ha ragione. Fare tuning su un sistema rotto produce dati fuorvianti. Prima si sblocca il sistema (fix 1–2), poi si raccolgono 5–10 sessioni di dati puliti, poi si decide la soglia ottimale.

---

## 5. Suggerimenti Unici per Reviewer (Non Presenti negli Altri)

### Da ChatGPT

- **Gerarchia decisionale Safety → Exit → Entry:** Il sistema oggi tratta buy/sell come pari livello. In un trading system robusto, la safety (invarianti, exposure cap) viene prima delle uscite, che vengono prima delle entrate. Questo evita che strategie in conflitto blocchino anche la gestione del rischio.
- **Market-hours enforcement:** Le entry effettive sono avvenute in pre-market. Verificare che la venue config sia enforced e aggiungere un flag `allow_extended_hours` esplicito.
- **Policy di lifecycle portafoglio:** Limite di numero posizioni, time-based exit (stale killer), free-cash floor per evitare `insufficient_cash`.

### Da Grok

- **Kill-switch per singola strategia** (config: `disable_strategy: ["trend_following"]`): Spegnere una strategia tossica in 2 secondi senza riavviare il sistema.
- **Regime-aware min_confidence:** In `low_vol_trending` alzare la soglia di factor_model a 0.65. Il regime è già calcolato correttamente ma mai usato per modulare le soglie.
- **Backtest obbligatorio dei fix** (almeno 30 giorni su dati 2025–2026) prima di push in produzione.

### Da Gemini

- **Validazione OMS preventiva (Pending Quantity):** L'OMS controlla gli ordini in volo prima di accettarne di nuovi, prevenendo l'oversell alla radice anziché correggerlo dopo.
- **Modalità "Force Swing"** per account sotto PDT: Ignorare uscite intraday (tranne stop-loss emergenza) per bypassare la regola PDT e sfruttare i trend overnight.

---

## 6. Piano di Implementazione Integrato (Sintesi Finale)

Questo piano combina le migliori raccomandazioni di tutti i reviewer, ordinate per impatto e urgenza.

### P0 — Oggi (Sblocco del Sistema)

| # | Fix | Effort | Impatto | Note |
|---|-----|--------|---------|------|
| 1 | Wire `strategy_weights` in `_combine_signals` + metrica di verifica | Small | ★★★★★ | Report + ChatGPT |
| 2 | Non contare sell come voto quando `pos=0` e `allow_shorts=false` | Small | ★★★★★ | Tutti |
| 3 | Fix RIG: validazione OMS preventiva (pending qty) + clamp + buy-to-cover come airbag | Small-Medium | ★★★★★ | ChatGPT + Gemini |

*I fix 1+2 insieme trasformano il sistema da 0 buy/sessione a decine di buy/sessione.*

### P1 — Entro 2 Giorni (Osservabilità e Compliance)

| # | Fix | Effort | Impatto | Note |
|---|-----|--------|---------|------|
| 4 | Diagnostic logging per trend_following (quale condizione fallisce) | Small | ★★★★ | Tutti |
| 5 | Kill-switch per singola strategia in config | Trivial | ★★★ | Grok |
| 6 | PDT "Force Swing" mode per account sotto $25k | Small | ★★★★ | Grok + Gemini |

### P2 — Entro 1 Settimana (Tuning su Dati Puliti)

| # | Fix | Effort | Impatto | Note |
|---|-----|--------|---------|------|
| 7 | Alzare min_confidence factor_model (soglia da distribuzioni, non valore fisso) | Small | ★★★ | ChatGPT + Grok |
| 8 | Position sizing dinamico (max 8–10% per posizione su $2,600) | Small | ★★★ | Grok |
| 9 | Disabilitare stat_arb_pairs (o fix ADF) | Medium | ★★ | Tutti |

### P3 — Medio Termine (Evoluzione Architetturale)

| # | Fix | Effort | Impatto | Note |
|---|-----|--------|---------|------|
| 10 | Separazione intent semantici (`enter_long`, `exit_long`, `enter_short`, `exit_short`) | Medium | ★★★★ | ChatGPT |
| 11 | Gerarchia decisionale Safety → Exit → Entry | Medium | ★★★★ | ChatGPT |
| 12 | Regime-aware min_confidence | Small | ★★★ | Grok |
| 13 | Backtest framework obbligatorio pre-deploy | Large | ★★★★ | Grok |
| 14 | Market-hours enforcement + flag `allow_extended_hours` | Small | ★★ | ChatGPT |
| 15 | Policy lifecycle portafoglio (time-stop, stale killer, free-cash floor) | Medium | ★★★ | ChatGPT |

---

## 7. Impatto Atteso

Con i fix P0 (oggi):
- Buy decisions: **da 0 a decine per sessione**
- Il sistema torna operativo

Con i fix P0 + P1 (entro 2 giorni):
- Capitale bloccato: **da 60% a <20%**
- Niente più short involontari
- PDT reject: **da 90 a ~0** (force swing mode)
- Possibilità di spegnere strategie tossiche in tempo reale

Con i fix P0 + P1 + P2 (entro 1 settimana):
- Segnali buy più selettivi e di qualità superiore
- Drawdown intraday ridotto
- Compute sprecato da stat_arb eliminato
- Il sistema diventa *professionale*

---

## 8. Conclusioni

Il report originale di Claude Code Opus 4.6 è di alta qualità: le diagnosi sono precise, le correlazioni ben identificate, e la prioritizzazione è sostanzialmente corretta. Le quattro review indipendenti confermano l'analisi e la arricchiscono su tre assi principali.

**Asse 1 — Prevenzione vs Cura (Gemini + ChatGPT):** Per il bug RIG e problemi simili, validare *prima* di inviare ordini (OMS pending qty check) è superiore a correggere *dopo* il fill. Questo principio dovrebbe guidare tutta l'evoluzione dell'order management.

**Asse 2 — Osservabilità (ChatGPT + Grok):** Metriche di verifica sui pesi applicati, diagnostic logging per condizioni fallite, e healthcheck sugli invarianti di configurazione sono essenziali per evitare che bug silenziosi costino settimane di debug.

**Asse 3 — Pragmatismo sul Capitale (Grok + Gemini):** Con $2,600 la regola PDT non è un dettaglio ma un vincolo strutturale. La modalità "Force Swing" e il position sizing sono necessari per operare efficacemente sotto questa soglia.

Il messaggio più importante è semplice: **i fix 1 e 2 sono entrambi a basso effort e insieme sbloccano completamente il sistema.** Tutto il resto è ottimizzazione incrementale su un sistema che, una volta sbloccato, potrà finalmente generare dati utili per il tuning successivo.

---

*Report generato il 2026-02-18 — Sintesi cross-review di 4 modelli AI sul session report Fricktrade v3.0*
