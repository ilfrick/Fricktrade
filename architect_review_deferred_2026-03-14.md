# Fricktrade v3.0 — Architect Review: Deferred Findings

**Review date:** 2026-03-14
**Branch:** v3.0
**Reviewer:** fricktrade-architect agent
**Status of this file:** Deferred items (lower priority). Critical/high-priority findings were fixed in commit c107a20.

---

## Finding 3 — WARNING: `_tmo_counters` Incremented Unlocked From Worker Threads

**File:line:** `app/agents/trader.py:538`

**Root cause.** `_record_skip()` is called from inside worker threads and increments `_tmo_counters["leverage_cap"]` / `_tmo_counters["pos_limit"]` without a lock. `_build_meta_orch_metrics()` reads and resets the same dict on the main thread. CPython's GIL makes individual `+=` on integers atomic in practice, but the read-reset sequence in `_build_meta_orch_metrics` is not. A wrong counter value only affects tactical orchestrator tuning, not financial decisions.

**Proposed fix:**
```python
self._tmo_counters_lock = threading.Lock()

# In _record_skip:
with self._tmo_counters_lock:
    self._tmo_counters["pos_limit"] += 1

# In _build_meta_orch_metrics:
with self._tmo_counters_lock:
    counters_copy = dict(self._tmo_counters)
    self._tmo_counters = {k: 0 for k in self._tmo_counters}
```

**Financial risk:** None direct. Affects tactical orchestrator input data.
**Effort:** 30 min.

---

## Finding 4 — INFO: `_raw_news_cache` Dict-Swap Read/Write Is GIL-Reliant

**File:line:** `app/agents/trader.py:3975, 3878`

**Root cause.** `_raw_news_cache` is replaced atomically on the main thread (dict reference swap) and read by worker threads. In CPython this is safe due to GIL, but is an implicit assumption that breaks under free-threaded CPython 3.13+ or sub-interpreters.

**Proposed fix:** Add a comment documenting the intentional CPython GIL reliance so it isn't "fixed" by future code into something broken. Alternatively, wrap with a `threading.RLock` for correctness.

**Financial risk:** None. Stale news reference is semantically wrong but financially negligible.
**Effort:** 15 min (comment), 30 min (proper lock).

---

## Finding 5 — WARNING: Cluster 2 (New Entries) Sees Pre-Exit Exposure — False Conservative Block

**File:line:** `app/agents/trader.py:2968` (`_run_all_batches_clustered`)

**Root cause.** The per-broker portfolio dict (`bp`) is snapshotted before Cluster 1 (exits) runs. After Cluster 1 sells free capital, Cluster 2 workers still see the pre-exit notional when computing exposure caps. The freed capital is tracked in `_pending_notional` (decremented only on order fill response, next cycle), not reflected in `bp`. This is a **false conservative** — blocks buys it shouldn't, does not create excess exposure.

**Proposed fix:** After Cluster 1 completes, refresh per-broker portfolio from the broker API before starting Cluster 2. This adds one API call per cycle but gives Cluster 2 accurate buying power. Alternatively, accept the conservative behaviour (existing design intent).

**Financial risk:** None (conservative). Reduces capital efficiency.
**Effort:** High (requires async portfolio re-fetch mid-cycle). Low (accept as-is).

---

## Finding 6 — INFO: Trailing Stop Gate `peak > avg_entry` Documents Implicit Behaviour

**File:line:** `app/agents/trader.py:819`

**Root cause.** The trailing stop condition requires `peak_price > avg_entry` before activating. For positions that immediately sell off without ever exceeding entry, the trailing stop never fires. The `hard_stop_pct: 3.0%` covers this case. This is the correct design — trailing stops protect gains, not entries. With `trailing_stop_pct: 1.5%`, there is a 1.5% band where neither stop fires (price above entry but below 3% hard stop, and trailing not yet activated). This is acceptable but should be documented.

**Proposed fix:** Add a comment at line 819 explaining the intentional "trailing stop only protects gains above entry" design. Consider adding a Prometheus metric that fires when a position's peak_price exceeds avg_entry to monitor activation frequency.

**Financial risk:** None. Behaviour is correct.
**Effort:** 10 min (comment).

---

## Finding 10 — WARNING: Cross-Broker Holder Split Depends on `portfolio["brokers"]` Being Populated

**File:line:** `app/agents/trader.py:2968` (`_run_all_batches_clustered`)

**Root cause.** `_portfolio_for_broker()` returns the top-level aggregate portfolio if `portfolio["brokers"][broker_name]` is missing. In that case, Alpaca's holder list would include Binance-held positions, causing Alpaca to attempt to sell a Binance position (which it doesn't hold). Depends on `BrokerRouter.get_portfolio()` always populating the `"brokers"` sub-dict.

**Proposed fix:** Add an assertion or fallback guard in `_portfolio_for_broker`:
```python
def _portfolio_for_broker(self, portfolio: dict, broker_name: str) -> dict:
    brokers = portfolio.get("brokers")
    if isinstance(brokers, dict) and broker_name in brokers:
        data = dict(brokers[broker_name])
        data.setdefault("broker", broker_name)
        return data
    # If broker data is missing, return empty positions rather than aggregate
    # to prevent cross-broker sell attempts on positions held elsewhere.
    logging.warning("No per-broker portfolio data for %s; using empty positions", broker_name)
    return {**portfolio, "positions": {}}
```

**Audit:** Verify `BrokerRouter.get_portfolio()` always populates `portfolio["brokers"]` before implementing the fallback.

**Financial risk:** Medium if `BrokerRouter` ever returns without the `"brokers"` key. Confirmed safe if it always populates it.
**Effort:** 30 min.

---

## Finding 11 — WARNING: `_enrich_signals` Skip in Vote Mode Hides Ollama Latency Bomb

**File:line:** `app/agents/trader.py:1249`

**Root cause.** `_enrich_signals` (which calls Ollama sentiment per symbol) is skipped in `vote` mode (`if self._combine_mode != "vote"`). This is correct — vote doesn't use signal confidence. However, if `combine_mode` is ever changed back to `weighted` or `confidence`, per-symbol Ollama calls fire for every symbol in every cycle, immediately stalling the loop.

**Proposed fix:** Add a startup warning:
```python
if self._combine_mode != "vote" and self._llm_sentiment is not None:
    logging.warning(
        "combine_mode=%s with LLM sentiment active — ensure Ollama wall-clock timeout "
        "is configured; %d symbols × 45s = potential %ds cycle stall",
        self._combine_mode, len(self._symbol_mgr.symbols or []),
        len(self._symbol_mgr.symbols or []) * 45,
    )
```

**Financial risk:** Latent. Safe in current vote mode.
**Effort:** 10 min.

---

## Finding 12 — INFO: 24h Backdate Guarantees `time_exit` Fires for All Positions Post-Restart

**File:line:** `app/agents/trader.py:3792`

**Root cause.** When loading `position_state` from checkpoint, `opened_at` is backdated by 24h. Any position held ≤ 24h before the restart will appear older than 24h after loading. With `regime_hold_minutes.base_minutes: 120` (2h), ALL positions trigger `time_exit` on the first cycle after restart.

**Behaviour depends on intent:**
- If "exit stale positions on restart" is desired → current behaviour is correct.
- If "preserve positions across restarts" is desired → backdate only to `min(original_oa - 24h, now - position_horizon_minutes * 60)` so positions opened within the horizon don't immediately exit.

**Proposed fix (if preservation desired):**
```python
_horizon_minutes = float(self._strategy_params.get("position_horizon_minutes", 0) or 120)
_min_age = timedelta(minutes=_horizon_minutes + 1)  # just past the horizon
_backdated = _oa - timedelta(hours=24)
# Don't backdate past the horizon: preserve positions not yet due for time_exit
if datetime.now(timezone.utc) - _backdated < _min_age:
    _backdated = datetime.now(timezone.utc) - _min_age
pos_restored["opened_at"] = _backdated
```

**Financial risk:** Depends on design intent. Current behaviour causes all positions to be evaluated for exit after restart — this may cause unnecessary sells.
**Effort:** 20 min.

---

## Finding 13 — INFO: `enabled: true` + `mode: vote` Semantically Contradictory in Config

**File:line:** `app/agents/trader.py:405`, `config/config.yaml`

**Root cause.** `llm_orchestrator.enabled: true` but `llm_orchestrator.mode: vote` means the portfolio LLM orchestrator is not initialised (the `_orch_mode != "vote"` guard prevents it). The `enabled` flag is meaningless when `mode: vote`. A future change to `mode: portfolio` without noticing this would re-enable the orchestrator, resuming ~$0.30/day Gemini spend.

**Proposed fix:** Either set `enabled: false` when `mode: vote`, or simplify the init condition to only check `mode == "portfolio"` without the `enabled` flag.

**Financial risk:** Latent spend risk if mode changes without reviewing enabled flag.
**Effort:** 5 min (config change).

---

## Summary

| # | Severity | File | Effort | Action |
|---|---|---|---|---|
| 3 | WARNING | trader.py:538 | 30 min | Add `_tmo_counters_lock` |
| 4 | INFO | trader.py:3975 | 10 min | Add comment documenting GIL reliance |
| 5 | WARNING | trader.py:2968 | Accept as-is or High | Document conservative design or re-fetch portfolio |
| 6 | INFO | trader.py:819 | 10 min | Add comment |
| 10 | WARNING | trader.py:2968 | 30 min | Guard `_portfolio_for_broker` against empty broker data |
| 11 | WARNING | trader.py:1249 | 10 min | Add startup warning for mode switch |
| 12 | INFO | trader.py:3792 | 20 min | Clarify restart behaviour intent |
| 13 | INFO | config.yaml | 5 min | Set `enabled: false` when `mode: vote` |
