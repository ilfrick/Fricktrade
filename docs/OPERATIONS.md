<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Operations Runbook

This guide covers daily operations, monitoring, interpreting output files, handling common issues, and manual intervention procedures.

---

## Daily Monitoring Checklist

### Morning (Before Market Open, ~9:15 ET)

- [ ] Check `docker compose ps` — all services healthy
- [ ] Verify Grafana shows expected account equity and recent heartbeat
- [ ] Check `data/reports/session/` for last night's post-session report
- [ ] Review tactical meta-orchestrator changes: `data/reports/meta_orch/changes.jsonl`
- [ ] Confirm no alerts in Alertmanager (http://localhost:9094)
- [ ] Check `data/checkpoints/trader.json` modified within last 2 min

### During Market Hours (9:30–16:00 ET)

- [ ] Monitor Grafana: `trades_total` rising, `skipped_orders` not dominated by one reason
- [ ] Watch for `meta_orch_health_grade` staying at 0 (green)
- [ ] Check `docker compose logs --tail=50 trader` for WARNING or ERROR messages
- [ ] If equity daily loss approaches threshold, monitor `risk_daily_loss_pct` gauge

### After Market Close

- [ ] Post-session report auto-runs; check `data/reports/session/report_YYYY-MM-DD.json`
- [ ] Review filled trades in Alpaca paper dashboard
- [ ] Check `data/reports/decision_trace/` for any anomalies

---

## Grafana Dashboards

Access at http://localhost:3002 (admin / `GRAFANA_PASSWORD`).

### Per-Account Dashboard

One dashboard per Alpaca account (`Realistic`, `Higher`). Key panels:

| Panel | Metric | What to Watch |
|-------|--------|---------------|
| Account Equity | `portfolio_equity_usd` | Should grow over time; sharp drops = large losses |
| Daily P&L % | `portfolio_day_pnl_pct` | Red if approaching `max_daily_loss_pct: 3.0` |
| Trade Count | `trades_total{side="buy/sell"}` | Low buy count with equity market open = blocked signals |
| Skip Reasons | `skipped_orders_by_broker{reason}` | Dominant `leverage_cap` or `account_blocked` = investigate |
| Positions | `position_value_by_broker` | Confirms active positions |

### Meta-Orchestrator Dashboard

| Panel | Metric | What to Watch |
|-------|--------|---------------|
| Health Grade | `meta_orch_health_grade` | 0=green, 1=yellow, 2=red |
| Strategy Weights | `meta_orch_strategy_weight` | Significant shifts = LLM making adjustments |
| Changes Applied | `meta_orch_changes_applied_total` | Frequent changes = system adapting |
| Cycle Latency | `meta_orch_last_cycle_latency_seconds` | > 30s = Gemini slow or budget hit |

### Common Skip Reasons (skipped_orders_by_broker)

| Reason | Cause | Action |
|--------|-------|--------|
| `leverage_cap` | Exposure at cap | Normal; reduces when positions exit |
| `account_blocked` | Daily loss limit hit | Check P&L; wait for next day or investigate |
| `min_order_notional` | Order below $10 | Normal for small accounts; reduce position count |
| `pending_sell` | Sell already queued | Normal; prevents duplicate sells |
| `pending_buy` | Buy already in flight | Normal; 900s dedup guard active |
| `pdt_block` | PDT restriction | Normal for small accounts; `force_swing: true` holds |
| `min_hold_active` | Within min_hold_minutes | Normal; too-early exit suppressed |
| `insufficient_stablecoin` | Crypto balance too low | Check Binance balance; `crypto_order_margin: 0.97` helps |
| `circuit_breaker` | Symbol down 5%+ | Normal; suppresses adding to losing position |

---

## Reading Decision Traces

Decision traces are written to `data/reports/decision_trace/` as JSONL files (one JSON object per line, one file per day).

### Sampling a Large Trace File

Large files (500MB+) can be sampled at evenly-spaced byte offsets:

```bash
python3 -c "
import json, os
f = 'data/reports/decision_trace/trace_2026-03-15.jsonl'
size = os.path.getsize(f)
offsets = [int(size * i / 5) for i in range(5)]
with open(f) as fh:
    for off in offsets:
        fh.seek(off)
        fh.readline()  # skip partial line
        print(json.loads(fh.readline()))
"
```

### Key Trace Fields

| Field | Description |
|-------|-------------|
| `symbol` | Symbol being evaluated |
| `broker` | Broker name (e.g., `alpaca:Realistic`) |
| `action` | Final action taken: `buy`, `sell`, `hold`, `skip` |
| `skip_reason` | Why skipped (if `action == "skip"`) |
| `buy_votes`, `sell_votes` | Vote counts from strategies |
| `strategies_buy`, `strategies_sell`, `strategies_hold` | Which strategies voted how |
| `shadow_combine` | What `_combine_signals()` would have returned (A/B tracking) |
| `llm_override` | True when portfolio LLM diverged from combine (only in portfolio mode) |
| `effective_weights` | Strategy weights at time of decision (only on `order_enqueued` records) |
| `regime` | Current HMM regime (`low_vol_trending`, `medium`, `high_vol_crisis`) |
| `pnl_pct` | Current position P&L (for held positions) |
| `context_mult` | Confidence adjustment multiplier from regime/alt-data enrichment |

### Common Analysis Patterns

**Why is nothing buying?**
```bash
grep '"action":"buy"' trace_2026-03-15.jsonl | wc -l  # count buys
grep '"skip_reason":"leverage_cap"' trace_2026-03-15.jsonl | wc -l  # count cap skips
```

**Which strategy is voting most?**
```bash
python3 -c "
import json
from collections import Counter
buys = Counter()
with open('data/reports/decision_trace/trace_2026-03-15.jsonl') as f:
    for line in f:
        r = json.loads(line)
        for s in r.get('strategies_buy', []):
            buys[s] += 1
print(buys.most_common(10))
"
```

**Note on equity symbols in trace during off-hours**: Equity symbols appearing in trace during closed hours with `last_price=None` come from the `tests-when-closed` service running backtests, not from live trading.

**Note on `llm_override` accuracy**: `shadow_combine` returns `"exit"` while portfolio orchestrator returns `"sell"` for the same close action, causing `llm_override=true` spuriously for position closes. Only use `llm_override` to analyze buy vs. hold decisions, not sell decisions.

---

## Reading Post-Session Reports

Post-session reports are written to `data/reports/session/report_YYYY-MM-DD.json` after each market session.

Key fields:
- `total_pnl_pct`: session P&L
- `trade_count`: fills this session
- `win_rate`: fraction of closed trades that were profitable
- `strategy_breakdown`: per-strategy win rate and trade count
- `llm_summary`: Gemini's written analysis of the session
- `recommended_adjustments`: Gemini's weight recommendations (not auto-applied unless `auto_apply_weights: true`)

The strategic meta-orchestrator reads the last 7 reports each Sunday to generate `data/reports/meta_orch/strategic_baseline.json`.

---

## Tactical Meta-Orchestrator

The tactical orchestrator runs every 15 minutes and may adjust these parameters:

**Applied immediately (operational params)**:
- `trading_limits.crypto_order_margin`: [0.93, 0.99]
- `execution.stuck_cooldown_minutes`: [5, 30]
- `execution.insufficient_stablecoin_cooldown_minutes`: [2, 15]

**Applied after 5-minute delay (strategy params)**:
- `strategy.params.min_hold_minutes`: [5, 30]
- `strategy.params.regime_hold_minutes.base_minutes`: [60, 480]
- `risk.hard_stop_pct`: [0.5, 3.0]
- `risk.crypto.hard_stop_pct`: [1.5, 6.0]
- `risk.trailing_stop_pct`: [1.0, 10.0]
- `risk.take_profit_pct`: [0.5, 5.0]
- `risk.crypto.take_profit_pct`: [2.0, 8.0]
- Strategy weights: [0.03, 0.45] per strategy

### Reading the Change Log

```bash
# Show last 10 changes
tail -10 data/reports/meta_orch/changes.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    print(r.get('timestamp'), r.get('parameter'), r.get('old_value'), '->', r.get('new_value'))
"
```

### Manual Override of Tactical Changes

To revert a parameter to its config file value:
1. Edit `config/config.yaml` directly
2. Restart the trader: `curl -X POST http://localhost:18081/restart`

The tactical orchestrator will see the new baseline on its next cycle but may re-adjust within bounds.

To disable the tactical orchestrator temporarily:
```yaml
tactical_meta_orchestrator:
  enabled: false
```
Then restart trader.

---

## Common Operational Issues

### Dust Positions (qty < 1e-6)

Dust positions are silently skipped for all exit checks (commit `08c8979`). They will never be automatically closed.

**Detection:**
```bash
# Check for sub-minimum positions in live account
curl http://localhost:18081/ui  # or check Alpaca dashboard
```

**Resolution:**
- Log into Alpaca paper dashboard and manually close the position
- For Binance dust: `_spot_close_position()` calls `client.transfer_dust()` — may trigger automatically on next non-dust close of the same symbol

### Stuck Symbols (pending_buy not released)

A symbol gets stuck when its order times out and the pending_buy guard is not released. This prevents new entries for 900 seconds (15 minutes).

**Symptoms**: Symbol always shows `pending_buy` as skip reason in trace despite no active order.

**Cause**: Order timed out but `_release_pending_buy()` was not called for that terminal status.

**Resolution**: Restart trader; all pending guards reset on restart.

**Prevention**: `stuck_cooldown_minutes: 15` and `stuck_blacklist_after: 3` prevent compounding.

### Leverage Cap Persisting

`pending_leverage_cap` blocks new buys when the pending notional reserve is too high.

**Cause**: `_pending_notional` leaked (not released) on order failures.

**Check**:
```bash
grep "pending_leverage_cap" docker compose logs trader --tail=100
grep "release_pending_notional" docker compose logs trader --tail=100
```

**Resolution**: Restart trader resets all pending guards.

### PDT Force-Swing Active

When `force_swing: true`, positions on small accounts (< $2500) that would trigger PDT are held overnight.

**Symptoms**: Position open past 16:00 ET with reason `pdt_block` in trace.

**This is intentional behavior.** The position will be exited next session.

To bypass: increase account equity above `equity_threshold: 2500` or set `risk.pdt.force_swing: false` (will then block the exit instead, not close the position).

### Daily Loss Limit Hit

When `max_daily_loss_pct: 3.0` is reached, all new entries are blocked for the rest of the day.

**Symptoms**: `account_blocked` as skip reason for all buy attempts.

**Check**: `day_pnl_pct` gauge in Grafana.

**Reset**: Occurs automatically at midnight ET (daily loss tracker resets).

**Manual override** (use carefully): Change `risk.max_daily_loss_pct` to a higher value via `/config/update` API endpoint, then restore at end of day.

### LLM Budget Exceeded

**Symptoms**: LLM modules return errors; meta-orchestrator stops adjusting; macro regime stale.

**Check**:
```bash
grep "budget" docker compose logs trader --tail=50
```

**Reset**: Budget resets at midnight UTC automatically.

**Temporary fix**: Increase `llm.cost.daily_budget_usd` in config.yaml and restart.

### Binance Demo API Timeouts

The Binance demo API is unreliable and can block for 25+ seconds. The broker uses a persistent thread pool with hard wall-clock timeouts:
- `_spot_account()`: 25s timeout
- `_spot_positions()`: 20s timeout
- `_spot_open_orders()`: 15s timeout

**Symptoms**: Binance cycle logs show timeout warnings; cached fallback values used.

**This is expected behavior.** The fallback to `_last_spot_account` prevents stalls.

---

## Safely Restarting the Trader

The trader is stateful but uses checkpoints to recover state on restart.

**What survives restart:**
- Position state (re-fetched from broker on first cycle)
- Historical trade records (in `data/`)
- Model files and tactical changes log

**What resets on restart:**
- All pending guards (`_pending_buy_symbols`, `_pending_sell_qty`, `_pending_notional`)
- Exit backoff timers (`_exit_backoff_until`)
- Stuck cooldown timers (`_stuck_cooldown`)
- LLM decision cache

**Safe restart procedure:**
1. Check no orders in flight: look at Alpaca dashboard for pending orders
2. Cancel pending orders manually if needed
3. `docker compose restart trader`
4. Monitor logs for reconnection errors

**Via API:**
```bash
curl -X POST http://localhost:18081/restart
```

**Hard restart** (rebuilds from code):
```bash
docker compose up -d --build trader
```

---

## Adding or Removing Symbols

### Adding Symbols

The universe is dynamic — `load_universe("alpaca_active_all")` fetches all active Alpaca assets each cycle. New symbols appear automatically when they become tradable on Alpaca.

To manually force a symbol into the universe (bypass AI filter):
1. Not directly supported; the AI filter runs on all candidates
2. Alternative: temporarily lower `data.dynamic_symbols.filters.relative_volume_min` to include more symbols

### Removing Symbols

To blacklist a symbol permanently:
```yaml
trading_limits:
  blocked_symbols:
    - PEPE/USD
    - SHIB/USD
```
Restart trader for this to take effect.

To temporarily suppress a symbol, it will naturally exit the AI filter if it stops meeting volume/spread criteria.

---

## Manually Adjusting Strategy Weights

**Note**: In `combine: vote` mode, weights are ignored. This section applies only if you switch to `combine: weights`.

Via the config UI at http://localhost:18081/ui:
1. Navigate to `strategy.params` section
2. Modify the relevant weight value
3. Click "Apply" — this writes to `config.yaml` and signals a reload

Via API:
```bash
curl -X POST http://localhost:18081/config/update \
  -H "Content-Type: application/json" \
  -d '{"path": "orchestrator.strategy_weights.crypto_momentum", "value": 0.20}'
```

Changes take effect on the next trading cycle (no restart needed).

---

## Enabling Portfolio LLM Mode

The portfolio orchestrator is currently disabled (`llm_orchestrator.mode: vote`). To re-enable:

1. Set `llm_orchestrator.mode: portfolio` in config.yaml
2. Ensure `GOOGLE_GEMINI_API_KEY` is set and has budget
3. Restart trader
4. Monitor `meta_orch_llm_override_win_rate` in Grafana

**Cost**: ~$0.30/day with 3 brokers × 33 symbols per cycle.

**Warning**: In portfolio mode, Gemini receives all symbol data every trading cycle. Ensure `llm_orchestrator.max_tokens: 8192` to prevent truncated JSON responses.

---

## Switching from Paper to Live Trading

1. Open an Alpaca live brokerage account and pass KYC
2. Generate live API keys from the live dashboard
3. Update `.env`:
   ```
   ALPACA_API_KEY=your_live_key
   ALPACA_API_SECRET=your_live_secret
   ```
4. Update `config.yaml`:
   ```yaml
   brokers:
     alpaca:
       base_url: https://api.alpaca.markets
   ```
5. Review and tighten risk parameters:
   - Lower `risk.max_daily_loss_pct` (e.g., 1.0 for live)
   - Lower `risk.max_positions` if starting small
   - Set `trading_limits.allow_shorts: false` until short behavior is validated
6. Restart with `docker compose up -d --build`
7. Monitor the first few orders very closely

**Never run live with `risk.enabled: false`.** The broker account flags will still stop orders but the system-level guards will not.
