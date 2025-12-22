from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from app.utils.config import load_config
from app.utils.restart import restart_flag_path


app = FastAPI(title="Autotrader API")
CONFIG_PATH = Path("/app/config/config.yaml")


_DESCRIPTIONS = {
    "app.name": "Application name for logging and identification.",
    "app.env": "Environment label (e.g., prod, dev).",
    "app.timezone": "Timezone for market checks (IANA string).",
    "app.log_level": "Log verbosity level.",
    "market.venue": "Primary venue label (informational).",
    "market.default_currency": "Default currency for reporting.",
    "market.open_mode": "Market gate: any|all configured venues must be open.",
    "market.holiday_update.enabled": "Enable automated holiday refresh.",
    "market.holiday_update.interval_days": "Refresh interval in days.",
    "market.holiday_update.years_ahead": "How many years ahead to populate holidays.",
    "market.venues[].name": "Venue name (NYSE, Nasdaq, BorsaItaliana).",
    "market.venues[].timezone": "Venue timezone (IANA string).",
    "market.venues[].trading_hours.open": "Venue opening time (HH:MM).",
    "market.venues[].trading_hours.close": "Venue closing time (HH:MM).",
    "market.venues[].holidays": "Holiday dates (YYYY-MM-DD).",
    "brokers.alpaca.enabled": "Enable Alpaca broker.",
    "brokers.alpaca.base_url": "Alpaca API base URL (paper or live).",
    "brokers.alpaca.api_key": "Alpaca API key (use ENV var).",
    "brokers.alpaca.api_secret": "Alpaca API secret (use ENV var).",
    "brokers.ibkr.enabled": "Enable IBKR broker.",
    "brokers.ibkr.host": "IBKR host.",
    "brokers.ibkr.port": "IBKR port.",
    "brokers.ibkr.client_id": "IBKR client id.",
    "risk.max_daily_loss_pct": "Max daily loss percentage.",
    "risk.max_position_size_pct": "Max position size percentage.",
    "risk.max_portfolio_leverage": "Max leverage.",
    "risk.max_short_exposure_pct": "Max short exposure.",
    "risk.max_positions": "Max open positions.",
    "risk.cooldown_seconds": "Cooldown between trades.",
    "risk.hard_stop_pct": "Hard stop loss percentage.",
    "risk.trailing_stop_pct": "Trailing stop percentage.",
    "risk.circuit_breaker_drawdown_pct": "Circuit breaker drawdown percentage.",
    "strategy.name": "Strategy name to use.",
    "strategy.names": "List of strategy names for multi-strategy mode (e.g., intraday_momentum, pattern_trading, rl_policy).",
    "strategy.combine": "How to combine strategies: priority or vote.",
    "strategy.params.lookback_minutes": "Momentum lookback window in minutes.",
    "strategy.params.entry_threshold_pct": "Entry threshold percentage.",
    "strategy.params.exit_threshold_pct": "Exit threshold percentage.",
    "strategy.params.position_horizon_minutes": "Max holding horizon in minutes.",
    "strategy.params.allow_shorts": "Allow short positions.",
    "orchestrator.enabled": "Enable AI strategy orchestrator.",
    "orchestrator.mode": "Orchestrator mode: select or weight.",
    "orchestrator.top_k": "Max strategies selected per symbol.",
    "orchestrator.min_score": "Minimum score to include a strategy.",
    "orchestrator.normalize_scores": "Normalize orchestration scores.",
    "orchestrator.strategy_weights": "Weights per strategy (momentum, trend, volatility, relative_volume, session_gain_pct, spread, catalyst).",
    "orchestrator.ml.enabled": "Enable ML-based strategy orchestrator.",
    "orchestrator.ml.model_type": "ML orchestrator model type: lstm or mlp.",
    "orchestrator.ml.device": "Device for ML orchestrator (cpu, cuda, auto).",
    "orchestrator.ml.hidden_dim": "Hidden layer size for ML orchestrator.",
    "orchestrator.ml.dropout": "Dropout rate for ML orchestrator.",
    "orchestrator.ml.num_layers": "Number of LSTM layers when model_type is lstm.",
    "orchestrator.ml.seq_len": "Sequence length for LSTM orchestrator.",
    "orchestrator.ml.model_path": "Path to ML orchestrator model.",
    "orchestrator.ml.best_model_path": "Path to best ML orchestrator model.",
    "orchestrator.ml.use_best_model": "Prefer best ML orchestrator model if available.",
    "orchestrator.ml.best_score_path": "Path to best ML orchestrator score file.",
    "orchestrator.ml.learning_rate": "ML orchestrator learning rate.",
    "orchestrator.ml.weight_decay": "ML orchestrator weight decay.",
    "orchestrator.ml.batch_size": "ML orchestrator batch size.",
    "orchestrator.ml.buffer_size": "ML orchestrator replay buffer size.",
    "orchestrator.ml.update_steps_per_bar": "ML orchestrator update steps per bar.",
    "orchestrator.ml.epsilon": "ML orchestrator exploration rate.",
    "orchestrator.ml.min_price_move_pct": "Minimum price move to train the ML orchestrator.",
    "orchestrator.ml.reward_scale": "Reward scale for ML orchestrator.",
    "orchestrator.ml.max_grad_norm": "Gradient clipping for ML orchestrator.",
    "orchestrator.ml.save_interval_seconds": "Checkpoint interval for ML orchestrator.",
    "orchestrator.ml.score_ema_alpha": "EMA alpha for ML orchestrator performance score.",
    "orchestrator.ml.pretrain.enabled": "Enable ML orchestrator pretraining.",
    "orchestrator.ml.pretrain.in_trader": "Allow orchestrator pretrain inside trader process.",
    "orchestrator.ml.pretrain.lookback_days": "Pretrain lookback days.",
    "orchestrator.ml.pretrain.interval": "Pretrain data interval.",
    "orchestrator.ml.pretrain.max_symbols": "Max symbols for ML orchestrator pretrain.",
    "orchestrator.ml.pretrain.max_samples": "Max training samples for ML orchestrator pretrain.",
    "orchestrator.ml.pretrain.epochs": "ML orchestrator pretrain epochs.",
    "orchestrator.ml.pretrain.warmup_bars": "Warmup bars before generating samples.",
    "orchestrator.ml.pretrain.symbols_source": "Symbol source for pretrain (data or alpaca_active_random).",
    "orchestrator.ml.pretrain.timeout_seconds": "Timeout per yfinance pretrain download.",
    "orchestrator.ml.pretrain.retries": "Retries per yfinance pretrain download.",
    "orchestrator.learning.enabled": "Enable learning for orchestrator bias updates.",
    "orchestrator.learning.learning_rate": "Learning rate for strategy bias updates.",
    "orchestrator.learning.min_bias": "Minimum bias value per strategy.",
    "orchestrator.learning.max_bias": "Maximum bias value per strategy.",
    "orchestrator.learning.decay": "Bias decay per update.",
    "orchestrator.learning.min_price_move_pct": "Minimum price move percent before updating bias.",
    "orchestrator.learning.state_path": "Path to persist orchestrator biases.",
    "orchestrator.learning.save_interval_seconds": "Minimum seconds between orchestrator state saves.",
    "orchestrator.learning.use_best_state": "Prefer loading the best orchestrator state if available.",
    "orchestrator.learning.best_state_path": "Path to persist the best orchestrator biases.",
    "orchestrator.learning.best_score_path": "Path to persist the best orchestrator score.",
    "orchestrator.learning.score_ema_alpha": "EMA alpha for orchestrator performance scoring.",
    "learning.enabled": "Enable RL policy inference.",
    "learning.model_path": "Path to policy model file.",
    "learning.best_model_path": "Path to best policy model file (auto-selected).",
    "learning.use_best_model": "Prefer the best model if available.",
    "learning.device": "Device for RL model (cpu, cuda, auto).",
    "learning.window_size": "Feature window size.",
    "learning.features.include_returns": "Include returns in features.",
    "learning.features.sma_periods": "SMA periods.",
    "learning.features.ema_periods": "EMA periods.",
    "learning.features.rsi_periods": "RSI periods.",
    "learning.guardrail.enabled": "Enable rule-based guardrail.",
    "learning.guardrail.mode": "Guardrail mode (confirm|veto).",
    "learning.guardrail.params.lookback_minutes": "Guardrail lookback minutes.",
    "learning.guardrail.params.entry_threshold_pct": "Guardrail entry threshold.",
    "learning.guardrail.params.exit_threshold_pct": "Guardrail exit threshold.",
    "learning.guardrail.params.allow_shorts": "Guardrail allows shorts.",
    "learning.online.enabled": "Enable online updates.",
    "learning.online.update_interval_minutes": "Online update interval minutes.",
    "learning.online.timesteps": "Online update timesteps.",
    "learning.online.eval_split": "Online update eval split.",
    "learning.training.data_dir": "Training data directory.",
    "learning.training.interval": "Training interval.",
    "learning.training.timesteps": "Training timesteps.",
    "learning.training.initial_cash": "Training initial cash.",
    "learning.training.commission_pct": "Training commission percent.",
    "learning.training.slippage_bps": "Training slippage in bps.",
    "learning.training.eval_split": "Training eval split.",
    "learning.training.resume": "Resume from existing model if present.",
    "learning.training.report_path": "Training report JSON path.",
    "learning.training.best_report_path": "Best-model report JSON path.",
    "learning.training.report_plot_dir": "Training report plots path.",
    "backtest.data_dir": "Backtest data directory.",
    "backtest.start": "Backtest start date (YYYY-MM-DD).",
    "backtest.end": "Backtest end date (YYYY-MM-DD).",
    "backtest.initial_cash": "Backtest initial cash.",
    "backtest.commission_pct": "Backtest commission percent.",
    "backtest.slippage_bps": "Backtest slippage in bps.",
    "backtest.use_gpu": "Enable GPU acceleration (if available).",
    "data.provider": "Data provider (yfinance).",
    "data.symbols": "Symbols to trade.",
    "data.interval": "Data interval (e.g., 1m).",
    "data.lookback_days": "Lookback days for live data.",
    "data.session_gain_mode": "Session gain mode: gap (vs prior close) or session (from open).",
    "data.dynamic_symbols.enabled": "Enable dynamic symbol scanning.",
    "data.dynamic_symbols.provider": "Dynamic symbol provider (alpaca).",
    "data.dynamic_symbols.feed": "Alpaca data feed (iex or sip).",
    "data.dynamic_symbols.refresh_minutes": "Refresh cadence for dynamic symbols.",
    "data.dynamic_symbols.max_symbols": "Maximum symbols to trade after scanning.",
    "data.dynamic_symbols.max_universe": "Maximum symbols to scan from the universe.",
    "data.dynamic_symbols.universe": "Universe source (alpaca_active or comma-separated list).",
    "data.dynamic_symbols.cash_aware": "Limit symbols to those affordable with current cash.",
    "data.dynamic_symbols.cash_buffer_pct": "Reserve cash buffer percentage for affordability filter.",
    "data.dynamic_symbols.cash_cap_mode": "Cash cap mode: cash (only available cash) or risk (min cash and risk max position).",
    "data.dynamic_symbols.cash_max_pct": "Max percent of cash allowed when capping symbol prices.",
    "data.dynamic_symbols.timeout_seconds": "Timeout for Alpaca snapshot scans.",
    "data.dynamic_symbols.retries": "Retries for Alpaca snapshot scans.",
    "data.dynamic_symbols.fallback.enabled": "Enable relaxed filters when the main scan yields no symbols.",
    "data.dynamic_symbols.fallback.relative_volume_min": "Fallback minimum relative volume.",
    "data.dynamic_symbols.fallback.premarket_gain_min_pct": "Fallback minimum premarket gain percent.",
    "data.dynamic_symbols.fallback.min_shares_traded": "Fallback minimum session volume.",
    "data.dynamic_symbols.fallback.max_spread_pct": "Fallback maximum spread percent.",
    "data.dynamic_symbols.fallback.require_catalyst": "Fallback requires a catalyst.",
    "data.start": "Historical start date override.",
    "data.end": "Historical end date override.",
    "data.proxy": "Proxy for data downloads.",
    "data.rate_limit_seconds": "Rate limit between requests.",
    "data.output_dir": "Data output directory.",
    "data.sources[].provider": "Ingestion source provider.",
    "data.sources[].enabled": "Enable ingestion source.",
    "data.sources[].symbols": "Symbols for source.",
    "data.sources[].interval": "Interval for source.",
    "data.sources[].lookback_days": "Lookback days for source.",
    "data.sources[].rate_limit_seconds": "Rate limit for source.",
    "data.sources[].api_key": "API key for source.",
    "news.enabled": "Enable news catalyst filtering.",
    "news.provider": "News provider (alpaca).",
    "news.base_url": "News API base URL.",
    "news.api_key": "News API key (defaults to Alpaca key).",
    "news.api_secret": "News API secret (defaults to Alpaca secret).",
    "news.lookback_hours": "Lookback window for catalyst news.",
    "news.cache_minutes": "Cache duration for news lookups.",
    "news.keywords": "Optional keyword filters for news headlines.",
    "news.timeout_seconds": "Timeout for news API requests.",
    "news.retries": "Retries for news API requests.",
    "pattern_trading.selection.price_min": "Minimum price for pattern trading.",
    "pattern_trading.selection.price_max": "Maximum price for pattern trading.",
    "pattern_trading.selection.relative_volume_min": "Minimum relative volume.",
    "pattern_trading.selection.premarket_gain_min_pct": "Minimum session gain percent.",
    "pattern_trading.selection.min_shares_traded": "Minimum intraday shares traded.",
    "pattern_trading.selection.max_spread_pct": "Maximum spread percent.",
    "pattern_trading.selection.strict_spread": "Require spread data to be present.",
    "pattern_trading.selection.require_catalyst": "Require news catalyst.",
    "pattern_trading.pattern.ma_periods": "Moving average periods for trend confirmation.",
    "pattern_trading.pattern.pullback_max_retrace_pct": "Maximum pullback retrace percent.",
    "pattern_trading.entry.breakout_lookback_bars": "Lookback bars for breakout.",
    "pattern_trading.entry.volume_confirm_mult": "Volume multiplier for entry confirmation.",
    "pattern_trading.risk.stop_loss_pct": "Stop loss percent below entry.",
    "pattern_trading.risk.partial_take_profit_pct": "Partial take profit percent.",
    "pattern_trading.risk.trailing_stop_pct": "Trailing stop percent.",
    "execution.open_orders.enabled": "Enable periodic open-order checks.",
    "execution.open_orders.interval_seconds": "Open-order refresh interval in seconds.",
    "execution.open_orders.skip_if_pending": "Skip new signals if an order is pending for the symbol.",
    "logging.file_path": "Log file path (rotating).",
    "logging.max_bytes": "Log rotation size in bytes.",
    "logging.backup_count": "Number of rotated log files to retain.",
    "monitoring.prometheus_port": "Prometheus metrics port.",
    "monitoring.metrics_path": "Metrics path.",
}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/config")
async def get_config():
    cfg = load_config("/app/config/config.yaml")
    cfg["brokers"]["alpaca"]["api_key"] = "***"
    cfg["brokers"]["alpaca"]["api_secret"] = "***"
    if "news" in cfg:
        cfg["news"]["api_key"] = "***"
        cfg["news"]["api_secret"] = "***"
    return cfg


@app.get("/config/raw")
async def get_config_raw():
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle) or {}
    _mask_secrets(raw_cfg)
    return {"yaml": yaml.safe_dump(raw_cfg, sort_keys=False)}


@app.get("/config/schema")
async def get_config_schema():
    return {"descriptions": _DESCRIPTIONS}


@app.post("/config/update")
async def update_config(payload: dict[str, Any]):
    if "yaml" not in payload:
        raise HTTPException(status_code=400, detail="Missing yaml field")
    raw_yaml = payload["yaml"]
    try:
        new_cfg = yaml.safe_load(raw_yaml) or {}
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}") from exc
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        current_cfg = yaml.safe_load(handle) or {}
    _merge_secrets(new_cfg, current_cfg)
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        yaml.safe_dump(new_cfg, handle, sort_keys=False)
    return {"status": "ok", "restart_required": True}


@app.post("/restart")
async def request_restart():
    restart_flag_path().write_text(datetime.utcnow().isoformat(), encoding="utf-8")
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
async def ui():
    return HTMLResponse(_render_ui())


def _mask_secrets(cfg: dict) -> None:
    try:
        cfg["brokers"]["alpaca"]["api_key"] = "***"
        cfg["brokers"]["alpaca"]["api_secret"] = "***"
    except Exception:
        pass
    try:
        cfg["news"]["api_key"] = "***"
        cfg["news"]["api_secret"] = "***"
    except Exception:
        pass
    try:
        cfg["data"]["sources"][2]["api_key"] = "***"
    except Exception:
        pass


def _merge_secrets(target: dict, source: dict) -> None:
    for path in (
        ("brokers", "alpaca", "api_key"),
        ("brokers", "alpaca", "api_secret"),
        ("news", "api_key"),
        ("news", "api_secret"),
        ("data", "sources", 2, "api_key"),
    ):
        try:
            current = source
            for key in path:
                current = current[key]
            new_val = target
            for key in path[:-1]:
                new_val = new_val[key]
            if new_val[path[-1]] == "***":
                new_val[path[-1]] = current
        except Exception:
            continue


def _render_ui() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Autotrader Config</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap');
    body { font-family: "Space Grotesk", ui-sans-serif, system-ui; margin: 0; background: #0f172a; color: #e2e8f0; }
    header { padding: 16px 24px; background: #111827; border-bottom: 1px solid #1f2937; }
    main { display: grid; grid-template-columns: 1.2fr 1fr; gap: 16px; padding: 16px 24px; }
    textarea { width: 100%; height: 72vh; background: #0b1220; color: #e2e8f0; border: 1px solid #1f2937; border-radius: 8px; padding: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12px; }
    .panel { background: #0b1220; border: 1px solid #1f2937; border-radius: 8px; padding: 12px; }
    .actions { display: flex; gap: 8px; margin-top: 12px; }
    button { background: #2563eb; color: white; border: none; padding: 8px 12px; border-radius: 6px; cursor: pointer; }
    button.secondary { background: #334155; }
    .status { margin-top: 8px; font-size: 12px; color: #94a3b8; }
    .desc { font-size: 12px; line-height: 1.4; padding: 6px 0; display: grid; grid-template-columns: minmax(180px, 0.9fr) 1.1fr; gap: 12px; }
    .desc b { color: #e2e8f0; display: block; }
    .group { margin-top: 12px; border-top: 1px solid #1f2937; padding-top: 10px; }
    .group-title { font-size: 13px; letter-spacing: 0.04em; text-transform: uppercase; color: #7dd3fc; margin-bottom: 6px; }
    .desc span { color: #cbd5f5; }
    .group-meta { font-size: 12px; color: #94a3b8; margin-bottom: 6px; }
    code { background: #111827; color: #e2e8f0; padding: 2px 6px; border-radius: 6px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
    input { width: 100%; padding: 6px; margin-bottom: 8px; border-radius: 6px; border: 1px solid #1f2937; background: #0b1220; color: #e2e8f0; }
    select { width: 100%; padding: 6px; margin-bottom: 8px; border-radius: 6px; border: 1px solid #1f2937; background: #0b1220; color: #e2e8f0; }
  </style>
</head>
<body>
  <header><h2>Autotrader Configuration</h2></header>
  <main>
    <section class="panel">
      <textarea id="config"></textarea>
      <div class="actions">
        <button onclick="applyConfig()">Apply Changes</button>
        <button class="secondary" onclick="requestRestart()">Restart Agent</button>
      </div>
      <div class="status" id="status">Loading...</div>
    </section>
    <section class="panel">
      <div class="group-meta">
        Orchestrator pretraining runs out-of-band. Use:
        <code>docker compose run --rm trader python3 -m app.main pretrain-orchestrator --config /app/config/config.yaml</code>
      </div>
      <input id="filter" placeholder="Filter parameters..." oninput="renderDescriptions()"/>
      <select id="groupFilter" onchange="renderDescriptions()">
        <option value="all">All groups</option>
      </select>
      <div id="descriptions"></div>
    </section>
  </main>
  <script>
    let descriptions = {};
    const groupOrder = [
      "market",
      "brokers",
      "strategy",
      "orchestrator",
      "learning",
      "risk",
      "execution",
      "data",
      "news",
      "backtest",
      "logging",
      "monitoring",
      "other"
    ];
    const groupLabels = {
      market: "Market Schedule & Gate",
      brokers: "Broker Adapters",
      strategy: "Strategy Engine",
      orchestrator: "AI Orchestrator",
      learning: "RL Policy & Guardrails",
      risk: "Risk Management",
      execution: "Execution & Orders",
      data: "Data Ingestion & Feeds",
      news: "News & Catalysts",
      backtest: "Backtesting",
      logging: "Logging & Alerts",
      monitoring: "Monitoring",
      other: "Other"
    };
    async function loadConfig() {
      const res = await fetch('/config/raw');
      const data = await res.json();
      document.getElementById('config').value = data.yaml;
      document.getElementById('status').textContent = 'Loaded configuration.';
    }
    async function loadDescriptions() {
      const res = await fetch('/config/schema');
      const data = await res.json();
      descriptions = data.descriptions || {};
      const groupSelect = document.getElementById('groupFilter');
      groupOrder.forEach(group => {
        const option = document.createElement('option');
        option.value = group;
        option.textContent = groupLabels[group] || group;
        groupSelect.appendChild(option);
      });
      renderDescriptions();
    }
    function renderDescriptions() {
      const filter = document.getElementById('filter').value.toLowerCase();
      const groupFilter = document.getElementById('groupFilter').value;
      const container = document.getElementById('descriptions');
      container.innerHTML = '';
      const grouped = {};
      Object.keys(descriptions).forEach(key => {
        const group = key.split('.')[0] || 'other';
        if (!grouped[group]) grouped[group] = [];
        grouped[group].push(key);
      });
      groupOrder.forEach(group => {
        if (groupFilter !== 'all' && groupFilter !== group) return;
        const keys = (grouped[group] || []).sort();
        const filteredKeys = keys.filter(key => !filter || key.toLowerCase().includes(filter));
        if (!filteredKeys.length) return;
        const section = document.createElement('div');
        section.className = 'group';
        section.innerHTML = `<div class="group-title">${groupLabels[group] || group}</div><div class="group-meta">${filteredKeys.length} parameters</div>`;
        filteredKeys.forEach(key => {
          const div = document.createElement('div');
          div.className = 'desc';
          div.innerHTML = `<b>${key}</b><span>${descriptions[key]}</span>`;
          section.appendChild(div);
        });
        container.appendChild(section);
      });
    }
    async function applyConfig() {
      const payload = { yaml: document.getElementById('config').value };
      const res = await fetch('/config/update', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) });
      const data = await res.json();
      document.getElementById('status').textContent = data.restart_required ? 'Config saved. Restart required.' : 'Config saved.';
    }
    async function requestRestart() {
      const res = await fetch('/restart', { method: 'POST' });
      const data = await res.json();
      document.getElementById('status').textContent = data.status === 'ok' ? 'Restart requested.' : 'Restart failed.';
    }
    loadConfig();
    loadDescriptions();
  </script>
</body>
</html>
    """
