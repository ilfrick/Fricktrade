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
    "strategy.params.lookback_minutes": "Momentum lookback window in minutes.",
    "strategy.params.entry_threshold_pct": "Entry threshold percentage.",
    "strategy.params.exit_threshold_pct": "Exit threshold percentage.",
    "strategy.params.position_horizon_minutes": "Max holding horizon in minutes.",
    "strategy.params.allow_shorts": "Allow short positions.",
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
        cfg["data"]["sources"][2]["api_key"] = "***"
    except Exception:
        pass


def _merge_secrets(target: dict, source: dict) -> None:
    for path in (
        ("brokers", "alpaca", "api_key"),
        ("brokers", "alpaca", "api_secret"),
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
    body { font-family: ui-sans-serif, system-ui; margin: 0; background: #0f172a; color: #e2e8f0; }
    header { padding: 16px 24px; background: #111827; border-bottom: 1px solid #1f2937; }
    main { display: grid; grid-template-columns: 1.2fr 1fr; gap: 16px; padding: 16px 24px; }
    textarea { width: 100%; height: 72vh; background: #0b1220; color: #e2e8f0; border: 1px solid #1f2937; border-radius: 8px; padding: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12px; }
    .panel { background: #0b1220; border: 1px solid #1f2937; border-radius: 8px; padding: 12px; }
    .actions { display: flex; gap: 8px; margin-top: 12px; }
    button { background: #2563eb; color: white; border: none; padding: 8px 12px; border-radius: 6px; cursor: pointer; }
    button.secondary { background: #334155; }
    .status { margin-top: 8px; font-size: 12px; color: #94a3b8; }
    .desc { font-size: 12px; line-height: 1.4; border-bottom: 1px solid #1f2937; padding: 6px 0; }
    .desc b { color: #e2e8f0; display: block; }
    input { width: 100%; padding: 6px; margin-bottom: 8px; border-radius: 6px; border: 1px solid #1f2937; background: #0b1220; color: #e2e8f0; }
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
      <input id="filter" placeholder="Filter parameters..." oninput="renderDescriptions()"/>
      <div id="descriptions"></div>
    </section>
  </main>
  <script>
    let descriptions = {};
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
      renderDescriptions();
    }
    function renderDescriptions() {
      const filter = document.getElementById('filter').value.toLowerCase();
      const container = document.getElementById('descriptions');
      container.innerHTML = '';
      Object.keys(descriptions).sort().forEach(key => {
        if (filter && !key.toLowerCase().includes(filter)) return;
        const div = document.createElement('div');
        div.className = 'desc';
        div.innerHTML = `<b>${key}</b>${descriptions[key]}`;
        container.appendChild(div);
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
