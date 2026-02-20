# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse

from app.utils.config import load_config
from app.utils.restart import restart_flag_path


app = FastAPI(title="Fricktrade API")
CONFIG_PATH = Path("/app/config/config.yaml")


def _load_raw_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}


def _resolve_auth_token(cfg: dict) -> str | None:
    auth_cfg = cfg.get("api", {}).get("auth", {}) if isinstance(cfg, dict) else {}
    if not auth_cfg.get("enabled", False):
        return None
    token_env = str(auth_cfg.get("token_env", "FRICKTRADE_API_TOKEN"))
    token = os.getenv(token_env, "")
    if not token:
        raise HTTPException(status_code=500, detail="API auth enabled but token is missing")
    return token


def _require_auth(request: Request) -> None:
    token = _resolve_auth_token(_load_raw_config())
    if not token:
        return
    header = request.headers.get("x-api-key", "")
    auth = request.headers.get("authorization", "")
    provided = header
    if auth.lower().startswith("bearer "):
        provided = auth.split(" ", 1)[1].strip()
    if not provided or provided != token:
        raise HTTPException(status_code=401, detail="Unauthorized")


_DESCRIPTIONS = {
    "app.name": "Application name for logging and identification.",
    "app.env": "Environment label (e.g., prod, dev).",
    "app.timezone": "Timezone for market checks (IANA string).",
    "app.log_level": "Log verbosity level.",
    "api.auth.enabled": "Require API token for config and restart endpoints.",
    "api.auth.token_env": "Environment variable holding the API auth token.",
    "market.venue": "Primary venue label (informational).",
    "market.default_currency": "Default currency for reporting.",
    "market.default_symbol_venue": "Fallback venue when a symbol is not mapped.",
    "market.symbol_venues": "Manual symbol -> venue mappings.",
    "market.symbol_currencies": "Manual symbol -> currency mappings.",
    "market.symbol_sectors": "Manual symbol -> sector mappings for exposure caps.",
    "market.open_mode": "Market gate: any|all configured venues must be open.",
    "market.holiday_update.enabled": "Enable automated holiday refresh.",
    "market.holiday_update.interval_days": "Refresh interval in days.",
    "market.holiday_update.years_ahead": "How many years ahead to populate holidays.",
    "market.venues[].name": "Venue name (NYSE, Nasdaq, BorsaItaliana).",
    "market.venues[].timezone": "Venue timezone (IANA string).",
    "market.extended_hours.enabled": "Allow trading during configured extended hours.",
    "market.venues[].trading_hours.open": "Venue opening time (HH:MM).",
    "market.venues[].trading_hours.close": "Venue closing time (HH:MM).",
    "market.venues[].trading_hours.extended_open": "Venue extended-hours open time (HH:MM).",
    "market.venues[].trading_hours.extended_close": "Venue extended-hours close time (HH:MM).",
    "market.venues[].holidays": "Holiday dates (YYYY-MM-DD).",
    "brokers.alpaca.enabled": "Enable Alpaca broker.",
    "brokers.alpaca.base_url": "Alpaca API base URL (paper or live).",
    "brokers.alpaca.api_key": "Alpaca API key (use ENV var).",
    "brokers.alpaca.api_secret": "Alpaca API secret (use ENV var).",
    "brokers.alpaca.accounts": "Optional list of Alpaca accounts (each becomes its own broker instance).",
    "brokers.alpaca.accounts[].name": "Account label used in broker routing (alpaca:<name>).",
    "brokers.alpaca.accounts[].enabled": "Enable this Alpaca account entry.",
    "brokers.alpaca.accounts[].base_url": "Alpaca API base URL for this account.",
    "brokers.alpaca.accounts[].api_key": "Alpaca API key for this account (use ENV var).",
    "brokers.alpaca.accounts[].api_secret": "Alpaca API secret for this account (use ENV var).",
    "brokers.ibkr.enabled": "Enable IBKR broker.",
    "brokers.ibkr.host": "IBKR host.",
    "brokers.ibkr.port": "IBKR port.",
    "brokers.ibkr.client_id": "IBKR client id.",
    "brokers.ibkr.account_id": "Optional IBKR account id for routing positions/orders.",
    "brokers.ibkr.currency": "IBKR default currency for contracts.",
    "brokers.ibkr.accounts": "Optional list of IBKR accounts (each becomes its own broker instance).",
    "brokers.ibkr.accounts[].name": "Account label used in broker routing (ibkr:<name>).",
    "brokers.ibkr.accounts[].enabled": "Enable this IBKR account entry.",
    "brokers.ibkr.accounts[].host": "IBKR host for this account.",
    "brokers.ibkr.accounts[].port": "IBKR port for this account.",
    "brokers.ibkr.accounts[].client_id": "IBKR client id for this account.",
    "brokers.ibkr.accounts[].account_id": "IBKR account id for this account.",
    "brokers.ibkr.accounts[].currency": "IBKR contract currency override for this account.",
    "risk.max_daily_loss_pct": "Max daily loss percentage.",
    "risk.max_position_size_pct": "Max position size percentage.",
    "risk.max_portfolio_leverage": "Max leverage.",
    "risk.max_short_exposure_pct": "Max short exposure.",
    "risk.max_positions": "Max open positions.",
    "risk.cooldown_seconds": "Cooldown between trades.",
    "risk.hard_stop_pct": "Hard stop loss percentage.",
    "risk.trailing_stop_pct": "Trailing stop percentage.",
    "risk.circuit_breaker_drawdown_pct": "Circuit breaker drawdown percentage.",
    "risk.vol_targeting.enabled": "Enable volatility-targeted position sizing.",
    "risk.vol_targeting.target_vol_pct": "Target realized volatility percentage for sizing.",
    "risk.vol_targeting.min_scale": "Minimum sizing scale factor under vol targeting.",
    "risk.vol_targeting.max_scale": "Maximum sizing scale factor under vol targeting.",
    "risk.stress.enabled": "Enable stress haircuts on new positions.",
    "risk.stress.shock_pct": "Percent shock used to reduce position sizing.",
    "risk.liquidity_haircut.enabled": "Enable liquidity-based sizing haircuts.",
    "risk.liquidity_haircut.max_participation": "Max participation of session volume for sizing.",
    "risk.liquidity_haircut.min_session_volume": "Minimum session volume before applying volume haircut.",
    "risk.liquidity_haircut.max_spread_pct": "Max spread percentage before blocking trades.",
    "risk.liquidity_haircut.volume_haircut_pct": "Haircut percentage applied when volume is thin.",
    "risk.var.enabled": "Enable VaR/CVaR gating before new entries.",
    "risk.var.window": "Rolling window size (bars) for VaR/CVaR estimation.",
    "risk.var.confidence": "Confidence level for VaR/CVaR.",
    "risk.var.max_var_pct": "Max allowed VaR percentage before blocking new trades.",
    "risk.var.max_cvar_pct": "Max allowed CVaR percentage before blocking new trades.",
    "risk.exposure_caps.enabled": "Enable exposure caps by venue/sector.",
    "risk.exposure_caps.venues": "Venue -> max exposure percentage mapping.",
    "risk.exposure_caps.sectors": "Sector -> max exposure percentage mapping.",
    "risk.kill_switch_profiles.enabled": "Enable risk profile switching.",
    "risk.kill_switch_profiles.mode": "Profile mode: static or adaptive.",
    "risk.kill_switch_profiles.current": "Active profile name when mode is static.",
    "risk.kill_switch_profiles.adaptive.low_vol_max_pct": "Realized vol threshold for low-risk profile.",
    "risk.kill_switch_profiles.adaptive.high_vol_min_pct": "Realized vol threshold for high-risk profile.",
    "risk.kill_switch_profiles.profiles": "Profile definitions that override risk limits.",
    "strategy.name": "Strategy name to use.",
    "strategy.names": "List of strategy names for multi-strategy mode (e.g., intraday_momentum, pattern_trading, rl_policy).",
    "strategy.combine": "How to combine strategies: priority or vote.",
    "strategy.params.lookback_minutes": "Momentum lookback window in minutes.",
    "strategy.params.entry_threshold_pct": "Entry threshold percentage.",
    "strategy.params.exit_threshold_pct": "Exit threshold percentage.",
    "strategy.params.position_horizon_minutes": "Max holding horizon in minutes.",
    "strategy.params.allow_shorts": "Allow short positions.",
    "strategy.params.trend_following.fast_window": "Trend strategy fast MA window.",
    "strategy.params.trend_following.slow_window": "Trend strategy slow MA window.",
    "strategy.params.trend_following.breakout_pct": "Trend strategy breakout threshold.",
    "strategy.params.trend_following.exit_pct": "Trend strategy exit threshold.",
    "strategy.params.factor_model.momentum_weight": "Factor model momentum weight.",
    "strategy.params.factor_model.liquidity_weight": "Factor model liquidity weight.",
    "strategy.params.factor_model.volatility_weight": "Factor model volatility weight.",
    "strategy.params.factor_model.buy_threshold": "Factor model buy score threshold.",
    "strategy.params.factor_model.sell_threshold": "Factor model sell score threshold.",
    "strategy.params.stat_arb_pairs.lookback": "Stat-arb pairs lookback window.",
    "strategy.params.stat_arb_pairs.z_entry": "Stat-arb entry z-score.",
    "strategy.params.stat_arb_pairs.z_exit": "Stat-arb exit z-score.",
    "strategy.params.stat_arb_pairs.refresh_minutes": "Stat-arb pair refresh interval.",
    "strategy.params.stat_arb_pairs.max_pairs": "Maximum active stat-arb pairs.",
    "strategy.params.market_maker.base_spread_pct": "Market maker base spread percentage.",
    "strategy.params.market_maker.inventory_target_pct": "Market maker inventory target percentage.",
    "strategy.params.market_maker.skew_pct": "Market maker inventory skew percentage.",
    "strategy.params.market_maker.min_qty": "Market maker minimum order quantity.",
    "strategy.params.top_movers_rf.data_dir": "Directory containing historical *_YYYY-MM-DD_1m.csv bars used to train the RF top-movers model.",
    "strategy.params.top_movers_rf.model_path": "Path to the persisted RF model bundle used at runtime.",
    "strategy.params.top_movers_rf.auto_train_on_start": "Auto-train the RF model bundle when missing or stale.",
    "strategy.params.top_movers_rf.retrain_if_older_hours": "Retrain RF bundle when the file age exceeds this many hours.",
    "strategy.params.top_movers_rf.top_n": "Training label size: per-day top-N gainers treated as positives.",
    "strategy.params.top_movers_rf.cutoff_minutes": "Early-session window used for top-mover nowcast features.",
    "strategy.params.top_movers_rf.entry_warmup_minutes": "Minimum minutes before low-zone entry scoring starts.",
    "strategy.params.top_movers_rf.entry_horizon_minutes": "Future window used in entry-label training.",
    "strategy.params.top_movers_rf.low_zone_tol_pct": "Label tolerance (% above day low) for entry positives.",
    "strategy.params.top_movers_rf.rebound_target_pct": "Required rebound percent for entry-positive labels.",
    "strategy.params.top_movers_rf.buy_nowcast_min": "Minimum nowcast probability required to emit buy.",
    "strategy.params.top_movers_rf.buy_entry_min": "Minimum entry probability required to emit buy.",
    "strategy.params.top_movers_rf.buy_score_min": "Minimum weighted score (nowcast+entry) required to emit buy.",
    "strategy.params.top_movers_rf.exit_score_max": "Exit when weighted score falls below this threshold while in position.",
    "strategy.params.top_movers_rf.exit_pullback_from_high_pct": "Exit when pullback from session high exceeds this percentage.",
    "strategy.params.top_movers_rf.nowcast_weight": "Weight for nowcast probability in the combined top-movers score.",
    "strategy.params.top_movers_rf.entry_weight": "Weight for entry probability in the combined top-movers score.",
    "strategy.params.top_movers_rf.respect_account_blocks": "Hold when account flags indicate trading is blocked.",
    "strategy.params.top_movers_rf.respect_pdt_soft_block": "Avoid new entries when PDT soft-block conditions are detected.",
    "strategy.params.top_movers_rf.min_buying_power": "Minimum buying power required before top-movers RF can emit buy.",
    "strategy.signal_bias_guard.enabled": "Enable signal bias guard for strategy actions.",
    "strategy.signal_bias_guard.threshold": "Bias threshold used to block counter-trend actions.",
    "strategy.single_sided_conviction_multiplier": "Multiplier applied to min_conviction when only buy or only sell has non-zero score (0-1).",
    "orchestrator.mode": "Orchestrator mode: direct, select, or weight.",
    "orchestrator.top_k": "Max strategies selected per symbol.",
    "orchestrator.min_score": "Minimum score to include a strategy.",
    "orchestrator.rl.enabled": "Enable RL-based orchestrator.",
    "orchestrator.rl.model_type": "RL orchestrator model type: lstm or mlp.",
    "orchestrator.rl.device": "Device for RL orchestrator (cpu, cuda, auto).",
    "orchestrator.rl.hidden_dim": "Hidden layer size for RL orchestrator.",
    "orchestrator.rl.dropout": "Dropout rate for RL orchestrator.",
    "orchestrator.rl.num_layers": "Number of LSTM layers when model_type is lstm.",
    "orchestrator.rl.seq_len": "Sequence length for LSTM orchestrator.",
    "orchestrator.rl.model_path": "Path to RL orchestrator model.",
    "orchestrator.rl.best_model_path": "Path to best RL orchestrator model.",
    "orchestrator.rl.use_best_model": "Prefer best RL orchestrator model if available.",
    "orchestrator.rl.best_score_path": "Path to best RL orchestrator score file.",
    "orchestrator.rl.learning_rate": "RL orchestrator learning rate.",
    "orchestrator.rl.weight_decay": "RL orchestrator weight decay.",
    "orchestrator.rl.batch_size": "RL orchestrator batch size.",
    "orchestrator.rl.buffer_size": "RL orchestrator replay buffer size.",
    "orchestrator.rl.update_steps_per_bar": "RL orchestrator update steps per bar.",
    "orchestrator.rl.epsilon": "RL orchestrator exploration rate.",
    "orchestrator.rl.min_price_move_pct": "Minimum price move to train the RL orchestrator.",
    "orchestrator.rl.reward_mode": "Reward mode: price_move (default) or policy (match RL policy rewards).",
    "orchestrator.rl.reward_scale": "Reward scale for RL orchestrator.",
    "orchestrator.rl.entropy_coef": "Entropy coefficient for RL policy exploration.",
    "orchestrator.rl.baseline_alpha": "EMA alpha for RL reward baseline.",
    "orchestrator.rl.max_grad_norm": "Gradient clipping for RL orchestrator.",
    "orchestrator.rl.save_interval_seconds": "Checkpoint interval for RL orchestrator.",
    "orchestrator.rl.score_ema_alpha": "EMA alpha for RL orchestrator performance score.",
    "orchestrator.rl.time_penalty_per_bar": "Per-bar time penalty applied to RL orchestrator rewards.",
    "orchestrator.rl.pretrain.enabled": "Enable RL orchestrator pretraining.",
    "orchestrator.rl.pretrain.in_trader": "Allow orchestrator pretrain inside trader process.",
    "orchestrator.rl.pretrain.provider": "Pretrain data provider (yfinance or alpaca).",
    "orchestrator.rl.pretrain.alpaca_api_key": "Alpaca API key for pretraining (env-supported).",
    "orchestrator.rl.pretrain.alpaca_api_secret": "Alpaca API secret for pretraining (env-supported).",
    "orchestrator.rl.pretrain.lookback_days": "Pretrain lookback days.",
    "orchestrator.rl.pretrain.interval": "Pretrain data interval.",
    "orchestrator.rl.pretrain.window_days": "Rolling window size in days for intraday pretrain.",
    "orchestrator.rl.pretrain.coverage_days": "Total historical coverage in days for pretrain.",
    "orchestrator.rl.pretrain.step_days": "Step size in days between pretrain windows.",
    "orchestrator.rl.pretrain.max_symbols": "Max symbols for RL orchestrator pretrain.",
    "orchestrator.rl.pretrain.max_samples": "Max training samples for RL orchestrator pretrain.",
    "orchestrator.rl.pretrain.epochs": "RL orchestrator pretrain epochs.",
    "orchestrator.rl.pretrain.warmup_bars": "Warmup bars before generating samples.",
    "orchestrator.rl.pretrain.symbols_source": "Symbol source for pretrain (data or alpaca_active_random).",
    "healthwatch.market_shutdown.enabled": "Enable market-based stack sleep/wake in healthwatch.",
    "healthwatch.market_shutdown.project_name": "Docker Compose project name to control.",
    "healthwatch.market_shutdown.check_interval_seconds": "Scheduler polling interval for market sleep/wake.",
    "healthwatch.market_shutdown.start_before_minutes": "Minutes before market open to start services.",
    "healthwatch.market_shutdown.heartbeat_minutes": "Heartbeat log interval for market scheduler.",
    "healthwatch.market_shutdown.write_state": "Write ops state file for other services to consume.",
    "healthwatch.market_shutdown.state_path": "Path to the ops state file written by healthwatch.",
    "healthwatch.market_shutdown.keep_services": "Services that stay up when market is closed (must include docker-socket-proxy).",
    "healthwatch.market_shutdown.stop_services": "Explicit services to stop (overrides keep list).",
    "kill_switch.armed": "Arm the kill switch interlock.",
    "kill_switch.confirm_code": "Confirmation code for kill switch actions.",
    "kill_switch.required_code": "Required code (env-supported) for kill switch actions.",
    "kill_switch.confirm_phrase": "Fallback confirmation phrase when no required_code is set.",
    "kill_switch.force_sleep": "Force the stack into sleep mode regardless of market hours.",
    "kill_switch.force_liquidate": "Force liquidation of all positions and cancel open orders.",
    "reports.daily_top_movers.enabled": "Enable daily top movers reporting at market close.",
    "reports.daily_top_movers.top_n": "Number of top movers to include per broker/venue.",
    "reports.daily_top_movers.universe": "Universe source for top movers (alpaca_active or symbols list).",
    "reports.daily_top_movers.max_universe": "Maximum number of symbols to scan for top movers.",
    "reports.daily_top_movers.feed": "Alpaca data feed for the report (iex or sip).",
    "reports.daily_top_movers.output_dir": "Output directory for daily top movers CSVs.",
    "reports.daily_top_movers.training_enabled": "Copy top movers bars into training data directory.",
    "reports.daily_top_movers.training_data_dir": "Training data directory to receive top mover bars.",
    "reports.daily_top_movers.check_interval_seconds": "Scheduler polling interval for daily reports.",
    "reports.daily_top_movers.close_delay_minutes": "Delay after market close before running the report.",
    "reports.daily_top_movers.broker_venues": "Per-broker venue override list for reporting.",
    "reports.daily_top_movers.use_symbol_venues_auto": "Auto-load symbol venue mappings from Alpaca.",
    "reports.daily_top_movers.prometheus_url": "Prometheus base URL for trade/skip metrics.",
    "reports.daily_top_movers.signal_thresholds.early_return_30m_pct": "Early momentum threshold (30m return %).",
    "reports.daily_top_movers.signal_thresholds.sustained_return_60m_pct": "Sustained momentum threshold (60m return %).",
    "reports.daily_top_movers.signal_thresholds.early_volume_pct": "Early volume surge threshold (first 30m % of day volume).",
    "reports.daily_top_movers.signal_thresholds.runup_pct": "Intraday run-up threshold (max % vs open).",
    "reports.daily_top_movers.signal_thresholds.drawdown_pct": "Intraday drawdown threshold (min % vs open).",
    "reports.daily_top_movers.news.enabled": "Include news headlines for top movers.",
    "reports.daily_top_movers.news.provider": "News provider for the report (alpaca).",
    "reports.daily_top_movers.news.max_headlines": "Maximum headlines per symbol in the report.",
    "reports.daily_top_movers.news.include_summaries": "Include news summaries when available.",
    "reports.daily_top_movers.news.correlation_window_minutes": "Time window after open to correlate news with moves.",
    "reports.daily_top_movers.decision_trace.enabled": "Include decision traces from the trading agent.",
    "reports.daily_top_movers.decision_trace.output_dir": "Directory containing decision trace JSONL files.",
    "reports.daily_top_movers.email.smtp_require_tls": "Override SMTP TLS requirement when using Alertmanager settings.",
    "reports.daily_top_movers.explain_ai.enabled": "Enable AI explanations for why symbols were not traded.",
    "reports.daily_top_movers.explain_ai.provider": "AI provider for explanations (ollama).",
    "reports.daily_top_movers.explain_ai.base_url": "Base URL for the AI provider (Ollama).",
    "reports.daily_top_movers.explain_ai.model": "Model name used for AI explanations.",
    "reports.daily_top_movers.explain_ai.timeout_seconds": "Timeout for AI explanations.",
    "reports.daily_top_movers.explain_ai.max_text_chars": "Max text length passed to the AI.",
    "reports.daily_top_movers.email.use_alertmanager_config": "Use Alertmanager SMTP settings for the report email.",
    "reports.daily_top_movers.email.alertmanager_config_path": "Path to alertmanager.yml for SMTP settings.",
    "orchestrator.rl.pretrain.timeout_seconds": "Timeout per yfinance pretrain download.",
    "orchestrator.rl.pretrain.retries": "Retries per yfinance pretrain download.",
    "learning.enabled": "Enable RL policy inference.",
    "learning.model_path": "Path to policy model file.",
    "learning.best_model_path": "Path to best policy model file (auto-selected).",
    "learning.use_best_model": "Prefer the best model if available.",
    "learning.online.respect_ops_state": "Pause online updates when ops state indicates sleep.",
    "learning.registry.enabled": "Enable model registry metadata tracking.",
    "learning.registry.path": "Path to the model registry JSON file.",
    "learning.registry.artifact_dir": "Directory to store versioned model artifacts.",
    "learning.registry.artifact_prefix": "Filename prefix for versioned model artifacts.",
    "learning.registry.active_path": "Path to the active model pointer file.",
    "learning.registry.use_active": "Prefer active model pointer over configured model path.",
    "learning.registry.publish_mode": "Active model publish mode (best or latest).",
    "learning.registry.refresh_minutes": "Minutes between active model pointer refresh checks.",
    "learning.drift.enabled": "Enable drift monitoring for RL policies.",
    "learning.drift.window": "Rolling feature window size for drift checks.",
    "learning.drift.feature_zscore_threshold": "Z-score threshold for feature drift detection.",
    "learning.drift.max_drift_feature_pct": "Max fraction of drifting features before triggering drift.",
    "learning.drift.pnl_window": "Rolling PnL window size for drift checks.",
    "learning.drift.max_pnl_drop_pct": "Max allowed average PnL drop before drift triggers.",
    "learning.drift.auto_rollback": "Reload best model when drift is detected.",
    "learning.drift.baseline_enabled": "Compute feature baseline stats during training.",
    "learning.drift.baseline_max_samples": "Max samples used for baseline feature stats.",
    "learning.drift.baseline_stride": "Stride used when sampling baseline feature stats.",
    "learning.device": "Device for RL model (cpu, cuda, auto).",
    "learning.window_size": "Feature window size.",
    "learning.features.include_returns": "Include returns in features.",
    "learning.features.include_signal_features": "Include intraday signal features in features.",
    "learning.features.signal_interval": "Interval used to compute intraday signal features.",
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
    "execution.algos.enabled": "Enable execution algos (TWAP/VWAP/POV).",
    "execution.algos.default": "Default execution algo (twap|vwap|pov).",
    "execution.algos.min_notional": "Minimum notional to use execution algos.",
    "execution.algos.adaptive.enabled": "Enable adaptive execution algo selection.",
    "execution.algos.adaptive.impact_bps_thresholds.vwap": "Impact bps threshold to prefer VWAP.",
    "execution.algos.adaptive.impact_bps_thresholds.pov": "Impact bps threshold to prefer POV.",
    "execution.algos.twap.duration_seconds": "TWAP duration in seconds.",
    "execution.algos.twap.slices": "TWAP number of slices.",
    "execution.algos.vwap.duration_seconds": "VWAP duration in seconds.",
    "execution.algos.vwap.profile": "VWAP volume profile weights.",
    "execution.algos.pov.max_participation": "POV max participation rate.",
    "execution.algos.pov.estimated_volume": "POV fallback estimated volume.",
    "execution.impact.base_bps": "Base impact bps applied to market impact estimates.",
    "execution.impact.volume_scale_bps": "Impact scaling bps applied by participation and volatility.",
    "execution.impact.min_vol_pct": "Minimum volatility percent used in impact estimates.",
    "execution.retry.enabled": "Enable order retry policy.",
    "execution.retry.max_attempts": "Maximum retry attempts for rejected orders.",
    "execution.retry.backoff_seconds": "Backoff seconds per retry attempt.",
    "execution.retry.max_notional": "Max notional allowed for retries per queue.",
    "execution.retry.reasons": "Rejection reasons eligible for retry.",
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
    "benchmarking.bootstrap_samples": "Bootstrap sample count for benchmark confidence intervals.",
    "benchmarking.bootstrap_confidence": "Bootstrap confidence level for benchmark intervals.",
    "benchmarking.seed": "Random seed for benchmark bootstraps/stress runs.",
    "benchmarking.mc.enabled": "Enable Monte Carlo stress runs for benchmarks.",
    "benchmarking.mc.runs": "Number of Monte Carlo stress runs per scenario.",
    "benchmarking.mc.slippage_bps_range": "Slippage bps range for Monte Carlo stress runs.",
    "benchmarking.mc.spread_bps_range": "Spread bps range for Monte Carlo stress runs.",
    "benchmarking.mc.commission_pct_range": "Commission percent range for Monte Carlo stress runs.",
    "backtest.data_dir": "Backtest data directory.",
    "backtest.start": "Backtest start date (YYYY-MM-DD).",
    "backtest.end": "Backtest end date (YYYY-MM-DD).",
    "backtest.initial_cash": "Backtest initial cash.",
    "backtest.commission_pct": "Backtest commission percent.",
    "backtest.spread_bps": "Backtest spread in bps (half applied per side).",
    "brokers.alpaca.fees.commission_pct": "Alpaca commission percent (estimated).",
    "brokers.alpaca.fees.per_trade_fee": "Alpaca per-trade fee in account currency.",
    "brokers.alpaca.fees.per_share_fee": "Alpaca per-share fee in account currency.",
    "brokers.alpaca.fees.min_fee": "Alpaca minimum fee in account currency.",
    "brokers.alpaca.fees.spread_pct": "Fallback spread percent if spread is missing.",
    "brokers.ibkr.fees.commission_pct": "IBKR commission percent (estimated).",
    "brokers.ibkr.fees.per_trade_fee": "IBKR per-trade fee in account currency.",
    "brokers.ibkr.fees.per_share_fee": "IBKR per-share fee in account currency.",
    "brokers.ibkr.fees.min_fee": "IBKR minimum fee in account currency.",
    "brokers.ibkr.fees.spread_pct": "Fallback spread percent if spread is missing.",
    "strategy.fee_aware.min_edge_pct": "Minimum price move percent required to cover fees.",
    "strategy.fee_aware.edge_multiplier": "Multiplier applied to estimated fee percent before approving trades.",
    "strategy.fee_aware.min_notional": "Minimum order notional used for fee estimation.",
    "strategy.performance.enabled": "Enable rolling strategy performance reporting.",
    "strategy.performance.window_days": "Rolling performance window length in days.",
    "strategy.performance.min_trades": "Minimum trades required before kill switch checks apply.",
    "strategy.performance.min_win_rate": "Minimum win rate before kill switch disables a strategy.",
    "strategy.performance.max_drawdown_pct": "Maximum drawdown percent before kill switch disables a strategy.",
    "strategy.performance.report_interval_minutes": "Interval (minutes) between performance report updates.",
    "strategy.performance.report_path": "Filesystem path for the JSON performance report.",
    "strategy.performance.kill_switch.enabled": "Enable the strategy kill switch.",
    "backtest.slippage_bps": "Backtest slippage in bps.",
    "backtest.use_gpu": "Enable GPU acceleration (if available).",
    "backtest.mode": "Backtest mode: agent (live logic) or sma (legacy).",
    "benchmarking.run_when_closed": "Run benchmarks when markets are closed.",
    "benchmarking.use_plan": "Enable walk-forward backtest plan for benchmarks.",
    "benchmarking.plan_window_days": "Benchmark plan window size in days.",
    "benchmarking.plan_step_days": "Benchmark plan step size in days.",
    "benchmarking.plan_liquidity_tiers": "Benchmark plan liquidity tiers.",
    "benchmarking.plan_sample_per_tier": "Benchmark samples per liquidity tier.",
    "benchmarking.output_path": "Benchmark report output path.",
    "benchmarking.plot_dir": "Benchmark plot output directory.",
    "benchmarking.pdf_path": "Benchmark PDF summary output path.",
    "data.provider": "Data provider (yfinance, alpaca, or brokers).",
    "data.symbols": "Symbols to trade.",
    "data.interval": "Data interval (e.g., 1m).",
    "data.lookback_days": "Lookback days for live data.",
    "data.session_gain_mode": "Session gain mode: gap (vs prior close) or session (from open).",
    "data.dynamic_symbols.enabled": "Enable dynamic symbol scanning.",
    "data.dynamic_symbols.provider": "Dynamic symbol provider (alpaca).",
    "data.dynamic_symbols.feed": "Alpaca data feed (iex or sip).",
    "data.dynamic_symbols.filters.price_min": "Dynamic scanner min price (non-pattern strategies).",
    "data.dynamic_symbols.filters.relative_volume_min": "Dynamic scanner minimum relative volume.",
    "data.dynamic_symbols.filters.premarket_gain_min_pct": "Dynamic scanner minimum premarket/session gain percent.",
    "data.dynamic_symbols.filters.min_shares_traded": "Dynamic scanner minimum shares traded.",
    "data.dynamic_symbols.filters.max_spread_pct": "Dynamic scanner maximum spread percent.",
    "data.dynamic_symbols.filters.strict_spread": "Enforce spread filter strictly.",
    "data.dynamic_symbols.filters.require_catalyst": "Require news catalyst for dynamic scanner (non-pattern strategies).",
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
    "data.dynamic_symbols.ai_filter.allow_orchestrator_fetch": "Allow orchestrator to fetch AI filter features on demand.",
    "data.dynamic_symbols.ai_filter.model_type": "AI filter model type (ppo or linear).",
    "data.dynamic_symbols.ai_filter.model_path": "AI filter model path (PPO uses .zip with sidecar metadata).",
    "data.dynamic_symbols.ai_filter.rl.timesteps": "PPO training timesteps for AI filter.",
    "data.dynamic_symbols.ai_filter.rl.learning_rate": "PPO learning rate for AI filter.",
    "data.dynamic_symbols.ai_filter.rl.batch_size": "PPO batch size for AI filter.",
    "data.dynamic_symbols.ai_filter.rl.n_steps": "PPO rollout steps for AI filter.",
    "data.dynamic_symbols.ai_filter.rl.gamma": "PPO discount factor for AI filter.",
    "data.dynamic_symbols.ai_filter.rl.ent_coef": "PPO entropy coefficient for AI filter.",
    "data.dynamic_symbols.ai_filter.rl.clip_range": "PPO clip range for AI filter.",
    "data.dynamic_symbols.ai_filter.rl.gae_lambda": "PPO GAE lambda for AI filter.",
    "data.dynamic_symbols.ai_filter.online.timesteps": "PPO online update timesteps for AI filter.",
    "data.dynamic_symbols.fallback.enabled": "Enable relaxed filters when the main scan yields no symbols.",
    "data.dynamic_symbols.fallback.relative_volume_min": "Fallback minimum relative volume.",
    "data.dynamic_symbols.fallback.premarket_gain_min_pct": "Fallback minimum premarket gain percent.",
    "data.dynamic_symbols.fallback.min_shares_traded": "Fallback minimum session volume.",
    "data.dynamic_symbols.fallback.max_spread_pct": "Fallback maximum spread percent.",
    "data.dynamic_symbols.fallback.require_catalyst": "Fallback requires a catalyst.",
    "data.quality.enabled": "Enable OHLCV data validation during ingestion.",
    "data.quality.report_path": "Path for the per-symbol data quality report.",
    "data.quality.gap_multiplier": "Gap multiplier used to flag missing bars.",
    "data.quality.outlier_zscore": "Z-score threshold for return outliers.",
    "data.adjustments.enabled": "Enable split/dividend adjustments during ingestion.",
    "data.adjustments.dir": "Directory holding per-symbol adjustment CSV files.",
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
    "news.provider": "News provider (alpaca or brokers).",
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
    "execution.open_orders.missing_grace_seconds": "Grace period before marking a missing order as completed.",
    "execution.brokers.enabled": "Enable multi-broker routing.",
    "execution.brokers.routing.default": "Default broker for order routing.",
    "execution.brokers.routing.symbols": "Symbol-to-broker routing map.",
    "execution.brokers.routing.strategies": "Strategy-to-broker routing map.",
    "execution.brokers.routing.fallback_enabled": "Allow fallback routing when a broker is down.",
    "logging.file_path": "Log file path (rotating).",
    "logging.max_bytes": "Log rotation size in bytes.",
    "logging.backup_count": "Number of rotated log files to retain.",
    "monitoring.prometheus_port": "Prometheus metrics port.",
    "monitoring.metrics_path": "Metrics path.",
    "monitoring.audit.enabled": "Enable audit trail logging for decisions.",
    "monitoring.audit.output_dir": "Directory for audit log JSONL files.",
    "monitoring.audit.include_features": "Include full feature snapshots in audit logs.",
    "monitoring.audit.include_market_state": "Include market state snapshots in audit logs.",
    "monitoring.audit.retention_days": "Days to retain audit log files (0 = disabled).",
    "monitoring.audit.enforce_reason_codes": "Enforce reason codes based on configured taxonomy.",
    "monitoring.audit.reason_codes_path": "Path to JSON list of allowed reason codes.",
    "monitoring.audit.signing.enabled": "Enable per-entry audit log signing.",
    "monitoring.audit.signing.secret_env": "Environment variable containing the audit signing secret.",
    "monitoring.compliance.enabled": "Enable compliance log exports.",
    "monitoring.compliance.output_dir": "Directory for compliance log files.",
    "monitoring.compliance.formats": "Compliance export formats (jsonl, csv).",
    "monitoring.compliance.include_features": "Include feature snapshots in compliance logs.",
    "monitoring.compliance.include_market_state": "Include market state snapshots in compliance logs.",
    "monitoring.compliance.retention_days": "Days to retain compliance exports (0 = disabled).",
    "monitoring.compliance.enforce_reason_codes": "Enforce reason codes based on configured taxonomy.",
    "monitoring.compliance.reason_codes_path": "Path to JSON list of allowed reason codes.",
    "monitoring.compliance.signing.enabled": "Enable compliance export signing.",
    "monitoring.compliance.signing.secret_env": "Environment variable containing the compliance signing secret.",
    "market_cache.allow_pickle": "Allow legacy pickle cache reads (unsafe; defaults off).",
}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/config", dependencies=[Depends(_require_auth)])
async def get_config():
    cfg = load_config(CONFIG_PATH)
    _mask_secrets(cfg)
    return cfg


@app.get("/config/raw", dependencies=[Depends(_require_auth)])
async def get_config_raw():
    raw_cfg = _load_raw_config()
    _mask_secrets(raw_cfg)
    return {"yaml": yaml.safe_dump(raw_cfg, sort_keys=False)}


@app.get("/config/schema", dependencies=[Depends(_require_auth)])
async def get_config_schema():
    return {"descriptions": _DESCRIPTIONS}


@app.post("/config/update", dependencies=[Depends(_require_auth)])
async def update_config(payload: dict[str, Any]):
    if "yaml" not in payload:
        raise HTTPException(status_code=400, detail="Missing yaml field")
    raw_yaml = payload["yaml"]
    try:
        new_cfg = yaml.safe_load(raw_yaml) or {}
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}") from exc
    current_cfg = _load_raw_config()
    unknown_keys = _validate_config_keys(new_cfg, current_cfg)
    if unknown_keys:
        joined = ", ".join(sorted(unknown_keys)[:20])
        suffix = "..." if len(unknown_keys) > 20 else ""
        raise HTTPException(
            status_code=400,
            detail=f"Unknown config keys: {joined}{suffix}",
        )
    _merge_secrets(new_cfg, current_cfg)
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        yaml.safe_dump(new_cfg, handle, sort_keys=False)
    return {"status": "ok", "restart_required": True}


@app.post("/restart", dependencies=[Depends(_require_auth)])
async def request_restart():
    restart_flag_path().write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
async def ui():
    return HTMLResponse(_render_ui())


def _mask_secrets(cfg: dict) -> None:
    try:
        cfg["brokers"]["alpaca"]["api_key"] = "***"
        cfg["brokers"]["alpaca"]["api_secret"] = "***"
    except (KeyError, TypeError):
        pass
    try:
        ibkr_cfg = cfg.get("brokers", {}).get("ibkr", {})
        if ibkr_cfg:
            for key in ("password", "api_key", "api_secret"):
                if key in ibkr_cfg:
                    ibkr_cfg[key] = "***"
            ibkr_accounts = ibkr_cfg.get("accounts", []) or []
            for acct in ibkr_accounts:
                if isinstance(acct, dict):
                    for key in ("password", "api_key", "api_secret"):
                        if key in acct:
                            acct[key] = "***"
    except (KeyError, TypeError):
        pass
    try:
        accounts = cfg.get("brokers", {}).get("alpaca", {}).get("accounts", []) or []
        for acct in accounts:
            if not isinstance(acct, dict):
                continue
            acct["api_key"] = "***"
            acct["api_secret"] = "***"
    except Exception:
        pass
    try:
        cfg["news"]["api_key"] = "***"
        cfg["news"]["api_secret"] = "***"
    except Exception:
        pass
    try:
        sources = cfg.get("data", {}).get("sources", []) or []
        for source in sources:
            if not isinstance(source, dict):
                continue
            if "api_key" in source:
                source["api_key"] = "***"
            if "api_secret" in source:
                source["api_secret"] = "***"
    except Exception:
        pass

def _merge_secrets(target: dict, source: dict) -> None:
    for path in (
        ("brokers", "alpaca", "api_key"),
        ("brokers", "alpaca", "api_secret"),
        ("news", "api_key"),
        ("news", "api_secret"),
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
    try:
        current_accounts = source.get("brokers", {}).get("alpaca", {}).get("accounts", []) or []
        new_accounts = target.get("brokers", {}).get("alpaca", {}).get("accounts", []) or []
        current_by_name = {
            str(item.get("name")): item
            for item in current_accounts
            if isinstance(item, dict) and item.get("name")
        }
        for idx, acct in enumerate(new_accounts):
            if not isinstance(acct, dict):
                continue
            current_acct = None
            name = acct.get("name")
            if name:
                current_acct = current_by_name.get(str(name))
            if current_acct is None and idx < len(current_accounts) and isinstance(current_accounts[idx], dict):
                current_acct = current_accounts[idx]
            if not current_acct:
                continue
            for key in ("api_key", "api_secret"):
                if acct.get(key) == "***":
                    acct[key] = current_acct.get(key, "")
    except Exception:
        pass
    try:
        current_sources = source.get("data", {}).get("sources", []) or []
        new_sources = target.get("data", {}).get("sources", []) or []
        current_by_provider = {
            str(item.get("provider")): item
            for item in current_sources
            if isinstance(item, dict) and item.get("provider")
        }
        current_by_name = {
            str(item.get("name")): item
            for item in current_sources
            if isinstance(item, dict) and item.get("name")
        }
        for idx, item in enumerate(new_sources):
            if not isinstance(item, dict):
                continue
            provider = item.get("provider")
            current_item = None
            name = item.get("name")
            if name:
                current_item = current_by_name.get(str(name))
            if current_item is None and provider:
                current_item = current_by_provider.get(str(provider))
            elif idx < len(current_sources) and isinstance(current_sources[idx], dict):
                current_item = current_sources[idx]
            if not current_item:
                continue
            for key in ("api_key", "api_secret"):
                if item.get(key) == "***":
                    item[key] = current_item.get(key, "")
    except Exception:
        pass


def _validate_config_keys(new_cfg: object, current_cfg: object, prefix: str = "") -> list[str]:
    unknown: list[str] = []
    if isinstance(new_cfg, dict) and isinstance(current_cfg, dict):
        for key, value in new_cfg.items():
            if key not in current_cfg:
                unknown.append(f"{prefix}{key}")
                continue
            child_prefix = f"{prefix}{key}."
            unknown.extend(_validate_config_keys(value, current_cfg.get(key), child_prefix))
        return unknown
    if isinstance(new_cfg, list) and isinstance(current_cfg, list):
        if not current_cfg:
            return unknown
        schema = current_cfg[0]
        for idx, item in enumerate(new_cfg):
            child_prefix = f"{prefix}[{idx}]."
            unknown.extend(_validate_config_keys(item, schema, child_prefix))
    return unknown


def _render_ui() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Fricktrade Config</title>
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
    .api-key { display: flex; gap: 8px; align-items: center; margin-bottom: 10px; }
    .api-key input { flex: 1; }
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
  <header><h2>Fricktrade Configuration</h2></header>
  <main>
    <section class="panel">
      <div class="api-key">
        <input id="apiKey" type="password" placeholder="API key (optional)"/>
        <button class="secondary" onclick="saveApiKey()">Save</button>
      </div>
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
    function apiHeaders() {
      const key = localStorage.getItem('apiKey') || '';
      return key ? { 'X-API-Key': key } : {};
    }
    function saveApiKey() {
      const key = document.getElementById('apiKey').value || '';
      if (key) {
        localStorage.setItem('apiKey', key);
      } else {
        localStorage.removeItem('apiKey');
      }
    }
    function loadApiKey() {
      const key = localStorage.getItem('apiKey') || '';
      document.getElementById('apiKey').value = key;
    }
    async function loadConfig() {
      const res = await fetch('/config/raw', { headers: apiHeaders() });
      if (res.status === 401) {
        document.getElementById('status').textContent = 'Unauthorized. Provide API key.';
        return;
      }
      const data = await res.json();
      document.getElementById('config').value = data.yaml;
      document.getElementById('status').textContent = 'Loaded configuration.';
    }
    async function loadDescriptions() {
      const res = await fetch('/config/schema', { headers: apiHeaders() });
      if (res.status === 401) {
        document.getElementById('status').textContent = 'Unauthorized. Provide API key.';
        return;
      }
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
      const headers = { 'Content-Type': 'application/json', ...apiHeaders() };
      const res = await fetch('/config/update', { method: 'POST', headers, body: JSON.stringify(payload) });
      const data = await res.json();
      if (!res.ok) {
        document.getElementById('status').textContent = data.detail || 'Config update failed.';
        return;
      }
      document.getElementById('status').textContent = data.restart_required ? 'Config saved. Restart required.' : 'Config saved.';
    }
    async function requestRestart() {
      const res = await fetch('/restart', { method: 'POST', headers: apiHeaders() });
      if (res.status === 401) {
        document.getElementById('status').textContent = 'Unauthorized. Provide API key.';
        return;
      }
      const data = await res.json();
      document.getElementById('status').textContent = data.status === 'ok' ? 'Restart requested.' : 'Restart failed.';
    }
    loadApiKey();
    loadConfig();
    loadDescriptions();
  </script>
</body>
</html>
    """
