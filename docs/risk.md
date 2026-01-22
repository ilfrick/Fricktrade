<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Risk Controls

## Purpose
Enforce exposure, leverage, and safety limits before execution.

## Implementation
- `app/risk/manager.py` enforces max exposure, leverage, and daily loss.
- `app/agents/trader.py` enforces `trading_limits` and account flags.
- `app/agents/trader.py` enforces VaR/CVaR limits, exposure caps, and kill switch profiles.
- `risk.enabled: false` bypasses risk checks, but broker account flags and trading limits still block orders.

## Configuration
`config/config.yaml`:
- `risk.enabled`
- `risk.max_daily_loss_pct`
- `risk.max_position_size_pct`
- `risk.max_short_exposure_pct`
- `risk.max_portfolio_leverage`
- `risk.cooldown_seconds`
- `risk.circuit_breaker_drawdown_pct`
- `risk.vol_targeting.enabled`
- `risk.vol_targeting.target_vol_pct`
- `risk.vol_targeting.min_scale`
- `risk.vol_targeting.max_scale`
- `risk.stress.enabled`
- `risk.stress.shock_pct`
- `risk.liquidity_haircut.enabled`
- `risk.liquidity_haircut.max_participation`
- `risk.liquidity_haircut.min_session_volume`
- `risk.liquidity_haircut.max_spread_pct`
- `risk.liquidity_haircut.volume_haircut_pct`
- `risk.var.enabled`
- `risk.var.window`
- `risk.var.confidence`
- `risk.var.max_var_pct`
- `risk.var.max_cvar_pct`
- `risk.exposure_caps.enabled`
- `risk.exposure_caps.venues`
- `risk.exposure_caps.sectors`
- `risk.kill_switch_profiles.enabled`
- `risk.kill_switch_profiles.mode`
- `risk.kill_switch_profiles.current`
- `risk.kill_switch_profiles.adaptive.low_vol_max_pct`
- `risk.kill_switch_profiles.adaptive.high_vol_min_pct`
- `risk.kill_switch_profiles.profiles.*`
- `market.symbol_sectors`
- `trading_limits.enabled`
- `trading_limits.allow_shorts`
- `trading_limits.blocked_symbols`
- `trading_limits.blocked_actions`
- `trading_limits.max_order_qty`
- `trading_limits.max_order_notional`
- `trading_limits.min_order_notional`
- `trading_limits.enforce_account_flags`

## Tuning Notes
- Lower `max_position_size_pct` for more diversified risk.
- Tighten `hard_stop_pct` and `trailing_stop_pct` for faster exits.
- Use `trading_limits.allow_shorts: false` to block shorts.
- Use `risk.var.*` to block new entries when tail risk grows.
- Configure `risk.exposure_caps.*` to limit per-venue or per-sector concentration.
- Use `risk.kill_switch_profiles` with `mode: adaptive` to tighten risk in high-volatility regimes.
- Enable `risk.stress.*` to reduce new risk under shock scenarios.
- Enable `risk.liquidity_haircut.*` to cap order sizing under thin liquidity.
