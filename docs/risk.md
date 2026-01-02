<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Risk Controls

## Purpose
Enforce exposure, leverage, and safety limits before execution.

## Implementation
- `app/risk/manager.py` enforces max exposure, leverage, and daily loss.
- `app/agents/trader.py` enforces `trading_limits` and account flags.

## Configuration
`config/config.yaml`:
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
- Enable `risk.vol_targeting.enabled` to scale position sizing by realized volatility.
