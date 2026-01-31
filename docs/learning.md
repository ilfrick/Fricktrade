<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Learning

## Purpose
Train and evaluate RL policies, and support online updates.

## Implementation

Additional ML: the AI symbol filter uses PPO (stable-baselines3) with online updates in `app/data/ai_filter.py`.
- Training: `app/learning/train_rl.py`.
- Online updates: `app/learning/online_update.py`.
- Evaluation: `app/learning/evaluate.py`.
- Features: `app/learning/features.py`.
- Risk-aware features: RL observations include risk config parameters (limits, vol/VAR haircuts, kill switches), broker account flags, and the last per-symbol risk decision (allow/block, reason, action). Changes to risk feature shape require retraining.
- Changing feature shapes (e.g., adding `buying_power_pct`) requires retraining; old checkpoints will be rejected.
- CUDA errors during training or inference disable GPU use until restart; training falls back to CPU.
- Registry: `app/learning/registry.py`.
- Drift monitoring: `app/learning/drift.py`.

## Configuration
`config/config.yaml`:
- `learning.enabled`
- `learning.device`
- `learning.window_size`
- `learning.reward_time_penalty_per_step`
- **Reward Shaping Parameters:**
  - `learning.reward_win_trade_bonus` - Bonus added per winning trade (default: 0.5)
  - `learning.reward_loss_trade_penalty` - Penalty for losing trades (default: 0.2)
  - `learning.reward_win_streak_bonus_scale` - Multiplier for consecutive wins (default: 0.1)
  - `learning.reward_loss_streak_penalty_scale` - Multiplier for consecutive losses (default: 0.15)
  - `learning.reward_max_streak_bonus` - Maximum bonus from win streaks (default: 1.0)
  - `learning.reward_max_streak_penalty` - Maximum penalty from loss streaks (default: 2.0)
  - `learning.reward_sharpe_bonus_scale` - Scale for Sharpe-like risk adjustment (default: 0.1)
  - `learning.reward_sharpe_window_size` - Rolling window for Sharpe calculation (default: 100)
  - `learning.reward_target_trade_frequency` - Target trade frequency as % of bars (default: 0.1)
  - `learning.reward_frequency_penalty_scale` - Penalty for deviating from target frequency (default: 0.5)
  - `learning.reward_pnl_mode` - PnL reward mode: `abs` or `pct` (default: `abs`)
  - `learning.reward_pnl_scale` - Scale factor for PnL reward (default: 1.0)
  - `learning.live_rewards.enabled` - Capture live trade rewards (default: false)
  - `learning.live_rewards.path` - JSONL path for live rewards
  - `learning.live_rewards.positions_path` - Persisted positions state
  - `learning.live_rewards.max_days` - Max days of rewards used in training
  - `learning.live_rewards.mode` - `add` or `override` (merge mode)
- **Time-Aware Penalty:**
  - `learning.enable_time_aware_penalty` - Use dynamic time-based penalties (default: true)
  - `learning.base_time_penalty_per_minute` - Base penalty per minute of inactivity (default: 0.01)
  - `learning.bar_interval_minutes` - Bar interval in minutes for time calculations (default: 5.0)
- `learning.model_path`
- `learning.best_model_path`
- `learning.use_best_model`
- `learning.registry.enabled`
- `learning.registry.path`
- `learning.registry.artifact_dir`
- `learning.registry.artifact_prefix`
- `learning.registry.active_path`
- `learning.registry.use_active`
- `learning.registry.publish_mode`
- `learning.registry.refresh_minutes`
- `learning.drift.enabled`
- `learning.drift.window`
- `learning.drift.feature_zscore_threshold`
- `learning.drift.max_drift_feature_pct`
- `learning.drift.pnl_window`
- `learning.drift.max_pnl_drop_pct`
- `learning.drift.auto_rollback`
- `learning.drift.baseline_enabled`
- `learning.drift.baseline_max_samples`
- `learning.drift.baseline_stride`
- `learning.training.*`
- `learning.online.*`
- `learning.online.respect_ops_state`
- `learning.features.include_signal_features` (adds intraday signal features to observations)
- `learning.features.signal_interval`
  - `learning.training.checkpoint_best_only` - Keep best-only checkpoints (false publishes latest)

## Train
```bash
docker compose run --rm trader python3 -m app.main train --config /app/config/config.yaml
```

## Evaluate
```bash
docker compose run --rm trader python3 -m app.main evaluate --config /app/config/config.yaml
```

## Online Updates
The learner service runs online updates when enabled:
- `learning.online.enabled: true`
- `learning.online.use_live_rewards` (apply live reward overrides in online updates)

## Reward System

The RL agent learns through a multi-component reward system designed to maximize profitable trades while managing risk:

### Reward Components

1. **Realized PnL** - Base reward from closed positions:
   - Long: `(exit_price - entry_price) × quantity - costs`
   - Short: `(entry_price - exit_price) × quantity - costs`
   - If `learning.reward_pnl_mode: pct`, PnL is normalized by `entry_price × |qty|`
   - Transaction costs include commission and slippage

2. **Win-Rate Shaping** - Direct incentives for profitable trades:
   - Win bonus: Additional reward for any profitable trade
   - Loss penalty: Additional penalty beyond PnL loss for losing trades
   - Encourages higher win rates even with smaller profits

3. **Streak Tracking** - Rewards consistency:
   - Win streaks: Growing bonus for consecutive winning trades (capped)
   - Loss streaks: Escalating penalty for consecutive losses (capped)
   - Helps break losing patterns and reinforce successful behaviors

4. **Risk-Adjusted Performance** - Sharpe-like metric:
   - Calculates mean/std of recent trade returns
   - Adds bonus for consistent profitability
   - Penalizes erratic performance

5. **Trade Frequency Control**:
   - Target frequency prevents overtrading and undertrading
   - Penalty for deviating from optimal activity level
   - Balances action with patience

6. **Time Penalties**:
   - **Legacy mode**: Static penalty per step
   - **Time-aware mode**: Dynamic penalty based on actual time since last trade
   - Encourages efficient capital deployment

### Metrics Tracked

The environment info dict includes:
- `portfolio_value` - Current portfolio value
- `total_realized_pnl` - Cumulative realized profits/losses
- `consecutive_wins` - Current win streak length
- `consecutive_losses` - Current loss streak length
- `gross_profits` - Sum of all winning trades
- `gross_losses` - Sum of all losing trades
- `profit_factor` - Ratio of gross profits to gross losses
- `trades_count` - Total trades executed this episode

## Governance
- Training writes metadata + feature baselines into `learning.registry.path`.
- Drift detection compares live features and rolling PnL against the baseline.
- Auto rollback toggles `learning.use_best_model` when drift is detected.
- The learner updates `learning.registry.active_path` when `publish_mode` allows it.
- The trader periodically reloads RL policies when the active model pointer changes.
