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
- **Reward Configuration (`learning.reward`):** Dense per-step reward with 5 components:
  - `learning.reward.nav_weight` - Weight for differential NAV change (default: 1.0)
  - `learning.reward.time_penalty_weight` - Weight for position holding time penalty (default: 0.5)
  - `learning.reward.profit_bonus_weight` - Bonus weight for realized profitable trades (default: 2.0)
  - `learning.reward.velocity_weight` - Weight for equity curve velocity (default: 0.3)
  - `learning.reward.nav_normalizer` - Normalizer for reward magnitudes (default: 100000.0)
  - `learning.reward.time_normalizer` - Minutes in trading day for time penalty scaling (default: 390.0)
  - `learning.reward.velocity_window` - Number of steps for equity velocity lookback (default: 20)
  - `learning.reward.bar_interval_minutes` - Minutes per price bar (default: 5.0)
- **Live Rewards:**
  - `learning.live_rewards.enabled` - Capture live trade rewards (default: false)
  - `learning.live_rewards.path` - JSONL path for live rewards
  - `learning.live_rewards.positions_path` - Persisted positions state
  - `learning.live_rewards.max_days` - Max days of rewards used in training
  - `learning.live_rewards.mode` - `add` or `override` (merge mode)
  - `learning.live_rewards.reward_pnl_mode` - PnL reward mode: `abs` or `pct` (default: `abs`)
  - `learning.live_rewards.reward_pnl_scale` - Scale factor for PnL reward (default: 1.0)
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

## Orchestrator Rewards
The RL orchestrator has its own reward loop used to train strategy selection:
- Default (`orchestrator.rl.reward_mode: price_move`): reward is based on the next price move after a strategy action.
- Policy-aligned (`orchestrator.rl.reward_mode: policy`): reward uses the same dense per-step reward system
  as the TradingEnv (`learning.reward`), including differential NAV, time penalty, profit bonus, and equity velocity.

When using `policy` mode, enable live rewards so the orchestrator sees realized PnL from fills:
- `learning.live_rewards.enabled: true`
- Rewards are recorded to `learning.live_rewards.path` and positions tracked in `learning.live_rewards.positions_path`.

Without live rewards, policy mode still applies time penalty and velocity shaping but will not see actual realized PnL.

## Reward System

The RL agent uses a dense per-step reward system that provides immediate feedback via NAV differential. Every step receives a reward based on NAV change, so the agent learns that holding a losing position is punished immediately, not just at close.

### Reward Components

1. **Differential NAV** (w1, default 1.0):
   - `(NAV_t - NAV_{t-1}) / nav_normalizer`
   - Dense per-step signal from portfolio value changes
   - Transaction costs are implicitly captured (already deducted from NAV)

2. **Time Penalty** (w2, default 0.5):
   - Grows linearly while in position: `-w2 * (steps_held * bar_minutes) / time_normalizer`
   - Penalizes holding positions too long
   - Only active while a position is open

3. **Realized Profit Bonus** (w3, default 2.0):
   - Extra reward on profitable trade close: `w3 * realized_pnl / nav_normalizer`
   - Only triggers when a trade closes with positive PnL

4. **Equity Velocity** (w4, default 0.3):
   - Trend of equity curve over recent steps: `w4 * mean(equity_changes[-N:]) / nav_normalizer`
   - Rewards upward-trending equity, penalizes declining equity
   - Lookback window configurable via `velocity_window`

5. **Live Reward Overrides**:
   - Additive adjustments from live fill rewards (unchanged from previous system)
   - Mode: `add` (default) or `override`

### Metrics Tracked

The environment info dict includes:
- `portfolio_value` - Current portfolio value
- `total_realized_pnl` - Cumulative realized profits/losses
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
