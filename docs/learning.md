# Learning

## Purpose
Train and evaluate RL policies, and support online updates.

## Implementation
- Training: `app/learning/train_rl.py`.
- Online updates: `app/learning/online_update.py`.
- Evaluation: `app/learning/evaluate.py`.
- Features: `app/learning/features.py`.

## Configuration
`config/config.yaml`:
- `learning.enabled`
- `learning.device`
- `learning.window_size`
- `learning.reward_time_penalty_per_step`
- `learning.model_path`
- `learning.best_model_path`
- `learning.use_best_model`
- `learning.training.*`
- `learning.online.*`
- `learning.features.include_signal_features` (adds intraday signal features to observations)
- `learning.features.signal_interval`

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
