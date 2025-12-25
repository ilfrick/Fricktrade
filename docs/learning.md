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
- `learning.model_path`
- `learning.best_model_path`
- `learning.use_best_model`
- `learning.training.*`
- `learning.online.*`
