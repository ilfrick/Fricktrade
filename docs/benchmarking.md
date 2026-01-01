# Benchmarking

## Purpose
Provide a repeatable, quantitative way to compare strategy performance across
baseline and stress scenarios (costs, guardrails, AI filter toggles).

## Runner
Use `scripts/benchmark_runner.py` to run a battery of agent-aligned backtests
and aggregate the results into a single JSON report.

Example:
```bash
python3 scripts/benchmark_runner.py --config config/config.yaml --use-plan \
  --plan-window-days 60 --plan-step-days 30 --output models/benchmark_report.json
```
Plots are saved next to the report (or to `--plot-dir`) as
`benchmark_returns.png`, `benchmark_risk_return.png`, and `benchmark_regimes.png`.
Use `--pdf-path` to write a multi-page summary PDF.

Bootstrap confidence intervals and Monte Carlo stress runs are controlled via
`benchmarking.bootstrap_*` and `benchmarking.mc.*` in `config/config.yaml`.

## Scenarios (default)
- `baseline`: current config.
- `stress_costs_moderate`: higher slippage/commission.
- `stress_costs_high`: aggressive slippage/commission.
- `no_news`: disable news catalysts.
- `no_signal_bias_guard`: disable signal bias guard.
- `no_ai_filter`: disable AI filter scoring.

## Outputs
The runner writes a JSON report containing:
- per-window runs (start/end, return %, trades, symbols)
- summary statistics (mean/min/max/std for returns, mean/min/max for trades)
- scorecard metrics (Sharpe, Sortino, Calmar, Ulcer index)
- regime summary buckets (high/low volatility vs trend/range)
- bootstrap confidence intervals for return metrics
- buy-and-hold baseline returns and alpha vs buy-and-hold
- Monte Carlo stress summary for randomized slippage/spread/commission

## Interpretation
- Compare `return_mean_pct` across scenarios to understand sensitivity.
- Compare `return_std_pct` for stability/regime risk.
- If `no_signal_bias_guard` improves returns but increases variance, keep guard
  enabled in live runs and tune thresholds instead of disabling.
- If `no_ai_filter` improves returns, reduce the AI filter universe size or
  retrain with updated features.
