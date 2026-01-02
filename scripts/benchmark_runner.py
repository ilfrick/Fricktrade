# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import random

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from app.backtest.agent_engine import run_agent_backtest
from app.utils.config import load_config


@dataclass
class ScenarioResult:
    name: str
    runs: list[dict]
    summary: dict


def _apply_updates(cfg: dict, updates: dict) -> dict:
    new_cfg = copy.deepcopy(cfg)
    for key, value in updates.items():
        section, field = key.split(".", 1)
        if section not in new_cfg:
            new_cfg[section] = {}
        target = new_cfg[section]
        parts = field.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return new_cfg


def _as_runs(result) -> list[dict]:
    if hasattr(result, "runs"):
        runs = []
        for run in result.runs:
            runs.append(
                {
                    "start": run.start,
                    "end": run.end,
                    "return_pct": float(run.return_pct),
                    "trades": int(run.trades),
                    "symbols": list(run.symbols),
                }
            )
        return runs
    return [
        {
            "start": getattr(result, "start", ""),
            "end": getattr(result, "end", ""),
            "return_pct": float(result.return_pct),
            "trades": int(result.trades),
            "symbols": list(getattr(result, "symbols", [])),
        }
    ]


def _summary(
    runs: list[dict],
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> dict:
    returns = [r["return_pct"] for r in runs if r.get("return_pct") is not None]
    trades = [r["trades"] for r in runs if r.get("trades") is not None]
    if not returns:
        return {"return_mean_pct": 0.0, "return_min_pct": 0.0, "return_max_pct": 0.0, "return_std_pct": 0.0}
    mean = sum(returns) / len(returns)
    var = sum((val - mean) ** 2 for val in returns) / max(len(returns), 1)
    std = var**0.5
    buy_hold = [r["buy_hold_return_pct"] for r in runs if r.get("buy_hold_return_pct") is not None]
    buy_hold_mean = sum(buy_hold) / len(buy_hold) if buy_hold else None
    buy_hold_std = None
    if buy_hold:
        bh_var = sum((val - buy_hold_mean) ** 2 for val in buy_hold) / max(len(buy_hold), 1)
        buy_hold_std = bh_var**0.5
    summary = {
        "return_mean_pct": mean,
        "return_min_pct": min(returns),
        "return_max_pct": max(returns),
        "return_std_pct": std,
        "trades_mean": sum(trades) / len(trades) if trades else 0.0,
        "trades_min": min(trades) if trades else 0,
        "trades_max": max(trades) if trades else 0,
    }
    summary["return_ci_pct"] = _bootstrap_ci(returns, bootstrap_samples, bootstrap_confidence, bootstrap_seed)
    if buy_hold_mean is not None:
        summary["buy_hold_mean_pct"] = buy_hold_mean
        summary["buy_hold_std_pct"] = buy_hold_std if buy_hold_std is not None else 0.0
        summary["buy_hold_ci_pct"] = _bootstrap_ci(buy_hold, bootstrap_samples, bootstrap_confidence, bootstrap_seed)
        alpha = [
            float(run["return_pct"]) - float(run["buy_hold_return_pct"])
            for run in runs
            if run.get("buy_hold_return_pct") is not None and run.get("return_pct") is not None
        ]
        if alpha:
            summary["alpha_mean_pct"] = sum(alpha) / len(alpha)
            summary["alpha_std_pct"] = (
                sum((val - summary["alpha_mean_pct"]) ** 2 for val in alpha) / max(len(alpha), 1)
            ) ** 0.5
            summary["alpha_ci_pct"] = _bootstrap_ci(alpha, bootstrap_samples, bootstrap_confidence, bootstrap_seed)
    summary.update(_scorecard(returns))
    return summary


def _bootstrap_ci(values: list[float], samples: int, confidence: float, seed: int) -> dict | None:
    if not values:
        return None
    if samples <= 0:
        return None
    alpha = (1.0 - confidence) / 2.0
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        batch = [rng.choice(values) for _ in range(len(values))]
        means.append(sum(batch) / len(batch))
    means.sort()
    low_idx = int(alpha * len(means))
    high_idx = int((1.0 - alpha) * len(means)) - 1
    low_idx = max(min(low_idx, len(means) - 1), 0)
    high_idx = max(min(high_idx, len(means) - 1), 0)
    return {"low": means[low_idx], "high": means[high_idx], "confidence": confidence}


def _scorecard(returns: list[float]) -> dict:
    if not returns:
        return {"sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "ulcer": 0.0}
    ret_series = [r / 100.0 for r in returns]
    mean = sum(ret_series) / len(ret_series)
    var = sum((r - mean) ** 2 for r in ret_series) / max(len(ret_series), 1)
    std = var**0.5
    downside = [r for r in ret_series if r < 0]
    if downside:
        d_mean = sum(downside) / len(downside)
        d_var = sum((r - d_mean) ** 2 for r in downside) / max(len(downside), 1)
        d_std = d_var**0.5
    else:
        d_std = 0.0
    sharpe = mean / std if std else 0.0
    sortino = mean / d_std if d_std else 0.0
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    dd_squares = []
    for r in ret_series:
        equity *= 1.0 + r
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak if peak else 0.0
        max_dd = max(max_dd, drawdown)
        dd_squares.append(drawdown**2)
    ulcer = (sum(dd_squares) / max(len(dd_squares), 1)) ** 0.5
    calmar = (mean / max_dd) if max_dd else 0.0
    return {"sharpe": sharpe, "sortino": sortino, "calmar": calmar, "ulcer": ulcer}


def _default_scenarios() -> list[tuple[str, dict]]:
    return [
        ("baseline", {}),
        ("stress_costs_moderate", {"backtest.slippage_bps": 5, "backtest.commission_pct": 0.1}),
        ("stress_costs_high", {"backtest.slippage_bps": 10, "backtest.commission_pct": 0.2}),
        ("no_news", {"news.enabled": False, "backtest.news_enabled": False}),
        ("no_signal_bias_guard", {"strategy.signal_bias_guard.enabled": False}),
        ("no_ai_filter", {"data.dynamic_symbols.ai_filter.enabled": False}),
    ]


def _load_frames(cfg: dict) -> dict[str, pd.DataFrame]:
    data_cfg = cfg.get("data", {})
    interval = data_cfg.get("interval", "5m")
    data_dir = Path(cfg.get("backtest", {}).get("data_dir", "/data"))
    symbols = data_cfg.get("symbols", [])
    if not symbols:
        pattern = f"*_{interval}.csv"
        for path in data_dir.glob(pattern):
            name = path.stem
            suffix = f"_{interval}"
            if not name.endswith(suffix):
                continue
            symbol = name[: -len(suffix)]
            if symbol:
                symbols.append(symbol.replace("_", "."))
    frames = {}
    for symbol in symbols:
        path = data_dir / f"{symbol.replace('.', '_')}_{interval}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=[0])
        df.rename(columns={df.columns[0]: "Datetime"}, inplace=True)
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
        df = df.dropna(subset=["Datetime"])
        df["Datetime"] = df["Datetime"].dt.tz_convert(None)
        frames[symbol] = df.set_index("Datetime").sort_index()
    return frames


def _buy_hold_return(frames: dict[str, pd.DataFrame], symbols: list[str], start: str, end: str) -> float | None:
    if not frames or not symbols:
        return None
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    returns = []
    for symbol in symbols:
        df = frames.get(symbol)
        if df is None or "Close" not in df.columns:
            continue
        window = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]
        if window.empty:
            continue
        first = float(window["Close"].iloc[0])
        last = float(window["Close"].iloc[-1])
        if first:
            returns.append((last - first) / first * 100.0)
    if not returns:
        return None
    return sum(returns) / len(returns)


def _window_regime(frames: dict[str, pd.DataFrame], start: str, end: str) -> dict:
    if not frames:
        return {"volatility": None, "trend_pct": None, "regime": "unknown"}
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    vols = []
    trends = []
    for df in frames.values():
        window = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]
        if window.empty or "Close" not in window.columns:
            continue
        close = window["Close"].astype(float).values
        if close.size < 3:
            continue
        returns = pd.Series(close).pct_change().dropna()
        if returns.empty:
            continue
        vols.append(float(returns.std()))
        first = close[0]
        last = close[-1]
        if first:
            trends.append((last - first) / first * 100.0)
    if not vols:
        return {"volatility": None, "trend_pct": None, "regime": "unknown"}
    vol = float(sum(vols) / len(vols))
    trend = float(sum(trends) / len(trends)) if trends else 0.0
    return {"volatility": vol, "trend_pct": trend, "regime": "pending"}


def _assign_regimes(runs: list[dict]) -> None:
    vols = [r["regime"]["volatility"] for r in runs if r.get("regime", {}).get("volatility") is not None]
    trends = [abs(r["regime"]["trend_pct"]) for r in runs if r.get("regime", {}).get("trend_pct") is not None]
    if not vols or not trends:
        return
    vol_threshold = float(pd.Series(vols).median())
    trend_threshold = float(pd.Series(trends).median())
    for run in runs:
        reg = run.get("regime", {})
        vol = reg.get("volatility")
        trend = reg.get("trend_pct")
        if vol is None or trend is None:
            reg["regime"] = "unknown"
            continue
        vol_band = "high_vol" if vol >= vol_threshold else "low_vol"
        trend_band = "trend" if abs(trend) >= trend_threshold else "range"
        reg["regime"] = f"{vol_band}_{trend_band}"


def _regime_summary(runs: list[dict]) -> dict:
    buckets: dict[str, list[float]] = {}
    for run in runs:
        reg = run.get("regime", {}).get("regime", "unknown")
        buckets.setdefault(reg, []).append(float(run.get("return_pct", 0.0)))
    summary = {}
    for key, vals in buckets.items():
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(len(vals), 1)
        summary[key] = {
            "count": len(vals),
            "return_mean_pct": mean,
            "return_std_pct": var**0.5,
            "scorecard": _scorecard(vals),
        }
    return summary


def _plot_summary(results: list[ScenarioResult], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    labels = [r.name for r in results]
    means = [r.summary.get("return_mean_pct", 0.0) for r in results]
    stds = [r.summary.get("return_std_pct", 0.0) for r in results]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, means, yerr=stds, capsize=4)
    ax.set_ylabel("Return % (mean)")
    ax.set_title("Benchmark Scenarios (Mean Return)")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    path = output_dir / "benchmark_returns.png"
    fig.savefig(path)
    paths.append(path)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for item in results:
        ax.scatter(
            item.summary.get("return_std_pct", 0.0),
            item.summary.get("return_mean_pct", 0.0),
            label=item.name,
        )
    ax.set_xlabel("Return Std %")
    ax.set_ylabel("Return Mean %")
    ax.set_title("Scenario Risk/Return")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "benchmark_risk_return.png"
    fig.savefig(path)
    paths.append(path)
    plt.close(fig)
    baseline = next((r for r in results if r.name == "baseline"), None)
    if baseline:
        buckets = baseline.summary.get("regime_summary", {})
        if buckets:
            labels = list(buckets.keys())
            means = [buckets[k].get("return_mean_pct", 0.0) for k in labels]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(labels, means)
            ax.set_ylabel("Return % (mean)")
            ax.set_title("Baseline Returns by Regime")
            ax.tick_params(axis="x", labelrotation=30)
            fig.tight_layout()
            path = output_dir / "benchmark_regimes.png"
            fig.savefig(path)
            paths.append(path)
            plt.close(fig)
    return paths


def _write_pdf(report_path: Path, plots: list[Path]) -> None:
    with PdfPages(report_path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            "Benchmark Report",
            fontsize=16,
            fontweight="bold",
            va="top",
        )
        ax.text(
            0.02,
            0.93,
            f"Generated at: {datetime.utcnow().isoformat()}",
            fontsize=10,
            va="top",
        )
        pdf.savefig(fig)
        plt.close(fig)
        for path in plots:
            fig = plt.figure()
            img = plt.imread(path)
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig(fig)
            plt.close(fig)


def run_benchmarks(cfg: dict, scenarios: list[tuple[str, dict]]) -> list[ScenarioResult]:
    results = []
    frames = _load_frames(cfg)
    bench_cfg = cfg.get("benchmarking", {}) or {}
    bootstrap_samples = int(bench_cfg.get("bootstrap_samples", 500))
    bootstrap_confidence = float(bench_cfg.get("bootstrap_confidence", 0.95))
    mc_cfg = bench_cfg.get("mc", {}) or {}
    seed = int(bench_cfg.get("seed", 42))
    rng = random.Random(seed)
    for name, updates in scenarios:
        scenario_cfg = _apply_updates(cfg, updates)
        result = run_agent_backtest(scenario_cfg)
        runs = _as_runs(result)
        for run in runs:
            run["regime"] = _window_regime(frames, run.get("start", ""), run.get("end", ""))
            run["buy_hold_return_pct"] = _buy_hold_return(
                frames,
                run.get("symbols", []),
                run.get("start", ""),
                run.get("end", ""),
            )
        _assign_regimes(runs)
        summary = _summary(runs, bootstrap_samples, bootstrap_confidence, seed)
        if mc_cfg.get("enabled", False):
            summary["mc_stress"] = _run_mc_stress(
                scenario_cfg,
                mc_cfg,
                rng,
                bootstrap_samples,
                bootstrap_confidence,
                seed,
            )
        results.append(ScenarioResult(name=name, runs=runs, summary=summary))
    return results


def _run_mc_stress(
    base_cfg: dict,
    mc_cfg: dict,
    rng: random.Random,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> dict:
    runs = int(mc_cfg.get("runs", 10))
    slippage_min, slippage_max = _range_or_default(mc_cfg.get("slippage_bps_range"), (2.0, 10.0))
    spread_min, spread_max = _range_or_default(mc_cfg.get("spread_bps_range"), (0.0, 8.0))
    comm_min, comm_max = _range_or_default(mc_cfg.get("commission_pct_range"), (0.02, 0.2))
    returns = []
    trades = []
    for _ in range(runs):
        updates = {
            "backtest.slippage_bps": rng.uniform(slippage_min, slippage_max),
            "backtest.spread_bps": rng.uniform(spread_min, spread_max),
            "backtest.commission_pct": rng.uniform(comm_min, comm_max),
        }
        scenario_cfg = _apply_updates(base_cfg, updates)
        result = run_agent_backtest(scenario_cfg)
        for run in _as_runs(result):
            returns.append(float(run.get("return_pct", 0.0)))
            trades.append(int(run.get("trades", 0)))
    summary = {
        "runs": runs,
        "return_mean_pct": sum(returns) / len(returns) if returns else 0.0,
        "return_min_pct": min(returns) if returns else 0.0,
        "return_max_pct": max(returns) if returns else 0.0,
        "return_std_pct": (
            sum((val - (sum(returns) / len(returns))) ** 2 for val in returns) / max(len(returns), 1)
        )
        ** 0.5
        if returns
        else 0.0,
        "trades_mean": sum(trades) / len(trades) if trades else 0.0,
    }
    summary["return_ci_pct"] = _bootstrap_ci(returns, bootstrap_samples, bootstrap_confidence, bootstrap_seed)
    return summary


def _range_or_default(value, default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="models/benchmark_report.json")
    parser.add_argument("--plot-dir", default="", help="Directory for plots (defaults to output dir).")
    parser.add_argument("--pdf-path", default="", help="Optional PDF summary output path.")
    parser.add_argument("--use-plan", action="store_true", help="Enable backtest.plan.* for walk-forward.")
    parser.add_argument("--plan-window-days", type=int, default=60)
    parser.add_argument("--plan-step-days", type=int, default=30)
    parser.add_argument("--plan-liquidity-tiers", type=int, default=3)
    parser.add_argument("--plan-sample-per-tier", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.use_plan:
        cfg = _apply_updates(
            cfg,
            {
                "backtest.plan.enabled": True,
                "backtest.plan.window_days": args.plan_window_days,
                "backtest.plan.step_days": args.plan_step_days,
                "backtest.plan.liquidity_tiers": args.plan_liquidity_tiers,
                "backtest.plan.sample_per_tier": args.plan_sample_per_tier,
            },
        )

    results = run_benchmarks(cfg, _default_scenarios())
    for item in results:
        item.summary["regime_summary"] = _regime_summary(item.runs)
    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "scenarios": [
            {
                "name": item.name,
                "summary": item.summary,
                "runs": item.runs,
            }
            for item in results
        ],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_dir = Path(args.plot_dir) if args.plot_dir else output_path.parent
    plots = _plot_summary(results, plot_dir)
    if args.pdf_path:
        _write_pdf(Path(args.pdf_path), plots)
    print(f"Benchmark report saved to {output_path}")


if __name__ == "__main__":
    main()
