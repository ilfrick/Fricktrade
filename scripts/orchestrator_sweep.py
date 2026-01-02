# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from app.agents.orchestrator import RLStrategyOrchestrator
from app.agents.trader import TradingAgent
from app.backtest.agent_engine import run_agent_backtest
from app.utils.config import load_config


@dataclass
class SweepRunResult:
    run_id: str
    params: dict
    windows: list[dict]
    mean_return_pct: float
    mean_trades: float


class _DummyBroker:
    def get_account(self) -> dict:
        return {"equity": 0.0, "cash": 0.0}

    def get_positions(self) -> list[dict]:
        return []

    def get_open_orders(self) -> list[dict]:
        return []

    def place_order(self, *args, **kwargs):
        raise RuntimeError("Dummy broker cannot place orders.")

    def close_position(self, *args, **kwargs):
        return None


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[0])
    df.rename(columns={df.columns[0]: "Datetime"}, inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["Datetime"])
    df["Datetime"] = df["Datetime"].dt.tz_convert(None)
    return df.set_index("Datetime").sort_index()


def _data_range(frames: dict[str, pd.DataFrame]) -> tuple[datetime, datetime]:
    starts = [df.index.min() for df in frames.values() if not df.empty]
    ends = [df.index.max() for df in frames.values() if not df.empty]
    if not starts or not ends:
        raise ValueError("No usable CSV data loaded.")
    return max(starts), min(ends)


def _build_windows(start: datetime, end: datetime, window_days: int, count: int) -> list[tuple[datetime, datetime]]:
    total_days = (end - start).days
    if total_days <= window_days:
        return [(end - timedelta(days=window_days), end)]
    count = max(1, count)
    step = max(1, int((total_days - window_days) / max(1, count - 1)))
    windows = []
    for i in range(count):
        win_start = start + timedelta(days=i * step)
        win_end = win_start + timedelta(days=window_days)
        if win_end > end:
            break
        windows.append((win_start, win_end))
    if not windows:
        windows.append((end - timedelta(days=window_days), end))
    return windows


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
        frames[symbol] = _load_csv(path)
    if not frames:
        raise FileNotFoundError(f"No CSV data found under {data_dir}")
    return frames


def _apply_cfg(cfg: dict, updates: dict) -> dict:
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


def _run_pretrain(cfg: dict) -> None:
    orchestrator_cfg = cfg.get("orchestrator", {})
    if not orchestrator_cfg.get("rl", {}).get("enabled", False):
        return
    agent = TradingAgent(_DummyBroker(), cfg)
    orchestrator = RLStrategyOrchestrator(orchestrator_cfg)
    orchestrator.run_pretrain(
        agent._strategy_names,
        agent._build_strategy,
        agent._strategy_params,
        cfg.get("data", {}),
    )


def _evaluate_windows(cfg: dict, windows: list[tuple[datetime, datetime]]) -> list[dict]:
    results = []
    for start, end in windows:
        cfg["backtest"]["start"] = start.strftime("%Y-%m-%d")
        cfg["backtest"]["end"] = end.strftime("%Y-%m-%d")
        result = run_agent_backtest(cfg)
        if hasattr(result, "average_return_pct"):
            return_pct = float(result.average_return_pct)
            trades = float(result.total_trades)
        else:
            return_pct = float(result.return_pct)
            trades = float(result.trades)
        results.append(
            {
                "start": cfg["backtest"]["start"],
                "end": cfg["backtest"]["end"],
                "return_pct": return_pct,
                "trades": trades,
            }
        )
    return results


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _build_grid() -> list[dict]:
    grid = []
    for model_type in ("lstm", "mlp"):
        for hidden_dim in (32, 64):
            for seq_len in (10, 20):
                for lr in (1e-3, 5e-4):
                    grid.append(
                        {
                            "model_type": model_type,
                            "hidden_dim": hidden_dim,
                            "seq_len": seq_len,
                            "learning_rate": lr,
                        }
                    )
    random.shuffle(grid)
    return grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--output", default="models/orchestrator_sweep.json")
    parser.add_argument("--pretrain-coverage-days", type=int, default=730)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    pretrain_interval = (
        base_cfg.get("orchestrator", {})
        .get("rl", {})
        .get("pretrain", {})
        .get("interval", base_cfg.get("data", {}).get("interval", "5m"))
    )
    coverage_days = args.pretrain_coverage_days
    coverage_note = None
    if str(pretrain_interval).endswith("m") and coverage_days > 60:
        coverage_days = 60
        coverage_note = "Intraday data via yfinance is limited to ~60 days; coverage was clamped."
    frames = _load_frames(base_cfg)
    data_start, data_end = _data_range(frames)
    windows = _build_windows(data_start, data_end, args.window_days, args.windows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[SweepRunResult] = []

    baseline_cfg = _apply_cfg(
        base_cfg,
        {
            "orchestrator.rl.enabled": False,
        },
    )
    baseline_windows = _evaluate_windows(baseline_cfg, windows)
    baseline_mean = _mean([w["return_pct"] for w in baseline_windows])
    baseline_trades = _mean([w["trades"] for w in baseline_windows])
    results.append(
        SweepRunResult(
            run_id="baseline",
            params={"mode": "baseline"},
            windows=baseline_windows,
            mean_return_pct=baseline_mean,
            mean_trades=baseline_trades,
        )
    )

    grid = _build_grid()[: max(1, args.runs)]
    for idx, params in enumerate(grid, start=1):
        run_id = f"run_{idx:02d}"
        model_path = f"/data/orchestrator_model_{run_id}.pt"
        best_model_path = f"/data/orchestrator_model_best_{run_id}.pt"
        cfg = _apply_cfg(
            base_cfg,
            {
                "orchestrator.rl.enabled": True,
                "orchestrator.rl.model_type": params["model_type"],
                "orchestrator.rl.hidden_dim": params["hidden_dim"],
                "orchestrator.rl.seq_len": params["seq_len"],
                "orchestrator.rl.learning_rate": params["learning_rate"],
                "orchestrator.rl.model_path": model_path,
                "orchestrator.rl.best_model_path": best_model_path,
                "orchestrator.rl.use_best_model": True,
                "orchestrator.rl.pretrain.coverage_days": coverage_days,
            },
        )
        _run_pretrain(cfg)
        run_windows = _evaluate_windows(cfg, windows)
        results.append(
            SweepRunResult(
                run_id=run_id,
                params=params,
                windows=run_windows,
                mean_return_pct=_mean([w["return_pct"] for w in run_windows]),
                mean_trades=_mean([w["trades"] for w in run_windows]),
            )
        )

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data_range": {"start": str(data_start), "end": str(data_end)},
        "window_days": args.window_days,
        "windows": args.windows,
        "pretrain_interval": pretrain_interval,
        "pretrain_coverage_days": coverage_days,
        "pretrain_note": coverage_note,
        "results": [r.__dict__ for r in results],
    }
    output_path.write_text(json.dumps(payload, indent=2))
    best = max(results, key=lambda r: r.mean_return_pct)
    print(f"Best run: {best.run_id} return={best.mean_return_pct:.2f}% trades={best.mean_trades:.1f}")
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
