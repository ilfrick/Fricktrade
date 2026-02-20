#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "scikit-learn is required. Install with: /home/nicola/Fricktrade/testenv/bin/python -m pip install scikit-learn"
    ) from exc


FILE_RE = re.compile(r"^(?P<symbol>.+)_(?P<day>\d{4}-\d{2}-\d{2})_1m\.csv$")


@dataclass
class ForecastConfig:
    data_dir: Path
    output_dir: Path
    top_n: int
    min_bars: int
    random_state: int
    forecast_date: date | None


def _parse_args() -> ForecastConfig:
    parser = argparse.ArgumentParser(description="Train a simple daily top-movers forecaster from intraday CSV history.")
    parser.add_argument("--data-dir", default="/home/nicola/Fricktrade/data", help="Directory containing *_YYYY-MM-DD_1m.csv files.")
    parser.add_argument(
        "--output-dir",
        default="/home/nicola/Fricktrade/data/reports/modeling",
        help="Output directory for forecast and metrics artifacts.",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Top-N movers per day used as classification target.")
    parser.add_argument("--min-bars", type=int, default=30, help="Minimum intraday bars required to keep a symbol-day sample.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for tree models.")
    parser.add_argument(
        "--forecast-date",
        default="",
        help="Forecast date in YYYY-MM-DD (model uses previous day features). If omitted, forecasts next day after latest sample date.",
    )
    args = parser.parse_args()
    target = date.fromisoformat(args.forecast_date) if args.forecast_date else None
    return ForecastConfig(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        top_n=max(int(args.top_n), 1),
        min_bars=max(int(args.min_bars), 5),
        random_state=int(args.random_state),
        forecast_date=target,
    )


def _load_ohlcv(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None

    # Normalize common column variants.
    col_map: dict[str, str] = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in {"open", "o"}:
            col_map[c] = "open"
        elif cl in {"high", "h"}:
            col_map[c] = "high"
        elif cl in {"low", "l"}:
            col_map[c] = "low"
        elif cl in {"close", "c"}:
            col_map[c] = "close"
        elif cl in {"volume", "v"}:
            col_map[c] = "volume"
    if not {"open", "high", "low", "close"}.issubset(set(col_map.values())):
        return None
    df = df.rename(columns=col_map)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        return None
    return df


def _symbol_day_features(path: Path, min_bars: int) -> dict | None:
    match = FILE_RE.match(path.name)
    if not match:
        return None
    symbol = match.group("symbol")
    day = date.fromisoformat(match.group("day"))
    df = _load_ohlcv(path)
    if df is None or len(df) < min_bars:
        return None

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)
    if len(c) < 2 or o[0] <= 0:
        return None

    open_px = float(o[0])
    close_px = float(c[-1])
    high_px = float(np.max(h))
    low_px = float(np.min(l))
    day_return_pct = (close_px - open_px) / open_px * 100.0
    runup_pct = (high_px - open_px) / open_px * 100.0
    drawdown_pct = (low_px - open_px) / open_px * 100.0
    abs_range_pct = (high_px - low_px) / open_px * 100.0

    returns = np.diff(c) / np.maximum(c[:-1], 1e-9)
    return_std_pct = float(np.std(returns) * 100.0)
    return_mean_pct = float(np.mean(returns) * 100.0)

    n30 = min(30, len(c))
    n60 = min(60, len(c))
    ret_30m_pct = (float(c[n30 - 1]) - open_px) / open_px * 100.0 if n30 >= 5 else 0.0
    ret_60m_pct = (float(c[n60 - 1]) - open_px) / open_px * 100.0 if n60 >= 5 else 0.0

    total_vol = float(np.sum(v)) if len(v) else 0.0
    early_vol_pct = (float(np.sum(v[:n30])) / total_vol * 100.0) if total_vol > 0 and n30 > 0 else 0.0
    vol_per_bar = total_vol / max(len(v), 1)
    close_vs_vwap = 0.0
    if total_vol > 0:
        vwap = float(np.sum(c * v) / total_vol)
        if vwap > 0:
            close_vs_vwap = (close_px - vwap) / vwap * 100.0

    # Simple trend slope on normalized closes.
    idx = np.arange(len(c), dtype=float)
    norm_c = c / max(open_px, 1e-9)
    slope = float(np.polyfit(idx, norm_c, 1)[0]) if len(c) >= 3 else 0.0

    return {
        "symbol": symbol,
        "date": day,
        "bars": int(len(c)),
        "open_px": open_px,
        "close_px": close_px,
        "day_return_pct": day_return_pct,
        "runup_pct": runup_pct,
        "drawdown_pct": drawdown_pct,
        "range_pct": abs_range_pct,
        "return_std_pct": return_std_pct,
        "return_mean_pct": return_mean_pct,
        "ret_30m_pct": ret_30m_pct,
        "ret_60m_pct": ret_60m_pct,
        "early_vol_pct": early_vol_pct,
        "total_volume": total_vol,
        "vol_per_bar": vol_per_bar,
        "close_vs_vwap_pct": close_vs_vwap,
        "trend_slope": slope,
    }


def _build_feature_frame(cfg: ForecastConfig) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(cfg.data_dir.glob("*_????-??-??_1m.csv")):
        row = _symbol_day_features(path, cfg.min_bars)
        if row:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)
    return df


def _top_membership_by_day(features: pd.DataFrame, top_n: int) -> dict[date, set[str]]:
    top_map: dict[date, set[str]] = {}
    for d, group in features.groupby("date"):
        ranked = group.sort_values("day_return_pct", ascending=False).head(top_n)
        top_map[d] = set(ranked["symbol"].tolist())
    return top_map


def _build_supervised_frame(features: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    dates = sorted(features["date"].unique().tolist())
    next_date = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    top_map = _top_membership_by_day(features, top_n)
    lookup = features.set_index(["date", "symbol"])["day_return_pct"].to_dict()

    rows = []
    for _, row in features.iterrows():
        d = row["date"]
        if d not in next_date:
            continue
        d_next = next_date[d]
        sym = row["symbol"]
        target_membership = 1 if sym in top_map.get(d_next, set()) else 0
        target_next_return = lookup.get((d_next, sym), np.nan)
        out = row.to_dict()
        out["target_date"] = d_next
        out["target_top_n"] = int(target_membership)
        out["target_next_return_pct"] = float(target_next_return) if pd.notna(target_next_return) else np.nan
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"symbol", "date", "target_date", "target_top_n", "target_next_return_pct"}
    return [c for c in df.columns if c not in excluded]


def _walkforward_membership(df: pd.DataFrame, feat_cols: list[str], top_n: int, random_state: int) -> pd.DataFrame:
    metrics: list[dict] = []
    dates = sorted(df["date"].unique().tolist())
    for d in dates[1:]:
        train = df[df["date"] < d]
        test = df[df["date"] == d]
        if train.empty or test.empty:
            continue
        y_train = train["target_top_n"].astype(int)
        if y_train.nunique() < 2:
            continue
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        model.fit(train[feat_cols], y_train)
        proba = model.predict_proba(test[feat_cols])[:, 1]
        ranked = test.assign(score=proba).sort_values("score", ascending=False)
        k = min(top_n, len(ranked))
        precision_at_k = float(ranked.head(k)["target_top_n"].mean())
        baseline = float(test["target_top_n"].mean())
        metrics.append(
            {
                "date": d.isoformat(),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "k": int(k),
                "precision_at_k": precision_at_k,
                "baseline_positive_rate": baseline,
            }
        )
    return pd.DataFrame(metrics)


def _walkforward_return(df: pd.DataFrame, feat_cols: list[str], random_state: int) -> pd.DataFrame:
    metrics: list[dict] = []
    dates = sorted(df["date"].unique().tolist())
    for d in dates[1:]:
        train = df[(df["date"] < d) & df["target_next_return_pct"].notna()]
        test = df[(df["date"] == d) & df["target_next_return_pct"].notna()]
        if len(train) < 30 or len(test) == 0:
            continue
        reg = RandomForestRegressor(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        )
        reg.fit(train[feat_cols], train["target_next_return_pct"].astype(float))
        pred = reg.predict(test[feat_cols])
        truth = test["target_next_return_pct"].astype(float).to_numpy()
        mae = float(np.mean(np.abs(pred - truth)))
        rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
        metrics.append(
            {
                "date": d.isoformat(),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "mae": mae,
                "rmse": rmse,
            }
        )
    return pd.DataFrame(metrics)


def _fit_membership(train: pd.DataFrame, feat_cols: list[str], random_state: int):
    y = train["target_top_n"].astype(int)
    if len(train) == 0:
        return None, 0.0
    if y.nunique() < 2:
        return None, float(y.mean())
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(train[feat_cols], y)
    return model, float(y.mean())


def _fit_return(train: pd.DataFrame, feat_cols: list[str], random_state: int):
    use = train[train["target_next_return_pct"].notna()]
    if len(use) < 30:
        return None
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(use[feat_cols], use["target_next_return_pct"].astype(float))
    return model


def _run_forecast(
    cfg: ForecastConfig,
    features: pd.DataFrame,
    supervised: pd.DataFrame,
    feat_cols: list[str],
) -> tuple[pd.DataFrame, date, date]:
    all_dates = sorted(features["date"].unique().tolist())
    if not all_dates:
        raise SystemExit("No feature rows found in data directory.")

    if cfg.forecast_date:
        forecast_date = cfg.forecast_date
        base_date = forecast_date - timedelta(days=1)
    else:
        base_date = all_dates[-1]
        forecast_date = base_date + timedelta(days=1)

    base_rows = features[features["date"] == base_date].copy()
    if base_rows.empty:
        raise SystemExit(f"No feature rows available for base date {base_date.isoformat()}.")

    train = supervised[supervised["date"] < base_date].copy()
    if train.empty:
        raise SystemExit(f"Not enough historical labels before {base_date.isoformat()} to train a forecast model.")

    clf, prior = _fit_membership(train, feat_cols, cfg.random_state)
    if clf is None:
        prob = np.full(len(base_rows), prior, dtype=float)
    else:
        prob = clf.predict_proba(base_rows[feat_cols])[:, 1]

    reg = _fit_return(train, feat_cols, cfg.random_state)
    pred_ret = reg.predict(base_rows[feat_cols]) if reg is not None else np.full(len(base_rows), np.nan)

    out = base_rows[["symbol", "date", "day_return_pct", "runup_pct", "drawdown_pct", "ret_30m_pct", "ret_60m_pct"]].copy()
    out["prob_top_n_next_day"] = prob.astype(float)
    out["pred_next_return_pct"] = pred_ret.astype(float)

    rank_return = pd.Series(pred_ret).rank(pct=True)
    if rank_return.notna().any():
        rank_return = rank_return.fillna(float(rank_return.dropna().mean()))
    else:
        rank_return = pd.Series(np.full(len(out), 0.5))
    out["ensemble_score"] = 0.65 * out["prob_top_n_next_day"] + 0.35 * rank_return.to_numpy(dtype=float)
    out = out.sort_values("ensemble_score", ascending=False).reset_index(drop=True)
    out["forecast_date"] = forecast_date.isoformat()
    out["base_date"] = base_date.isoformat()
    return out, base_date, forecast_date


def main() -> None:
    cfg = _parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    features = _build_feature_frame(cfg)
    if features.empty:
        raise SystemExit("No valid *_YYYY-MM-DD_1m.csv files found.")
    supervised = _build_supervised_frame(features, cfg.top_n)
    if supervised.empty:
        raise SystemExit("Not enough sequential dates to build supervised labels.")

    feat_cols = _feature_columns(supervised)
    wf_membership = _walkforward_membership(supervised, feat_cols, cfg.top_n, cfg.random_state)
    wf_return = _walkforward_return(supervised, feat_cols, cfg.random_state)
    forecast, base_date, forecast_date = _run_forecast(cfg, features, supervised, feat_cols)

    forecast_path = cfg.output_dir / (
        f"top_movers_forecast_{forecast_date.isoformat()}_from_{base_date.isoformat()}.csv"
    )
    summary_path = cfg.output_dir / f"top_movers_model_summary_{base_date.isoformat()}.json"
    wf_membership_path = cfg.output_dir / "top_movers_walkforward_membership_metrics.csv"
    wf_return_path = cfg.output_dir / "top_movers_walkforward_return_metrics.csv"

    forecast.to_csv(forecast_path, index=False)
    wf_membership.to_csv(wf_membership_path, index=False)
    wf_return.to_csv(wf_return_path, index=False)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(cfg.data_dir),
        "output_dir": str(cfg.output_dir),
        "top_n": cfg.top_n,
        "min_bars": cfg.min_bars,
        "random_state": cfg.random_state,
        "rows_features": int(len(features)),
        "rows_supervised": int(len(supervised)),
        "dates_features": [d.isoformat() for d in sorted(features["date"].unique().tolist())],
        "dates_supervised": [d.isoformat() for d in sorted(supervised["date"].unique().tolist())],
        "walkforward_membership_rows": int(len(wf_membership)),
        "walkforward_membership_precision_at_k_mean": (
            float(wf_membership["precision_at_k"].mean()) if not wf_membership.empty else None
        ),
        "walkforward_membership_baseline_mean": (
            float(wf_membership["baseline_positive_rate"].mean()) if not wf_membership.empty else None
        ),
        "walkforward_return_rows": int(len(wf_return)),
        "walkforward_return_mae_mean": float(wf_return["mae"].mean()) if not wf_return.empty else None,
        "walkforward_return_rmse_mean": float(wf_return["rmse"].mean()) if not wf_return.empty else None,
        "forecast_base_date": base_date.isoformat(),
        "forecast_date": forecast_date.isoformat(),
        "forecast_top_symbols": forecast.head(cfg.top_n)[
            ["symbol", "prob_top_n_next_day", "pred_next_return_pct", "ensemble_score"]
        ].to_dict(orient="records"),
        "artifacts": {
            "forecast_csv": str(forecast_path),
            "walkforward_membership_csv": str(wf_membership_path),
            "walkforward_return_csv": str(wf_return_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Features: {len(features)} rows across {features['date'].nunique()} dates")
    print(f"Supervised: {len(supervised)} rows")
    print(f"Forecast written: {forecast_path}")
    print(f"Summary written: {summary_path}")


if __name__ == "__main__":
    main()
