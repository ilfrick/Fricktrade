#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "scikit-learn is required. Install with: /home/nicola/Fricktrade/testenv/bin/python -m pip install scikit-learn"
    ) from exc

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


FILE_RE = re.compile(r"^(?P<symbol>.+)_(?P<day>\d{4}-\d{2}-\d{2})_1m\.csv$")


@dataclass
class Config:
    data_dir: Path
    output_dir: Path
    model_type: str
    top_n: int
    min_bars: int
    cutoff_bars: int
    random_state: int
    forecast_date: date | None
    entry_horizon_bars: int
    entry_warmup_bars: int
    low_zone_tol_pct: float
    rebound_target_pct: float
    entry_k: int
    xgb_n_estimators_nowcast: int
    xgb_n_estimators_entry: int
    xgb_max_depth_nowcast: int
    xgb_max_depth_entry: int
    xgb_learning_rate: float
    xgb_subsample: float
    xgb_colsample_bytree: float
    xgb_min_child_weight: float
    xgb_gamma: float
    xgb_reg_lambda: float


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate same-day top-mover nowcast + intraday low-zone entry model "
            "from *_YYYY-MM-DD_1m.csv history."
        )
    )
    parser.add_argument("--data-dir", default="/home/nicola/Fricktrade/data")
    parser.add_argument("--output-dir", default="/home/nicola/Fricktrade/data/reports/modeling")
    parser.add_argument("--model", choices=["random_forest", "xgboost"], default="random_forest")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-bars", type=int, default=60)
    parser.add_argument("--cutoff-bars", type=int, default=30, help="Use first N bars to predict same-day top movers.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--forecast-date",
        default="",
        help="Date to score using same-day early bars (YYYY-MM-DD). If omitted, uses latest date in dataset.",
    )
    parser.add_argument("--entry-horizon-bars", type=int, default=90, help="Future bars window for rebound target.")
    parser.add_argument("--entry-warmup-bars", type=int, default=20, help="Minimum seen bars before entry scoring.")
    parser.add_argument("--low-zone-tol-pct", type=float, default=0.35, help="Near-low tolerance in percent.")
    parser.add_argument("--rebound-target-pct", type=float, default=2.0, help="Required future rebound percent.")
    parser.add_argument("--entry-k", type=int, default=10, help="Top-K intraday entry candidates for precision@k.")
    parser.add_argument("--xgb-n-estimators-nowcast", type=int, default=450)
    parser.add_argument("--xgb-n-estimators-entry", type=int, default=320)
    parser.add_argument("--xgb-max-depth-nowcast", type=int, default=6)
    parser.add_argument("--xgb-max-depth-entry", type=int, default=5)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.9)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.9)
    parser.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    parser.add_argument("--xgb-gamma", type=float, default=0.0)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    args = parser.parse_args()
    forecast_date = date.fromisoformat(args.forecast_date) if args.forecast_date else None
    return Config(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        model_type=str(args.model),
        top_n=max(int(args.top_n), 1),
        min_bars=max(int(args.min_bars), 20),
        cutoff_bars=max(int(args.cutoff_bars), 5),
        random_state=int(args.random_state),
        forecast_date=forecast_date,
        entry_horizon_bars=max(int(args.entry_horizon_bars), 5),
        entry_warmup_bars=max(int(args.entry_warmup_bars), 5),
        low_zone_tol_pct=max(float(args.low_zone_tol_pct), 0.0),
        rebound_target_pct=max(float(args.rebound_target_pct), 0.1),
        entry_k=max(int(args.entry_k), 1),
        xgb_n_estimators_nowcast=max(int(args.xgb_n_estimators_nowcast), 10),
        xgb_n_estimators_entry=max(int(args.xgb_n_estimators_entry), 10),
        xgb_max_depth_nowcast=max(int(args.xgb_max_depth_nowcast), 2),
        xgb_max_depth_entry=max(int(args.xgb_max_depth_entry), 2),
        xgb_learning_rate=max(float(args.xgb_learning_rate), 1e-4),
        xgb_subsample=min(max(float(args.xgb_subsample), 0.2), 1.0),
        xgb_colsample_bytree=min(max(float(args.xgb_colsample_bytree), 0.2), 1.0),
        xgb_min_child_weight=max(float(args.xgb_min_child_weight), 0.0),
        xgb_gamma=max(float(args.xgb_gamma), 0.0),
        xgb_reg_lambda=max(float(args.xgb_reg_lambda), 0.0),
    )


def _build_classifier(cfg: Config, y: pd.Series, kind: str):
    y = y.astype(int)
    positives = int(y.sum())
    negatives = int(len(y) - positives)

    if cfg.model_type == "xgboost":
        if XGBClassifier is None:
            raise SystemExit(
                "xgboost not available. Install with "
                "/home/nicola/Fricktrade/testenv/bin/python -m pip install xgboost"
            )
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_estimators": (
                cfg.xgb_n_estimators_nowcast if kind == "nowcast" else cfg.xgb_n_estimators_entry
            ),
            "max_depth": cfg.xgb_max_depth_nowcast if kind == "nowcast" else cfg.xgb_max_depth_entry,
            "learning_rate": cfg.xgb_learning_rate,
            "subsample": cfg.xgb_subsample,
            "colsample_bytree": cfg.xgb_colsample_bytree,
            "min_child_weight": cfg.xgb_min_child_weight,
            "gamma": cfg.xgb_gamma,
            "reg_lambda": cfg.xgb_reg_lambda,
            "random_state": cfg.random_state,
            "n_jobs": -1,
        }
        if positives > 0 and negatives > 0:
            params["scale_pos_weight"] = float(negatives) / float(positives)
        return XGBClassifier(**params)

    if kind == "nowcast":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=cfg.random_state,
            n_jobs=-1,
        )

    return RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=cfg.random_state,
        n_jobs=-1,
    )


def _load_ohlcv(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None

    col_map: dict[str, str] = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in {"datetime", "timestamp", "date", "time"}:
            col_map[c] = "datetime"
        elif cl in {"open", "o"}:
            col_map[c] = "open"
        elif cl in {"high", "h"}:
            col_map[c] = "high"
        elif cl in {"low", "l"}:
            col_map[c] = "low"
        elif cl in {"close", "c"}:
            col_map[c] = "close"
        elif cl in {"volume", "v"}:
            col_map[c] = "volume"

    required = {"open", "high", "low", "close"}
    if not required.issubset(set(col_map.values())):
        return None

    df = df.rename(columns=col_map)
    if "volume" not in df.columns:
        df["volume"] = 0.0

    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df = df.sort_values("datetime")

    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if df.empty:
        return None
    return df


def _load_symbol_days(cfg: Config) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(cfg.data_dir.glob("*_????-??-??_1m.csv")):
        m = FILE_RE.match(path.name)
        if not m:
            continue
        sym = m.group("symbol")
        day = date.fromisoformat(m.group("day"))
        df = _load_ohlcv(path)
        if df is None or len(df) < cfg.min_bars:
            continue
        rows.append({"symbol": sym, "date": day, "frame": df})
    return rows


def _nowcast_features(df: pd.DataFrame, cutoff: int) -> dict:
    c = df["close"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)

    n = min(cutoff, len(c))
    c0 = c[:n]
    h0 = h[:n]
    l0 = l[:n]
    v0 = v[:n]

    open_px = float(c0[0])
    last_px = float(c0[-1])
    high_px = float(np.max(h0))
    low_px = float(np.min(l0))
    vol_sum = float(np.sum(v0))

    ret = np.diff(c0) / np.maximum(c0[:-1], 1e-9)
    ret_std = float(np.std(ret) * 100.0) if len(ret) else 0.0
    ret_mean = float(np.mean(ret) * 100.0) if len(ret) else 0.0

    vwap_dist = 0.0
    if vol_sum > 0:
        vwap = float(np.sum(c0 * v0) / vol_sum)
        if vwap > 0:
            vwap_dist = (last_px - vwap) / vwap * 100.0

    idx = np.arange(len(c0), dtype=float)
    norm = c0 / max(open_px, 1e-9)
    slope = float(np.polyfit(idx, norm, 1)[0]) if len(c0) >= 3 else 0.0

    pos_in_range = 0.5
    if high_px > low_px:
        pos_in_range = (last_px - low_px) / (high_px - low_px)

    return {
        "bars_seen": int(n),
        "ret_from_open_pct": (last_px - open_px) / max(open_px, 1e-9) * 100.0,
        "runup_pct": (high_px - open_px) / max(open_px, 1e-9) * 100.0,
        "drawdown_pct": (low_px - open_px) / max(open_px, 1e-9) * 100.0,
        "range_pct": (high_px - low_px) / max(open_px, 1e-9) * 100.0,
        "ret_std_pct": ret_std,
        "ret_mean_pct": ret_mean,
        "vol_sum": vol_sum,
        "vol_per_bar": vol_sum / max(len(v0), 1),
        "vwap_dist_pct": vwap_dist,
        "trend_slope": slope,
        "pos_in_range": float(pos_in_range),
    }


def _build_nowcast_dataset(symbol_days: list[dict], cfg: Config) -> pd.DataFrame:
    if not symbol_days:
        return pd.DataFrame()

    # Same-day ranking label (top-N by full-day return).
    per_day_rows: dict[date, list[dict]] = {}
    for row in symbol_days:
        d = row["date"]
        df = row["frame"]
        c = df["close"].to_numpy(dtype=float)
        day_ret = (float(c[-1]) - float(c[0])) / max(float(c[0]), 1e-9) * 100.0
        per_day_rows.setdefault(d, []).append({"symbol": row["symbol"], "day_return_pct": day_ret})

    top_by_day: dict[date, set[str]] = {}
    for d, rows in per_day_rows.items():
        ranked = sorted(rows, key=lambda x: x["day_return_pct"], reverse=True)
        top_by_day[d] = {x["symbol"] for x in ranked[: cfg.top_n]}

    out: list[dict] = []
    for row in symbol_days:
        d = row["date"]
        sym = row["symbol"]
        df = row["frame"]
        if len(df) < cfg.cutoff_bars:
            continue
        feats = _nowcast_features(df, cfg.cutoff_bars)
        out.append(
            {
                "date": d,
                "symbol": sym,
                "target_top_n": 1 if sym in top_by_day.get(d, set()) else 0,
                **feats,
            }
        )

    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_values(["date", "symbol"]).reset_index(drop=True)


def _build_entry_dataset(symbol_days: list[dict], cfg: Config) -> pd.DataFrame:
    out: list[dict] = []
    horizon = cfg.entry_horizon_bars
    warmup = cfg.entry_warmup_bars
    low_tol = cfg.low_zone_tol_pct
    rebound = cfg.rebound_target_pct

    for row in symbol_days:
        d = row["date"]
        sym = row["symbol"]
        df = row["frame"]
        c = df["close"].to_numpy(dtype=float)
        h = df["high"].to_numpy(dtype=float)
        l = df["low"].to_numpy(dtype=float)
        v = df["volume"].to_numpy(dtype=float)
        ts = df["datetime"] if "datetime" in df.columns else None

        if len(c) <= warmup + horizon:
            continue

        day_low = float(np.min(l))
        eps = 1e-9

        for t in range(warmup, len(c) - horizon):
            current = float(c[t])
            past_c = c[: t + 1]
            past_h = h[: t + 1]
            past_l = l[: t + 1]
            past_v = v[: t + 1]
            future_h = h[t + 1 : t + 1 + horizon]

            if current <= 0:
                continue

            # Label: near eventual day low and followed by meaningful rebound soon.
            dist_to_day_low_pct = (current - day_low) / max(day_low, eps) * 100.0
            future_max = float(np.max(future_h)) if len(future_h) else current
            future_rebound_pct = (future_max - current) / max(current, eps) * 100.0
            target = 1 if (dist_to_day_low_pct <= low_tol and future_rebound_pct >= rebound) else 0

            ret = np.diff(past_c) / np.maximum(past_c[:-1], eps)
            ret_std = float(np.std(ret) * 100.0) if len(ret) else 0.0

            high_so_far = float(np.max(past_h))
            low_so_far = float(np.min(past_l))
            open_px = float(past_c[0])
            vol_recent = float(np.sum(past_v[-5:]))
            vol_avg20 = float(np.mean(past_v[-20:])) if len(past_v) >= 20 else float(np.mean(past_v))
            vol_spike = vol_recent / max(vol_avg20 * 5.0, eps)
            mom_5 = (current - float(past_c[max(0, len(past_c) - 6)])) / max(current, eps) * 100.0 if len(past_c) >= 6 else 0.0
            mom_20 = (current - float(past_c[max(0, len(past_c) - 21)])) / max(current, eps) * 100.0 if len(past_c) >= 21 else 0.0

            vwap_dist = 0.0
            vol_sum = float(np.sum(past_v))
            if vol_sum > 0:
                vwap = float(np.sum(past_c * past_v) / vol_sum)
                if vwap > 0:
                    vwap_dist = (current - vwap) / vwap * 100.0

            out.append(
                {
                    "date": d,
                    "symbol": sym,
                    "minute_index": int(t),
                    "ts": ts.iloc[t].isoformat() if ts is not None and pd.notna(ts.iloc[t]) else "",
                    "target_entry": int(target),
                    "dist_to_day_low_pct": dist_to_day_low_pct,
                    "future_rebound_pct": future_rebound_pct,
                    "ret_from_open_pct": (current - open_px) / max(open_px, eps) * 100.0,
                    "pullback_from_high_pct": (current - high_so_far) / max(high_so_far, eps) * 100.0,
                    "dist_to_low_so_far_pct": (current - low_so_far) / max(low_so_far, eps) * 100.0,
                    "ret_std_pct": ret_std,
                    "mom_5_pct": mom_5,
                    "mom_20_pct": mom_20,
                    "vol_spike": vol_spike,
                    "vwap_dist_pct": vwap_dist,
                    "bars_seen": int(t + 1),
                }
            )

    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_values(["date", "symbol", "minute_index"]).reset_index(drop=True)


def _feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    skip = {"date", "symbol", "ts", target_col}
    # Prevent target leakage in entry models: these fields require future/full-day knowledge.
    leakage = {"dist_to_day_low_pct", "future_rebound_pct"}
    return [c for c in df.columns if c not in skip and c not in leakage]


def _walkforward_nowcast(df: pd.DataFrame, feat_cols: list[str], cfg: Config) -> pd.DataFrame:
    rows: list[dict] = []
    dates = sorted(df["date"].unique().tolist())
    for d in dates[1:]:
        train = df[df["date"] < d]
        test = df[df["date"] == d]
        if train.empty or test.empty:
            continue
        y = train["target_top_n"].astype(int)
        if y.nunique() < 2:
            continue
        model = _build_classifier(cfg, y, kind="nowcast")
        model.fit(train[feat_cols], y)
        proba = model.predict_proba(test[feat_cols])[:, 1]
        ranked = test.assign(score=proba).sort_values("score", ascending=False)
        k = min(cfg.top_n, len(ranked))
        rows.append(
            {
                "date": d.isoformat(),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "k": int(k),
                "precision_at_k": float(ranked.head(k)["target_top_n"].mean()),
                "baseline_positive_rate": float(test["target_top_n"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _walkforward_entry(df: pd.DataFrame, feat_cols: list[str], cfg: Config) -> pd.DataFrame:
    rows: list[dict] = []
    dates = sorted(df["date"].unique().tolist())
    for d in dates[1:]:
        train = df[df["date"] < d]
        test = df[df["date"] == d]
        if train.empty or test.empty:
            continue
        y = train["target_entry"].astype(int)
        if y.nunique() < 2:
            continue
        model = _build_classifier(cfg, y, kind="entry")
        model.fit(train[feat_cols], y)
        proba = model.predict_proba(test[feat_cols])[:, 1]
        ranked = test.assign(score=proba).sort_values("score", ascending=False)
        k = min(cfg.entry_k, len(ranked))
        positives = int(test["target_entry"].sum())
        hit_k = int(ranked.head(k)["target_entry"].sum())
        recall_at_k = float(hit_k / positives) if positives > 0 else np.nan
        rows.append(
            {
                "date": d.isoformat(),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "k": int(k),
                "precision_at_k": float(ranked.head(k)["target_entry"].mean()),
                "baseline_positive_rate": float(test["target_entry"].mean()),
                "positives": positives,
                "recall_at_k": recall_at_k,
            }
        )
    return pd.DataFrame(rows)


def _fit_and_score_latest(
    df: pd.DataFrame,
    feat_cols: list[str],
    target_col: str,
    score_date: date,
    cfg: Config,
    kind: str,
) -> pd.DataFrame:
    train = df[df["date"] < score_date]
    test = df[df["date"] == score_date].copy()
    if train.empty or test.empty:
        return pd.DataFrame()
    y = train[target_col].astype(int)
    if y.nunique() < 2:
        return pd.DataFrame()
    model = _build_classifier(cfg, y, kind=kind)
    model.fit(train[feat_cols], y)
    test["score"] = model.predict_proba(test[feat_cols])[:, 1]
    return test.sort_values("score", ascending=False).reset_index(drop=True)


def main() -> None:
    cfg = _parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    symbol_days = _load_symbol_days(cfg)
    if not symbol_days:
        raise SystemExit("No valid symbol-day CSV files found.")

    nowcast_df = _build_nowcast_dataset(symbol_days, cfg)
    if nowcast_df.empty:
        raise SystemExit("Nowcast dataset is empty.")

    entry_df = _build_entry_dataset(symbol_days, cfg)
    if entry_df.empty:
        raise SystemExit("Entry dataset is empty.")

    dates = sorted(nowcast_df["date"].unique().tolist())
    score_date = cfg.forecast_date or dates[-1]
    if score_date not in set(dates):
        raise SystemExit(f"Forecast date {score_date.isoformat()} not available in dataset.")

    nowcast_feats = _feature_columns(nowcast_df, "target_top_n")
    entry_feats = _feature_columns(entry_df, "target_entry")

    wf_nowcast = _walkforward_nowcast(nowcast_df, nowcast_feats, cfg)
    wf_entry = _walkforward_entry(entry_df, entry_feats, cfg)

    score_nowcast = _fit_and_score_latest(
        nowcast_df, nowcast_feats, "target_top_n", score_date, cfg, kind="nowcast"
    )
    score_entry = _fit_and_score_latest(
        entry_df, entry_feats, "target_entry", score_date, cfg, kind="entry"
    )

    if not score_nowcast.empty:
        score_nowcast = score_nowcast.rename(columns={"target_top_n": "actual_top_n"})
    if not score_entry.empty:
        score_entry = score_entry.rename(columns={"target_entry": "actual_entry"})

    nowcast_path = cfg.output_dir / f"same_day_top_movers_scores_{score_date.isoformat()}.csv"
    entry_path = cfg.output_dir / f"same_day_entry_scores_{score_date.isoformat()}.csv"
    wf_nowcast_path = cfg.output_dir / "same_day_top_movers_walkforward.csv"
    wf_entry_path = cfg.output_dir / "same_day_entry_walkforward.csv"
    summary_path = cfg.output_dir / f"same_day_model_summary_{score_date.isoformat()}.json"

    score_nowcast.to_csv(nowcast_path, index=False)
    score_entry.to_csv(entry_path, index=False)
    wf_nowcast.to_csv(wf_nowcast_path, index=False)
    wf_entry.to_csv(wf_entry_path, index=False)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "score_date": score_date.isoformat(),
        "config": {
            "model_type": cfg.model_type,
            "top_n": cfg.top_n,
            "cutoff_bars": cfg.cutoff_bars,
            "entry_horizon_bars": cfg.entry_horizon_bars,
            "entry_warmup_bars": cfg.entry_warmup_bars,
            "low_zone_tol_pct": cfg.low_zone_tol_pct,
            "rebound_target_pct": cfg.rebound_target_pct,
            "entry_k": cfg.entry_k,
            "xgb_n_estimators_nowcast": cfg.xgb_n_estimators_nowcast,
            "xgb_n_estimators_entry": cfg.xgb_n_estimators_entry,
            "xgb_max_depth_nowcast": cfg.xgb_max_depth_nowcast,
            "xgb_max_depth_entry": cfg.xgb_max_depth_entry,
            "xgb_learning_rate": cfg.xgb_learning_rate,
            "xgb_subsample": cfg.xgb_subsample,
            "xgb_colsample_bytree": cfg.xgb_colsample_bytree,
            "xgb_min_child_weight": cfg.xgb_min_child_weight,
            "xgb_gamma": cfg.xgb_gamma,
            "xgb_reg_lambda": cfg.xgb_reg_lambda,
        },
        "dataset": {
            "symbol_days": int(len(symbol_days)),
            "nowcast_rows": int(len(nowcast_df)),
            "entry_rows": int(len(entry_df)),
            "date_min": str(min(dates)),
            "date_max": str(max(dates)),
        },
        "walkforward": {
            "nowcast_rows": int(len(wf_nowcast)),
            "nowcast_precision_at_k_mean": float(wf_nowcast["precision_at_k"].mean()) if not wf_nowcast.empty else None,
            "nowcast_baseline_mean": float(wf_nowcast["baseline_positive_rate"].mean()) if not wf_nowcast.empty else None,
            "entry_rows": int(len(wf_entry)),
            "entry_precision_at_k_mean": float(wf_entry["precision_at_k"].mean()) if not wf_entry.empty else None,
            "entry_baseline_mean": float(wf_entry["baseline_positive_rate"].mean()) if not wf_entry.empty else None,
            "entry_recall_at_k_mean": float(wf_entry["recall_at_k"].dropna().mean()) if (not wf_entry.empty and wf_entry["recall_at_k"].notna().any()) else None,
        },
        "latest_scores": {
            "nowcast_rows": int(len(score_nowcast)),
            "entry_rows": int(len(score_entry)),
            "top_nowcast_symbols": score_nowcast[["symbol", "score"]].head(cfg.top_n).to_dict(orient="records") if not score_nowcast.empty else [],
            "top_entry_candidates": score_entry[["symbol", "ts", "minute_index", "score"]].head(cfg.entry_k).to_dict(orient="records") if not score_entry.empty else [],
        },
        "artifacts": {
            "same_day_top_movers_scores": str(nowcast_path),
            "same_day_entry_scores": str(entry_path),
            "same_day_top_movers_walkforward": str(wf_nowcast_path),
            "same_day_entry_walkforward": str(wf_entry_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Summary: {summary_path}")
    print(f"Nowcast scores: {nowcast_path}")
    print(f"Entry scores: {entry_path}")


if __name__ == "__main__":
    main()
