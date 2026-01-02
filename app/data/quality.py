# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import json

import pandas as pd


@dataclass
class QualityIssue:
    key: str
    count: int
    sample: list[str]


def apply_adjustments(df: pd.DataFrame, adjustments: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or adjustments is None or adjustments.empty:
        return df
    adj = df.copy()
    adj = adj.sort_index()
    adjustments = adjustments.copy()
    if "Datetime" in adjustments.columns:
        adjustments["Datetime"] = pd.to_datetime(adjustments["Datetime"], utc=True, errors="coerce")
        adjustments = adjustments.dropna(subset=["Datetime"])
        adjustments = adjustments.set_index("Datetime")
    else:
        adjustments.index = pd.to_datetime(adjustments.index, utc=True, errors="coerce")
    adjustments = adjustments.sort_index()
    for ts, row in adjustments.iterrows():
        ts = _to_naive(ts)
        split_ratio = float(row.get("split_ratio", 0.0) or 0.0)
        dividend = float(row.get("dividend", 0.0) or 0.0)
        if split_ratio > 0:
            mask = adj.index < ts
            for col in ("Open", "High", "Low", "Close"):
                if col in adj.columns:
                    adj.loc[mask, col] = adj.loc[mask, col] / split_ratio
            if "Volume" in adj.columns:
                adj.loc[mask, "Volume"] = adj.loc[mask, "Volume"] * split_ratio
        if dividend > 0:
            if "Close" not in adj.columns:
                continue
            try:
                close_at = float(adj.loc[ts, "Close"])
            except Exception:
                continue
            if close_at <= 0:
                continue
            factor = (close_at - dividend) / close_at
            if factor <= 0:
                continue
            mask = adj.index < ts
            for col in ("Open", "High", "Low", "Close"):
                if col in adj.columns:
                    adj.loc[mask, col] = adj.loc[mask, col] * factor
    return adj


def validate_ohlcv(df: pd.DataFrame, interval: str, cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    issues: dict[str, QualityIssue] = {}
    if df is None or df.empty:
        issues["empty"] = QualityIssue("empty", 1, [])
        return {"summary": _issue_summary(issues), "issues": _issue_payload(issues)}
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        issues["missing_columns"] = QualityIssue("missing_columns", len(missing), missing)
        return {"summary": _issue_summary(issues), "issues": _issue_payload(issues)}

    nan_rows = df[required].isna().any(axis=1)
    if nan_rows.any():
        sample = [str(idx) for idx in df.index[nan_rows][:5]]
        issues["nan_rows"] = QualityIssue("nan_rows", int(nan_rows.sum()), sample)

    negative_price = (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    if negative_price.any():
        sample = [str(idx) for idx in df.index[negative_price][:5]]
        issues["negative_price"] = QualityIssue("negative_price", int(negative_price.sum()), sample)

    bad_high_low = (df["High"] < df[["Open", "Close"]].max(axis=1)) | (
        df["Low"] > df[["Open", "Close"]].min(axis=1)
    )
    if bad_high_low.any():
        sample = [str(idx) for idx in df.index[bad_high_low][:5]]
        issues["high_low_mismatch"] = QualityIssue("high_low_mismatch", int(bad_high_low.sum()), sample)

    zero_volume = df["Volume"] <= 0
    if zero_volume.any():
        sample = [str(idx) for idx in df.index[zero_volume][:5]]
        issues["zero_volume"] = QualityIssue("zero_volume", int(zero_volume.sum()), sample)

    gap_multiplier = float(cfg.get("gap_multiplier", 3.0))
    expected = _expected_delta(interval)
    if expected is not None and len(df.index) > 1:
        deltas = df.index.to_series().diff().dropna()
        gaps = deltas[deltas > expected * gap_multiplier]
        if not gaps.empty:
            sample = [str(idx) for idx in gaps.index[:5]]
            issues["gaps"] = QualityIssue("gaps", int(gaps.count()), sample)

    outlier_z = float(cfg.get("outlier_zscore", 6.0))
    returns = df["Close"].pct_change().dropna()
    if not returns.empty:
        mean = returns.mean()
        std = returns.std()
        if std and std > 0:
            zscores = (returns - mean) / std
            outliers = zscores[zscores.abs() > outlier_z]
            if not outliers.empty:
                sample = [str(idx) for idx in outliers.index[:5]]
                issues["return_outliers"] = QualityIssue("return_outliers", int(outliers.count()), sample)

    return {"summary": _issue_summary(issues), "issues": _issue_payload(issues)}


def write_quality_report(report: dict, path: str) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _expected_delta(interval: str) -> timedelta | None:
    if interval.endswith("m"):
        return timedelta(minutes=int(interval[:-1]))
    if interval.endswith("h"):
        return timedelta(hours=int(interval[:-1]))
    if interval.endswith("d"):
        return timedelta(days=int(interval[:-1]))
    return None


def _to_naive(ts):
    if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
        return ts.tz_convert(None)
    return ts


def _issue_summary(issues: dict[str, QualityIssue]) -> dict:
    return {
        "total_issues": sum(issue.count for issue in issues.values()),
        "issue_types": sorted(issues.keys()),
    }


def _issue_payload(issues: dict[str, QualityIssue]) -> dict:
    payload = {}
    for key, issue in issues.items():
        payload[key] = {"count": issue.count, "sample": issue.sample}
    return payload
