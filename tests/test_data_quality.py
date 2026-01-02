# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from datetime import datetime

import pandas as pd

from app.data.quality import apply_adjustments, validate_ohlcv


def _df_from_rows(rows):
    df = pd.DataFrame(rows)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df.set_index("Datetime").sort_index()


def test_validate_ohlcv_flags_issues():
    df = _df_from_rows(
        [
            {
                "Datetime": "2024-01-01 10:00:00",
                "Open": 10,
                "High": 9,
                "Low": 8,
                "Close": -1,
                "Volume": 0,
            },
            {
                "Datetime": "2024-01-01 10:05:00",
                "Open": 10,
                "High": 12,
                "Low": 9,
                "Close": 11,
                "Volume": 100,
            },
        ]
    )
    result = validate_ohlcv(df, "1m", {"gap_multiplier": 2, "outlier_zscore": 2})
    summary = result["summary"]
    issues = result["issues"]
    assert summary["total_issues"] > 0
    assert "negative_price" in issues
    assert "high_low_mismatch" in issues
    assert "zero_volume" in issues


def test_apply_adjustments_split_and_dividend():
    df = _df_from_rows(
        [
            {
                "Datetime": "2024-01-01",
                "Open": 100,
                "High": 110,
                "Low": 90,
                "Close": 100,
                "Volume": 1000,
            },
            {
                "Datetime": "2024-01-02",
                "Open": 120,
                "High": 130,
                "Low": 110,
                "Close": 120,
                "Volume": 800,
            },
        ]
    )
    adjustments = pd.DataFrame(
        [
            {
                "Datetime": "2024-01-02",
                "split_ratio": 2.0,
                "dividend": 10.0,
            }
        ]
    )
    adjusted = apply_adjustments(df, adjustments)
    first = adjusted.loc[datetime(2024, 1, 1)]
    # Split halves prices and doubles volume, then dividend factor (close 120, dividend 10) -> 0.9167
    expected_factor = (120 - 10) / 120 / 2.0
    assert abs(first["Close"] - 100 * expected_factor) < 1e-6
    assert abs(first["Volume"] - 1000 * 2.0) < 1e-6
