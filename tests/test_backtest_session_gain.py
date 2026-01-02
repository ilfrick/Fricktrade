# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from datetime import datetime

import pandas as pd

from app.backtest.agent_engine import _SymbolState


def test_session_prev_close_tracks_prior_day():
    state = _SymbolState(max_len=10)
    state.update(datetime(2024, 1, 1, 9, 30), _row(100.0))
    state.update(datetime(2024, 1, 1, 9, 35), _row(105.0))
    state.update(datetime(2024, 1, 2, 9, 30), _row(90.0))
    assert state.prev_close == 105.0
    assert state.session_open == 90.0


def _row(close: float) -> pd.Series:
    return pd.Series(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": 1000.0,
        }
    )
