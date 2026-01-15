# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("gymnasium")

from app.learning.env import TradingEnv


def test_env_handles_short_series() -> None:
    df = pd.DataFrame(
        {
            "Close": [100.0, 101.0],
            "Volume": [1000, 1100],
        }
    )
    env = TradingEnv(df, window_size=50, initial_cash=1000.0)
    obs, _ = env.reset()
    assert obs is not None
