# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
#
# Tests for two-phase symbol dispatch — position-holder barrier.
# Stdlib-only; no prometheus_client or tensorflow required.

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as fut_wait


# ---------------------------------------------------------------------------
# Two-phase symbol dispatch — position-holder barrier
# ---------------------------------------------------------------------------

class TestTwoPhaseSymbolDispatch:
    """Verify position holders are fully processed before new-entry candidates."""

    def test_holders_and_non_holders_partitioned_correctly(self):
        symbols   = ["AAPL", "MSFT", "HELD1", "HELD2", "GOOG"]
        positions = {"HELD1": {"qty": 10}, "HELD2": {"qty": 5}}
        holders     = [s for s in symbols if s     in positions]
        non_holders = [s for s in symbols if s not in positions]
        assert set(holders)     == {"HELD1", "HELD2"}
        assert set(non_holders) == {"AAPL", "MSFT", "GOOG"}
        assert not set(holders) & set(non_holders)

    def test_barrier_enforced__no_non_holder_starts_before_holders_complete(self):
        holder_done              = threading.Event()
        non_holder_started_early = threading.Event()

        def run_holder(sym):
            time.sleep(0.05)
            holder_done.set()

        def run_non_holder(sym):
            if not holder_done.is_set():
                non_holder_started_early.set()

        executor = ThreadPoolExecutor(max_workers=4)
        try:
            fut_wait([executor.submit(run_holder, s) for s in ["HELD1", "HELD2"]])
            fut_wait([executor.submit(run_non_holder, s) for s in ["AAPL", "MSFT", "GOOG"]])
        finally:
            executor.shutdown(wait=True)

        assert not non_holder_started_early.is_set(), (
            "A non-holder started before all holders completed"
        )

    def test_no_positions_all_run_as_non_holders(self):
        symbols   = ["AAPL", "MSFT", "GOOG"]
        positions = {}
        holders     = [s for s in symbols if s     in positions]
        non_holders = [s for s in symbols if s not in positions]
        assert holders == []
        assert set(non_holders) == set(symbols)

    def test_all_positions_all_run_as_holders(self):
        symbols   = ["AAPL", "MSFT"]
        positions = {"AAPL": {"qty": 5}, "MSFT": {"qty": 3}}
        holders     = [s for s in symbols if s     in positions]
        non_holders = [s for s in symbols if s not in positions]
        assert set(holders) == set(symbols)
        assert non_holders == []
