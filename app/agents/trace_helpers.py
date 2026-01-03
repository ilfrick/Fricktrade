# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Any


def start_trace() -> dict[str, Any]:
    return {"started_at": time.perf_counter(), "stages": {}}


def mark_stage(trace: dict[str, Any], name: str, start: float, end: float | None = None) -> None:
    if not trace:
        return
    duration = (end or time.perf_counter()) - start
    trace.setdefault("stages", {})[name] = duration


@contextmanager
def stage_timer(trace: dict[str, Any], name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        mark_stage(trace, name, start)
