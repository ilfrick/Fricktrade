# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from app.monitoring.metrics import BROKER_LAST_SUCCESS, BROKER_LATENCY, BROKER_REQUESTS

T = TypeVar("T")


def record_broker_call(
    broker: str,
    method: str,
    func: Callable[..., T],
    *args,
    is_success_check: Callable[[T], bool] | None = None,
    **kwargs,
) -> T:
    start = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        if is_success_check and not is_success_check(result):
            # If the check fails, treat as an error, but don't re-raise unless func explicitly failed
            BROKER_REQUESTS.labels(broker=broker, method=method, status="error").inc()
            BROKER_LATENCY.labels(broker=broker, method=method).observe(time.perf_counter() - start)
            return result # Return result even if check fails, for inspection by caller
        BROKER_REQUESTS.labels(broker=broker, method=method, status="success").inc()
        BROKER_LAST_SUCCESS.labels(broker=broker, method=method).set(time.time())
        BROKER_LATENCY.labels(broker=broker, method=method).observe(time.perf_counter() - start)
        return result
    except Exception:
        BROKER_REQUESTS.labels(broker=broker, method=method, status="error").inc()
        BROKER_LATENCY.labels(broker=broker, method=method).observe(time.perf_counter() - start)
        raise
