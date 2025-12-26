from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from app.monitoring.metrics import BROKER_LAST_SUCCESS, BROKER_LATENCY, BROKER_REQUESTS

T = TypeVar("T")


def record_broker_call(broker: str, method: str, func: Callable[..., T], *args, **kwargs) -> T:
    start = time.perf_counter()
    try:
        result = func(*args, **kwargs)
    except Exception:
        BROKER_REQUESTS.labels(broker=broker, method=method, status="error").inc()
        BROKER_LATENCY.labels(broker=broker, method=method).observe(time.perf_counter() - start)
        raise
    BROKER_REQUESTS.labels(broker=broker, method=method, status="success").inc()
    BROKER_LAST_SUCCESS.labels(broker=broker, method=method).set(time.time())
    BROKER_LATENCY.labels(broker=broker, method=method).observe(time.perf_counter() - start)
    return result
