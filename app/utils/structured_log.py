# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import json
import logging


class StructuredLogger:
    """Emits structured JSON log messages for critical trading events."""

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def event(self, level: str, event: str, **kw) -> None:
        """Log a structured event as JSON.

        Args:
            level: Log level (info, warning, error, debug).
            event: Event name (e.g. "trade_executed", "risk_blocked").
            **kw: Arbitrary key-value pairs to include in the log entry.
        """
        msg = json.dumps({"event": event, **kw}, default=str)
        getattr(self._logger, level)(msg)
