# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from dataclasses import dataclass, field


@dataclass
class ExecutionConfig:
    """Typed configuration for execution settings."""

    routing: dict = field(default_factory=dict)
    retry: dict = field(default_factory=dict)
    open_orders: dict = field(default_factory=dict)
    algos: dict = field(default_factory=dict)
    impact: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "ExecutionConfig":
        """Create an ExecutionConfig from a dict, ignoring unknown keys."""
        result = {}
        raw_routing = d.get("brokers", {}).get("routing", {})
        if raw_routing:
            result["routing"] = raw_routing
        for key in ("retry", "open_orders", "algos", "impact"):
            if key in d:
                result[key] = d[key]
        return cls(**result)

    def to_dict(self) -> dict:
        """Convert to a plain dict."""
        from dataclasses import asdict
        return asdict(self)
