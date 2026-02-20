# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from dataclasses import dataclass, field


@dataclass
class StrategyConfig:
    """Typed configuration for strategy settings."""

    name: str = "intraday_momentum"
    names: list[str] = field(default_factory=lambda: ["intraday_momentum"])
    combine: str = "priority"
    params: dict = field(default_factory=dict)
    fee_aware: dict = field(default_factory=dict)
    signal_bias_guard: dict = field(default_factory=dict)
    performance: dict = field(default_factory=dict)
    min_conviction: float = 0.0
    single_sided_conviction_multiplier: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyConfig":
        """Create a StrategyConfig from a dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert to a plain dict."""
        from dataclasses import asdict
        return asdict(self)
