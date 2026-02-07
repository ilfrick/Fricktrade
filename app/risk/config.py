# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from dataclasses import dataclass, field


@dataclass
class RiskConfig:
    """Typed configuration for risk management settings."""

    enabled: bool = True
    max_daily_loss_pct: float = 0.0
    max_position_size_pct: float = 100.0
    max_portfolio_leverage: float = 10.0
    max_short_exposure_pct: float = 100.0
    max_positions: int = 999
    cooldown_seconds: int = 0
    hard_stop_pct: float = 0.0
    trailing_stop_pct: float = 0.0
    circuit_breaker_drawdown_pct: float = 100.0
    # Nested configs stay as dicts for flexibility
    vol_targeting: dict = field(default_factory=dict)
    stress: dict = field(default_factory=dict)
    liquidity_haircut: dict = field(default_factory=dict)
    var: dict = field(default_factory=dict)
    exposure_caps: dict = field(default_factory=dict)
    kill_switch_profiles: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "RiskConfig":
        """Create a RiskConfig from a dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for key, value in d.items():
            if key in known:
                field_obj = cls.__dataclass_fields__[key]
                # Coerce scalar types
                if field_obj.type == "bool":
                    filtered[key] = bool(value)
                elif field_obj.type == "float":
                    try:
                        filtered[key] = float(value or 0.0)
                    except (TypeError, ValueError):
                        pass
                elif field_obj.type == "int":
                    try:
                        filtered[key] = int(value or 0)
                    except (TypeError, ValueError):
                        pass
                else:
                    filtered[key] = value
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert to a plain dict."""
        from dataclasses import asdict
        return asdict(self)
