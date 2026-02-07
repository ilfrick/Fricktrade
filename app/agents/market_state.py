# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from dataclasses import dataclass, field


@dataclass
class MarketState:
    """Typed container for market data passed through the trading pipeline.

    Mutable dataclass — strategies and enrichment mutate fields extensively.
    Thread safety is handled via deep-copies before passing to worker threads.
    """

    symbol: str = ""
    last_price: float | None = None
    prices: list[float] = field(default_factory=list)
    volumes: list[float] = field(default_factory=list)
    spread_pct: float = 0.0
    session_volume: float = 0.0
    last_bar_ts: str | None = None

    # Signal inputs
    signal_30m_return_pct: float = 0.0
    signal_60m_return_pct: float = 0.0
    signal_runup_pct: float = 0.0
    signal_drawdown_pct: float = 0.0
    signal_abs_move: float = 0.0
    signal_runup_abs: float = 0.0
    signal_drawdown_abs: float = 0.0
    signal_early_volume_pct: float = 0.0

    # Regime
    regime: int | None = None
    regime_name: str | None = None
    regime_probability: float = 0.0
    regime_probs: list[float] = field(default_factory=list)

    # Portfolio/account enrichment
    exposure_pct: float = 0.0
    short_exposure_pct: float = 0.0
    leverage: float = 0.0
    portfolio: dict = field(default_factory=dict)
    account_flags: dict = field(default_factory=dict)
    catalyst: dict | None = None
    open_orders: list = field(default_factory=list)
    market_venue: str | None = None
    market_extended: bool = False
    broker: str | None = None
    broker_override: str | None = None

    # Haircut outputs (set during sizing)
    stress_haircut_pct: float = 0.0
    liquidity_haircut_pct: float = 0.0
    max_participation: float = 0.0

    # Risk
    risk_outcome: dict = field(default_factory=dict)
    risk_disabled: bool = False

    # Strategy
    strategy_symbols: dict = field(default_factory=dict)
    qty: int = 0

    # Pipeline internals
    _decision_start: float | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "MarketState":
        """Create a MarketState from a plain dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert to a plain dict (for backward compat with dict-based APIs)."""
        from dataclasses import asdict
        return asdict(self)
