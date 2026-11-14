# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""Alpha gate — refuse to open positions on an unvalidated strategy configuration.

Motivation (2026-08-19 audit): the system's default state was "find a reason to
buy". It traded continuously on a configuration with no measured edge and lost
money slowly. Cash outperformed every strategy variant tested.

This inverts the default: **flat unless proven otherwise**. New entries require a
recorded hold-out pass whose fingerprint matches the strategy config actually
running. Exits are never blocked — an unvalidated system must still be able to
manage and close the positions it already holds.

The fingerprint deliberately covers only the parameters that change trading
behaviour, so unrelated edits (log levels, dashboards, broker credentials) do not
invalidate a passing result, while any change to entry/exit/risk logic does.

Record is written by `scripts/alpha_validation.py holdout`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

DEFAULT_RECORD_PATH = "/data/reports/alpha_validation.json"

# Config paths that materially change what the system trades. Keep this list in
# sync with anything that alters entry, exit, sizing, or cost assumptions.
_FINGERPRINT_KEYS = [
    "strategy.names",
    "strategy.combine",
    "strategy.weights",
    "strategy.min_conviction",
    "strategy.single_sided_conviction_multiplier",
    "strategy.params.crypto_mean_reversion",
    "strategy.params.crypto_momentum",
    "strategy.params.min_hold_minutes",
    "strategy.params.position_horizon_minutes",
    "strategy.params.alpha_decay_exit",
    "strategy.params.regime_hold_minutes",
    "data.interval",
    "risk.crypto.hard_stop_pct",
    "risk.crypto.trailing_stop_pct",
    "risk.crypto.take_profit_pct",
    "risk.crypto.atr_stop_mult",
    "risk.partial_take_profit_pct",
    "risk.partial_take_profit_ratio",
    "risk.vol_targeting",
]


def _extract(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def config_fingerprint(cfg: dict) -> str:
    """Stable short hash of the trading-relevant configuration."""
    snapshot = {k: _extract(cfg, k) for k in _FINGERPRINT_KEYS}
    raw = json.dumps(snapshot, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class GateDecision:
    blocked: bool
    reason: str

    def __bool__(self) -> bool:      # truthy when entries are blocked
        return self.blocked


def _gate_cfg(cfg: dict) -> dict:
    return (cfg.get("strategy", {}) or {}).get("alpha_gate", {}) or {}


def record_path(cfg: dict) -> Path:
    return Path(str(_gate_cfg(cfg).get("record_path", DEFAULT_RECORD_PATH)))


def load_record(cfg: dict) -> dict | None:
    path = record_path(cfg)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.warning("Alpha gate: could not read %s: %s", path, exc)
        return None


def check_entries(cfg: dict) -> GateDecision:
    """Return a GateDecision for opening NEW positions.

    Fails closed: if no matching passing record exists, entries are blocked. A system
    with no evidence of edge should not be deploying capital.

    `enabled` defaults to True when the block is absent. That default is load-bearing:
    the config validator allows key deletion, so with a False default, deleting a few
    lines from config.yaml would silently open the gate with reason "gate_disabled" —
    indistinguishable from a deliberate opt-out.
    """
    gate = _gate_cfg(cfg)
    if not gate.get("enabled", True):
        return GateDecision(False, "gate_disabled")

    record = load_record(cfg)
    if record is None:
        return GateDecision(True, "no_validation_record")
    if not record.get("passed", False):
        return GateDecision(True, "validation_failed")

    expected = config_fingerprint(cfg)
    actual = str(record.get("fingerprint", ""))
    if actual != expected:
        return GateDecision(True, "config_changed_since_validation")
    return GateDecision(False, "validated")


def write_record(cfg: dict, *, passed: bool, metrics: dict, path: Path | None = None) -> Path:
    """Persist a hold-out verdict for the current configuration."""
    from datetime import datetime, timezone

    target = path or record_path(cfg)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        # The configured path is the in-container mount (/data/...), which is not
        # writable when running on the host. Fall back to the repo's own data
        # directory, which is the same volume from the container's point of view.
        repo_data = Path(__file__).resolve().parents[2] / "data" / "reports"
        repo_data.mkdir(parents=True, exist_ok=True)
        target = repo_data / target.name
        logging.warning(
            "Alpha gate: %s not writable from here; wrote record to %s instead "
            "(same volume as /data inside the container).", path or record_path(cfg), target
        )
    payload = {
        "fingerprint": config_fingerprint(cfg),
        "passed": bool(passed),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    target.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return target
