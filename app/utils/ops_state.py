# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def load_ops_state(path: str | None) -> dict:
    if not path:
        return {}
    state_path = Path(path)
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def ops_state_name(state: dict) -> str:
    raw = state.get("state")
    return str(raw).lower().strip()


def ops_state_is_sleeping(state: dict) -> bool:
    return ops_state_name(state) in {"sleeping", "waiting", "force_sleep"}


def ops_state_is_running(state: dict) -> bool:
    return ops_state_name(state) == "running"


def ops_state_updated_at(state: dict) -> datetime | None:
    ts = state.get("updated_at")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None
