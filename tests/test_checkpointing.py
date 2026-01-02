# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from app.utils.checkpoint import load_checkpoint, maybe_save_checkpoint, read_checkpoint_config


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    cfg = {
        "checkpointing": {
            "enabled": True,
            "interval_seconds": 0,
            "dir": str(tmp_path),
            "keep_history": True,
            "retention": {"max_files": 2, "max_age_hours": 0},
        }
    }
    payload = {"alpha": 1, "beta": "ok"}
    saved_at = maybe_save_checkpoint("unit", payload, cfg, None)
    assert isinstance(saved_at, datetime)
    loaded = load_checkpoint("unit", cfg)
    assert loaded is not None
    assert loaded["payload"] == payload


def test_checkpoint_retention_by_count(tmp_path: Path) -> None:
    cfg = {
        "checkpointing": {
            "enabled": True,
            "interval_seconds": 0,
            "dir": str(tmp_path),
            "keep_history": True,
            "retention": {"max_files": 2, "max_age_hours": 0},
        }
    }
    for idx in range(3):
        maybe_save_checkpoint("retain", {"n": idx}, cfg, None)
    history = sorted(tmp_path.glob("retain-*.json"))
    assert len(history) == 2
    assert (tmp_path / "retain.json").exists()


def test_read_checkpoint_config_defaults() -> None:
    cfg = {}
    data = read_checkpoint_config(cfg)
    assert data.enabled is True
    assert data.interval_seconds == 60
    assert data.dir_path == "/data/checkpoints"
