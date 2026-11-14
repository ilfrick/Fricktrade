# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""Crash-safe file writes.

A process killed mid-write leaves a truncated file. For `config.yaml` that means every
service fails to start; for `trader.json` it means `load_checkpoint` raises
`JSONDecodeError`, returns None, and the entire position state is lost. Writing to a
temporary file in the same directory and renaming it into place is atomic on POSIX:
readers see either the old file or the new one, never a partial one.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path | str, text: str, *, encoding: str = "utf-8") -> Path:
    """Write `text` to `path` atomically (temp file in the same dir, then rename)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        # Never leave the temp file behind on failure.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return target


def atomic_write_json(path: Path | str, payload: Any, *, indent: int = 2) -> Path:
    """Serialise `payload` to JSON and write it atomically."""
    return atomic_write_text(path, json.dumps(payload, indent=indent))
