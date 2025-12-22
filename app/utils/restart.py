from __future__ import annotations

from datetime import datetime
from pathlib import Path


def restart_flag_path() -> Path:
    return Path("/app/config/restart.flag")


def should_restart(started_at: datetime) -> bool:
    flag = restart_flag_path()
    if not flag.exists():
        return False
    try:
        mtime = datetime.fromtimestamp(flag.stat().st_mtime)
    except OSError:
        return False
    return mtime > started_at
