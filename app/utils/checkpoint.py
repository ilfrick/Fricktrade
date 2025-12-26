from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class CheckpointConfig:
    enabled: bool
    interval_seconds: int
    dir_path: str
    max_age_hours: int
    max_files: int
    keep_history: bool


def read_checkpoint_config(cfg: dict) -> CheckpointConfig:
    checkpoint_cfg = cfg.get("checkpointing", {}) or {}
    retention = checkpoint_cfg.get("retention", {}) or {}
    return CheckpointConfig(
        enabled=bool(checkpoint_cfg.get("enabled", True)),
        interval_seconds=int(checkpoint_cfg.get("interval_seconds", 60)),
        dir_path=str(checkpoint_cfg.get("dir", "/data/checkpoints")),
        max_age_hours=int(retention.get("max_age_hours", 72)),
        max_files=int(retention.get("max_files", 200)),
        keep_history=bool(checkpoint_cfg.get("keep_history", True)),
    )


def load_checkpoint(name: str, cfg: dict) -> dict | None:
    cfg_obj = read_checkpoint_config(cfg)
    if not cfg_obj.enabled:
        return None
    path = Path(cfg_obj.dir_path) / f"{name}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        logging.warning("Checkpoint load failed for %s: %s", name, exc)
        return None


def maybe_save_checkpoint(name: str, payload: dict, cfg: dict, last_saved_at: datetime | None) -> datetime | None:
    cfg_obj = read_checkpoint_config(cfg)
    if not cfg_obj.enabled:
        return last_saved_at
    now = datetime.utcnow()
    if last_saved_at and (now - last_saved_at).total_seconds() < cfg_obj.interval_seconds:
        return last_saved_at
    base = Path(cfg_obj.dir_path)
    base.mkdir(parents=True, exist_ok=True)
    data = {"saved_at": now.isoformat(), "payload": payload}
    primary = base / f"{name}.json"
    try:
        primary.write_text(json.dumps(data, indent=2))
        if cfg_obj.keep_history:
            stamp = now.strftime("%Y%m%d%H%M%S")
            history = base / f"{name}-{stamp}.json"
            history.write_text(json.dumps(data, indent=2))
        _prune_history(base, name, cfg_obj)
        logging.info("Checkpoint saved: %s", primary)
    except Exception as exc:
        logging.warning("Checkpoint save failed for %s: %s", name, exc)
    return now


def _prune_history(base: Path, name: str, cfg: CheckpointConfig) -> None:
    if not cfg.keep_history:
        return
    history = sorted(base.glob(f"{name}-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if cfg.max_files > 0 and len(history) > cfg.max_files:
        for path in history[cfg.max_files :]:
            path.unlink(missing_ok=True)
    if cfg.max_age_hours > 0:
        cutoff = datetime.utcnow() - timedelta(hours=cfg.max_age_hours)
        for path in history:
            try:
                if datetime.utcfromtimestamp(path.stat().st_mtime) < cutoff:
                    path.unlink(missing_ok=True)
            except Exception:
                continue
