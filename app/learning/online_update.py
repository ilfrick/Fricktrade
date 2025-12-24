from __future__ import annotations

import copy
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch

from app.learning.train_rl import train_from_config
from app.utils.restart import should_restart


def _lock_path(cfg: dict) -> Path:
    data_dir = cfg.get("data", {}).get("output_dir", "/data")
    return Path(data_dir) / "online_learner.lock"


def _read_lock(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_lock(path: Path, owner: str) -> None:
    payload = {"owner": owner, "updated_at": datetime.utcnow().isoformat()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _lock_is_fresh(lock: dict, max_age_minutes: int = 10) -> bool:
    ts = lock.get("updated_at")
    if not ts:
        return False
    try:
        updated = datetime.fromisoformat(ts)
    except Exception:
        return False
    return datetime.utcnow() - updated <= timedelta(minutes=max_age_minutes)


def _ensure_exclusive(cfg: dict) -> bool:
    path = _lock_path(cfg)
    lock = _read_lock(path)
    gpu_available = torch.cuda.is_available()
    owner = "gpu" if gpu_available else "cpu"
    if lock:
        lock_owner = lock.get("owner")
        if lock_owner == "gpu" and owner == "cpu" and _lock_is_fresh(lock):
            logging.info("GPU learner active; exiting CPU learner.")
            return False
        if lock_owner == "cpu" and owner == "gpu":
            logging.info("GPU learner taking over from CPU learner.")
        if lock_owner == owner:
            _write_lock(path, owner)
            return True
        if _lock_is_fresh(lock):
            if owner == "cpu":
                logging.info("Learner lock held by %s; exiting CPU learner.", lock_owner)
                return False
    _write_lock(path, owner)
    return True


def run_online_updates(cfg: dict) -> None:
    learning_cfg = cfg.get("learning", {})
    online_cfg = learning_cfg.get("online", {})
    if not online_cfg.get("enabled"):
        logging.info("Online updates disabled in config")
        return

    interval_minutes = int(online_cfg.get("update_interval_minutes", 60))
    timesteps = int(online_cfg.get("timesteps", 1000))
    eval_split = float(online_cfg.get("eval_split", 0.1))
    resume = bool(learning_cfg.get("training", {}).get("resume", True))

    started_at = datetime.utcnow()
    while True:
        if not _ensure_exclusive(cfg):
            return
        if should_restart(started_at):
            logging.info("Restart requested; exiting online updates.")
            break
        loop_cfg = copy.deepcopy(cfg)
        loop_cfg.setdefault("learning", {}).setdefault("training", {})
        loop_cfg["learning"]["training"]["timesteps"] = timesteps
        loop_cfg["learning"]["training"]["eval_split"] = eval_split
        logging.info("Starting online update (%d timesteps)", timesteps)
        train_from_config(loop_cfg, resume=resume)
        logging.info("Online update complete; sleeping %d minutes", interval_minutes)
        time.sleep(interval_minutes * 60)
