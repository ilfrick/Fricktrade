from __future__ import annotations

import copy
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch

from app.learning.train_rl import train_from_config
from app.utils.checkpoint import load_checkpoint, maybe_save_checkpoint
from app.utils.restart import should_restart
from app.utils.ops_state import load_ops_state, ops_state_is_running, ops_state_is_sleeping


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


def _ensure_exclusive(cfg: dict) -> tuple[bool, str]:
    path = _lock_path(cfg)
    lock = _read_lock(path)
    gpu_available = torch.cuda.is_available()
    owner = "gpu" if gpu_available else "cpu"
    if lock:
        lock_owner = lock.get("owner")
        if lock_owner == "gpu" and owner == "cpu" and _lock_is_fresh(lock):
            logging.info("GPU learner active; idling CPU learner.")
            return False, owner
        if lock_owner == "cpu" and owner == "gpu":
            logging.info("GPU learner taking over from CPU learner.")
        if lock_owner == owner:
            _write_lock(path, owner)
            return True, owner
        if _lock_is_fresh(lock):
            if owner == "cpu":
                logging.info("Learner lock held by %s; idling CPU learner.", lock_owner)
                return False, owner
    _write_lock(path, owner)
    return True, owner


def _sleep_with_lock(
    cfg: dict,
    owner: str,
    total_seconds: int,
    refresh_seconds: int = 60,
    checkpoint_cb=None,
) -> None:
    path = _lock_path(cfg)
    remaining = total_seconds
    while remaining > 0:
        _write_lock(path, owner)
        interval = min(refresh_seconds, remaining)
        time.sleep(interval)
        if checkpoint_cb:
            checkpoint_cb()
        remaining -= interval


def _train_with_lock(loop_cfg: dict, resume: bool, cfg: dict, owner: str) -> None:
    path = _lock_path(cfg)
    errors: list[BaseException] = []

    def _run() -> None:
        try:
            train_from_config(loop_cfg, resume=resume)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    while thread.is_alive():
        _write_lock(path, owner)
        thread.join(timeout=60)
    if errors:
        raise errors[0]


def run_online_updates(cfg: dict) -> None:
    learning_cfg = cfg.get("learning", {})
    online_cfg = learning_cfg.get("online", {})
    if not online_cfg.get("enabled"):
        logging.info("Online updates disabled in config")
        return

    respect_ops_state = bool(online_cfg.get("respect_ops_state", True))
    interval_minutes = int(online_cfg.get("update_interval_minutes", 60))
    timesteps = int(online_cfg.get("timesteps", 1000))
    eval_split = float(online_cfg.get("eval_split", 0.1))
    resume = bool(learning_cfg.get("training", {}).get("resume", True))
    checkpoint_state = {"at": None}
    initial_delay = _initial_delay_seconds(cfg, interval_minutes)

    started_at = datetime.utcnow()
    while True:
        if initial_delay > 0:
            ok, owner = _ensure_exclusive(cfg)
            if ok:
                logging.info("Resuming learner after restart; sleeping %ds", initial_delay)
                def _checkpoint_tick() -> None:
                    checkpoint_state["at"] = _checkpoint_learner(
                        cfg,
                        checkpoint_state["at"],
                        status="sleeping",
                        owner=owner,
                        timesteps=timesteps,
                        eval_split=eval_split,
                    )

                _sleep_with_lock(cfg, owner, initial_delay, checkpoint_cb=_checkpoint_tick)
                initial_delay = 0
            else:
                time.sleep(60)
            continue
        ok, owner = _ensure_exclusive(cfg)
        if not ok:
            if should_restart(started_at):
                logging.info("Restart requested; exiting online updates.")
                break
            time.sleep(60)
            continue
        if respect_ops_state:
            ms_cfg = cfg.get("healthwatch", {}).get("market_shutdown", {}) or {}
            if ms_cfg.get("write_state", False):
                ops_state = load_ops_state(ms_cfg.get("state_path", "/data/system_state.json"))
                if ops_state_is_sleeping(ops_state):
                    logging.info("Ops state sleeping; pausing online updates.")
                    time.sleep(60)
                    continue
                if ops_state and not ops_state_is_running(ops_state):
                    logging.info("Ops state unknown; continuing with online updates.")
        if should_restart(started_at):
            logging.info("Restart requested; exiting online updates.")
            break
        loop_cfg = copy.deepcopy(cfg)
        loop_cfg.setdefault("learning", {}).setdefault("training", {})
        loop_cfg["learning"]["training"]["timesteps"] = timesteps
        loop_cfg["learning"]["training"]["eval_split"] = eval_split
        logging.info("Starting online update (%d timesteps)", timesteps)
        _train_with_lock(loop_cfg, resume=resume, cfg=cfg, owner=owner)
        checkpoint_state["at"] = _checkpoint_learner(
            cfg,
            checkpoint_state["at"],
            status="completed",
            owner=owner,
            timesteps=timesteps,
            eval_split=eval_split,
        )
        logging.info("Online update complete; sleeping %d minutes", interval_minutes)

        def _checkpoint_tick() -> None:
            checkpoint_state["at"] = _checkpoint_learner(
                cfg,
                checkpoint_state["at"],
                status="sleeping",
                owner=owner,
                timesteps=timesteps,
                eval_split=eval_split,
            )

        _sleep_with_lock(cfg, owner, interval_minutes * 60, checkpoint_cb=_checkpoint_tick)


def _checkpoint_learner(
    cfg: dict,
    last_saved_at: datetime | None,
    status: str,
    owner: str,
    timesteps: int,
    eval_split: float,
) -> datetime | None:
    payload = {
        "status": status,
        "owner": owner,
        "timesteps": timesteps,
        "eval_split": eval_split,
        "last_update_at": datetime.utcnow().isoformat(),
    }
    return maybe_save_checkpoint("learner", payload, cfg, last_saved_at)


def _initial_delay_seconds(cfg: dict, interval_minutes: int) -> int:
    data = load_checkpoint("learner", cfg)
    if not data:
        return 0
    payload = data.get("payload", {}) or {}
    last_update = payload.get("last_update_at")
    if not last_update:
        return 0
    try:
        last_dt = datetime.fromisoformat(last_update)
    except Exception:
        return 0
    next_dt = last_dt + timedelta(minutes=interval_minutes)
    delta = next_dt - datetime.utcnow()
    if delta.total_seconds() <= 0:
        return 0
    return int(delta.total_seconds())
