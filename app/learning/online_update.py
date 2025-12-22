from __future__ import annotations

import copy
import logging
import time

from app.learning.train_rl import train_from_config


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

    while True:
        loop_cfg = copy.deepcopy(cfg)
        loop_cfg.setdefault("learning", {}).setdefault("training", {})
        loop_cfg["learning"]["training"]["timesteps"] = timesteps
        loop_cfg["learning"]["training"]["eval_split"] = eval_split
        logging.info("Starting online update (%d timesteps)", timesteps)
        train_from_config(loop_cfg, resume=resume)
        logging.info("Online update complete; sleeping %d minutes", interval_minutes)
        time.sleep(interval_minutes * 60)
