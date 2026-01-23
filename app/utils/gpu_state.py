# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import json
import logging
import os
from pathlib import Path
from datetime import datetime

GPU_STATE_FILE = Path(os.environ.get("FRICKTRADE_GPU_STATE_FILE", "/data/gpu_state.json"))

def _load_gpu_state() -> dict:
    if not GPU_STATE_FILE.exists():
        return {"gpu_disabled": False}
    try:
        return json.loads(GPU_STATE_FILE.read_text())
    except Exception as exc:
        logging.warning("Failed to load GPU state from %s: %s", GPU_STATE_FILE, exc)
        return {"gpu_disabled": False}

def _save_gpu_state(state: dict) -> None:
    try:
        GPU_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        GPU_STATE_FILE.write_text(json.dumps(state))
    except Exception as exc:
        logging.error("Failed to save GPU state to %s: %s", GPU_STATE_FILE, exc)

def is_gpu_disabled() -> bool:
    return _load_gpu_state().get("gpu_disabled", False)

def disable_gpu_until_restart() -> None:
    logging.warning("Disabling GPU until next restart due to error.")
    _save_gpu_state({"gpu_disabled": True, "disabled_at": datetime.utcnow().isoformat()})

def enable_gpu() -> None:
    logging.info("Enabling GPU.")
    _save_gpu_state({"gpu_disabled": False})
