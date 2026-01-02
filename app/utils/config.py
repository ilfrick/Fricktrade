# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import os
from pathlib import Path
import yaml


def _interpolate_env(value: str) -> str:
    if not isinstance(value, str):
        return value
    if "${" not in value:
        return value
    out = value
    while "${" in out:
        start = out.find("${")
        end = out.find("}", start)
        if end == -1:
            break
        key = out[start + 2 : end]
        out = out[:start] + os.getenv(key, "") + out[end + 1 :]
    return out


def _walk(obj):
    if isinstance(obj, dict):
        return {k: _walk(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(v) for v in obj]
    return _interpolate_env(obj)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _walk(data)
