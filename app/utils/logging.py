# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(level: str, file_path: str | None = None, max_bytes: int = 5_000_000, backup_count: int = 5) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if file_path:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
