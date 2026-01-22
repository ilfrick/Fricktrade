# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(level: str, file_path: str | None = None, max_bytes: int = 5_000_000, backup_count: int = 5) -> None:
    # Ensure the root logger is configured
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Remove existing handlers to prevent duplicate messages
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add stream handler
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # Add file handler if specified
    if file_path:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Explicitly set yfinance logger level
    yfinance_logger = logging.getLogger("yfinance")
    yfinance_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Explicitly set urllib3 logger level to INFO
    urllib3_logger = logging.getLogger("urllib3")
    urllib3_logger.setLevel(logging.INFO)
    
    # Explicitly set peewee logger level to INFO
    peewee_logger = logging.getLogger("peewee")
    peewee_logger.setLevel(logging.INFO)
