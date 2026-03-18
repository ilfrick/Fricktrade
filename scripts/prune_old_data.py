# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""
Periodic data retention pruner.

Deletes data older than `retention_days` (default 7) from the directories
that grow unboundedly.  Runs once at startup then every `interval_hours`
(default 24).  Checkpoints are self-managed by checkpoint.py and are
intentionally excluded here.

Directories / file patterns pruned
-----------------------------------
  data/monitoring/YYYY-MM-DD/          full day directories
  data/reports/decision_trace/         YYYY-MM-DD.jsonl files
  data/reports/daily_top_movers/       YYYY-MM-DD/ directories
  data/reports/health/                 health_YYYY-MM-DD_HH.{json,txt} files
  data/                                root-level dated CSVs (SYMBOL_DATE_interval.csv)
                                       identified by mtime rather than name parsing

Directories intentionally NOT pruned
--------------------------------------
  data/checkpoints/    — managed by checkpoint.py retention settings
  data/market_cache/   — managed by cache TTL (Redis expiry + staleness check)
  data/training/       — model training datasets; small and valuable
  data/models/         — trained model files
  data/backtest_cache/ — reusable across sessions
  data/learning/       — RL learning state
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _is_older_than(path: Path, cutoff: datetime) -> bool:
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return mtime < cutoff
    except OSError:
        return False


def _remove_dir(path: Path) -> int:
    """Remove directory tree; return bytes freed."""
    try:
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        shutil.rmtree(path)
        return size
    except Exception as exc:
        logging.warning("Failed to remove dir %s: %s", path, exc)
        return 0


def _remove_file(path: Path) -> int:
    """Remove file; return bytes freed."""
    try:
        size = path.stat().st_size
        path.unlink()
        return size
    except Exception as exc:
        logging.warning("Failed to remove file %s: %s", path, exc)
        return 0


def prune(data_dir: Path, retention_days: int) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    total_bytes = 0
    total_items = 0

    logging.info(
        "Pruning data older than %d days (cutoff %s UTC) in %s",
        retention_days,
        cutoff.strftime("%Y-%m-%d"),
        data_dir,
    )

    # 1. monitoring/YYYY-MM-DD/ directories
    monitoring_dir = data_dir / "monitoring"
    if monitoring_dir.is_dir():
        for child in sorted(monitoring_dir.iterdir()):
            if child.is_dir() and _is_older_than(child, cutoff):
                freed = _remove_dir(child)
                logging.info("Removed monitoring/%s (%.1f MB)", child.name, freed / 1e6)
                total_bytes += freed
                total_items += 1

    # 2. reports/decision_trace/YYYY-MM-DD.jsonl
    trace_dir = data_dir / "reports" / "decision_trace"
    if trace_dir.is_dir():
        for f in sorted(trace_dir.glob("*.jsonl")):
            if _is_older_than(f, cutoff):
                freed = _remove_file(f)
                logging.info("Removed decision_trace/%s (%.1f MB)", f.name, freed / 1e6)
                total_bytes += freed
                total_items += 1

    # 3. reports/daily_top_movers/YYYY-MM-DD/
    top_movers_dir = data_dir / "reports" / "daily_top_movers"
    if top_movers_dir.is_dir():
        for child in sorted(top_movers_dir.iterdir()):
            if child.is_dir() and _is_older_than(child, cutoff):
                freed = _remove_dir(child)
                logging.info(
                    "Removed daily_top_movers/%s (%.1f MB)", child.name, freed / 1e6
                )
                total_bytes += freed
                total_items += 1

    # 4. reports/health/health_YYYY-MM-DD_HH.{json,txt}
    health_dir = data_dir / "reports" / "health"
    if health_dir.is_dir():
        for f in sorted(health_dir.iterdir()):
            if f.is_file() and _is_older_than(f, cutoff):
                freed = _remove_file(f)
                logging.info("Removed health/%s (%.1f KB)", f.name, freed / 1e3)
                total_bytes += freed
                total_items += 1

    # 5. Root-level dated CSVs: SYMBOL_YYYY-MM-DD_interval.csv
    #    Identified by mtime; undated files (SYMBOL_5m.csv, SYMBOL_1m.csv) are
    #    permanent reference data — they have recent mtime and won't match.
    for f in data_dir.glob("*_*m.csv"):
        # Only delete files whose name contains a date segment (8-digit block)
        # AND whose mtime is past the cutoff.  This protects SYMBOL_5m.csv whose
        # mtime is always fresh (rewritten on each collect run).
        parts = f.stem.split("_")
        has_date = any(len(p) == 10 and p[4] == "-" and p[7] == "-" for p in parts)
        if has_date and _is_older_than(f, cutoff):
            freed = _remove_file(f)
            total_bytes += freed
            total_items += 1

    if total_items == 0:
        logging.info("Nothing to prune — all data within retention window.")
    else:
        logging.info(
            "Pruning complete: removed %d items, freed %.2f GB",
            total_items,
            total_bytes / 1e9,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Periodic data retention pruner")
    parser.add_argument("--data-dir", default="/data", help="Base data directory")
    parser.add_argument(
        "--retention-days", type=int, default=7, help="Days of data to keep"
    )
    parser.add_argument(
        "--interval-hours", type=float, default=24.0, help="Hours between runs"
    )
    parser.add_argument(
        "--once", action="store_true", help="Run once and exit (for testing)"
    )
    args = parser.parse_args()

    _setup_logging()
    data_dir = Path(args.data_dir)

    if not data_dir.is_dir():
        logging.error("Data directory not found: %s", data_dir)
        return

    while True:
        try:
            prune(data_dir, args.retention_days)
        except Exception as exc:
            logging.error("Pruner run failed: %s", exc, exc_info=True)

        if args.once:
            break

        sleep_seconds = args.interval_hours * 3600
        next_run = datetime.now(timezone.utc) + timedelta(seconds=sleep_seconds)
        logging.info(
            "Next prune run at %s UTC", next_run.strftime("%Y-%m-%d %H:%M")
        )
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
