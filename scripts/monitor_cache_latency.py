#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    print("PyYAML is required to run this monitor.", file=sys.stderr)
    sys.exit(2)


ERROR_RE = re.compile(r"(error|exception|traceback|failed)", re.IGNORECASE)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _parse_interval_seconds(label: str) -> int:
    if label.endswith("m"):
        return int(label[:-1]) * 60
    if label.endswith("h"):
        return int(label[:-1]) * 3600
    if label.endswith("d"):
        return int(label[:-1]) * 86400
    return 0


def _next_nyse_close(config: dict) -> datetime:
    market = config.get("market", {})
    extended = bool(market.get("extended_hours", {}).get("enabled", False))
    venues = market.get("venues", [])
    nyse = None
    for venue in venues:
        if venue.get("name", "").lower() == "nyse":
            nyse = venue
            break
    if not nyse:
        raise RuntimeError("NYSE venue not found in config.")

    tz = nyse.get("timezone", "America/New_York")
    hours = nyse.get("trading_hours", {})
    close_key = "extended_close" if extended else "close"
    close_str = hours.get(close_key) or hours.get("close")
    if not close_str:
        raise RuntimeError("NYSE close time not configured.")

    holidays = set(nyse.get("holidays", []))
    now = datetime.now(timezone.utc).astimezone(
        datetime.now().astimezone().tzinfo if tz is None else __import__("zoneinfo").ZoneInfo(tz)
    )
    close_hour, close_minute = [int(x) for x in close_str.split(":")]

    for i in range(0, 14):
        day = (now.date() + timedelta(days=i))
        if day.weekday() >= 5:
            continue
        if day.isoformat() in holidays:
            continue
        candidate = datetime(
            day.year, day.month, day.day, close_hour, close_minute, tzinfo=now.tzinfo
        )
        if candidate > now:
            return candidate
    raise RuntimeError("Unable to find next NYSE close within 2 weeks.")


def _dir_stats(path: Path) -> dict:
    files = [p for p in path.glob("*") if p.is_file()]
    if not files:
        return {"count": 0}
    mtimes = [p.stat().st_mtime for p in files]
    now = time.time()
    return {
        "count": len(files),
        "newest_age_s": round(now - max(mtimes), 2),
        "oldest_age_s": round(now - min(mtimes), 2),
    }


def _filtered_stats(path: Path) -> dict:
    files = [p for p in path.glob("*.json") if p.is_file()]
    if not files:
        return {"count": 0}
    updated = []
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if "updated_at" in data:
                updated.append(float(data["updated_at"]))
        except Exception:
            continue
    if not updated:
        return _dir_stats(path)
    now = time.time()
    return {
        "count": len(files),
        "newest_age_s": round(now - max(updated), 2),
        "oldest_age_s": round(now - min(updated), 2),
    }


def _log_header(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n[{ts}] {message}\n")


def _append_log(log_path: Path, payload: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def _capture_log_errors(since_iso: str, out_path: Path) -> int:
    try:
        output = subprocess.check_output(
            ["docker", "compose", "logs", "--since", since_iso, "--no-color"],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        output = exc.output
    lines = [line for line in output.splitlines() if ERROR_RE.search(line)]
    if lines:
        _log_header(out_path, "container errors")
        with out_path.open("a", encoding="utf-8") as fh:
            for line in lines:
                fh.write(line + "\n")
    return len(lines)


def _container_status(out_path: Path) -> None:
    try:
        output = subprocess.check_output(
            ["docker", "compose", "ps", "--status", "running"],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        output = exc.output
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(output + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Monitoring interval in seconds",
    )
    parser.add_argument(
        "--log",
        default="data/monitoring/cache_latency_monitor.log",
        help="Cache latency log path",
    )
    parser.add_argument(
        "--error-log",
        default="data/monitoring/container_failures.log",
        help="Container failure log path",
    )
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    close_at = _next_nyse_close(config)
    _log_header(Path(args.log), f"monitor_start close_at={close_at.isoformat()}")
    _log_header(Path(args.error_log), "monitor_start")

    last_log_time = datetime.now(timezone.utc)

    while datetime.now(timezone.utc) < close_at.astimezone(timezone.utc):
        base_dir = Path("data/market_cache")
        bars_dir = base_dir / "bars"
        filtered_dir = base_dir / "filtered"
        yf_dir = base_dir / "yf_cache"

        bar_stats = {}
        if bars_dir.exists():
            for interval_dir in sorted([p for p in bars_dir.iterdir() if p.is_dir()]):
                bar_stats[interval_dir.name] = _dir_stats(interval_dir)

        filtered_stats = {}
        if filtered_dir.exists():
            for interval_dir in sorted([p for p in filtered_dir.iterdir() if p.is_dir()]):
                filtered_stats[interval_dir.name] = _filtered_stats(interval_dir)

        yf_stats = _dir_stats(yf_dir) if yf_dir.exists() else {"count": 0}

        max_age_multiplier = (
            config.get("market_cache", {}).get("max_age_multiplier", 1) or 1
        )
        thresholds = {
            k: _parse_interval_seconds(k) * max_age_multiplier for k in bar_stats
        }

        payload = {
            "bars": bar_stats,
            "filtered": filtered_stats,
            "yf_cache": yf_stats,
            "thresholds_s": thresholds,
        }
        _append_log(Path(args.log), payload)

        since_iso = last_log_time.isoformat()
        last_log_time = datetime.now(timezone.utc)
        error_count = _capture_log_errors(since_iso, Path(args.error_log))
        if error_count:
            _log_header(Path(args.error_log), f"errors_detected count={error_count}")
        _container_status(Path(args.error_log))

        time.sleep(args.interval)

    _log_header(Path(args.log), "monitor_stop")
    _log_header(Path(args.error_log), "monitor_stop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
