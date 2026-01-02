# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import argparse
import http.server
import json
import logging
import socketserver
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


DEFAULT_TARGETS = {
    "trader": "http://trader:8001/metrics",
    "api": "http://api:8000/health",
    "prometheus": "http://prometheus:9090/-/healthy",
    "grafana": "http://grafana:3000/api/health",
    "alertmanager": "http://alertmanager:9093/-/healthy",
}


class HealthState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status: dict[str, int] = {}
        self.checked_at: dict[str, float] = {}

    def update(self, service: str, up: int) -> None:
        with self._lock:
            self.status[service] = up
            self.checked_at[service] = time.time()

    def snapshot(self) -> tuple[dict[str, int], dict[str, float]]:
        with self._lock:
            return dict(self.status), dict(self.checked_at)


def _read_cfg(path: Path) -> dict[str, Any]:
    if not path.exists() or yaml is None:
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def _healthwatch_cfg(cfg: dict) -> dict:
    return cfg.get("healthwatch", {}) or {}


def _check_url(url: str, timeout_seconds: int) -> int:
    try:
        with urlopen(url, timeout=timeout_seconds) as resp:
            return 1 if 200 <= resp.status < 300 else 0
    except Exception:
        return 0


def _run_checks(state: HealthState, targets: dict[str, str], interval: int, timeout: int) -> None:
    while True:
        for name, url in targets.items():
            up = _check_url(url, timeout)
            state.update(name, up)
        time.sleep(interval)


def _render_metrics(state: HealthState) -> bytes:
    status, checked = state.snapshot()
    lines = [
        "# HELP healthwatch_service_up Service health (1=up, 0=down)",
        "# TYPE healthwatch_service_up gauge",
    ]
    for service, up in sorted(status.items()):
        lines.append(f'healthwatch_service_up{{service=\"{service}\"}} {up}')
    lines.append("# HELP healthwatch_last_check_timestamp_seconds Last check time for service")
    lines.append("# TYPE healthwatch_last_check_timestamp_seconds gauge")
    for service, ts in sorted(checked.items()):
        lines.append(f'healthwatch_last_check_timestamp_seconds{{service=\"{service}\"}} {ts}')
    lines.append("# HELP healthwatch_up Healthwatch process status")
    lines.append("# TYPE healthwatch_up gauge")
    lines.append("healthwatch_up 1")
    return ("\n".join(lines) + "\n").encode("utf-8")


class _Handler(http.server.BaseHTTPRequestHandler):
    state: HealthState

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in ("/", "/metrics"):
            self.send_response(404)
            self.end_headers()
            return
        data = _render_metrics(self.state)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/app/config/config.yaml")
    args = parser.parse_args()

    cfg = _read_cfg(Path(args.config))
    hw_cfg = _healthwatch_cfg(cfg)
    interval = int(hw_cfg.get("interval_seconds", 30))
    timeout = int(hw_cfg.get("timeout_seconds", 5))
    port = int(hw_cfg.get("port", 9105))
    targets = hw_cfg.get("targets", DEFAULT_TARGETS) or DEFAULT_TARGETS

    logging.info("Healthwatch starting: interval=%ss targets=%d", interval, len(targets))
    state = HealthState()
    for name in targets:
        state.update(name, 0)

    thread = threading.Thread(target=_run_checks, args=(state, targets, interval, timeout), daemon=True)
    thread.start()

    handler = _Handler
    handler.state = state
    with socketserver.TCPServer(("", port), handler) as httpd:
        logging.info("Healthwatch listening on :%d", port)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
