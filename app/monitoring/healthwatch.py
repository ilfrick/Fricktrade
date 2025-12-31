from __future__ import annotations

import argparse
import http.server
import json
import logging
import socketserver
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None
try:
    import docker
except Exception:  # pragma: no cover
    docker = None

from app.utils.market import is_market_open, next_market_open


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


def _market_shutdown_cfg(cfg: dict) -> dict:
    return _healthwatch_cfg(cfg).get("market_shutdown", {}) or {}


def _kill_switch_cfg(cfg: dict) -> dict:
    return cfg.get("kill_switch", {}) or {}


def _kill_switch_armed(cfg: dict) -> bool:
    ks_cfg = _kill_switch_cfg(cfg)
    if not ks_cfg.get("armed", False):
        return False
    confirm = str(ks_cfg.get("confirm_code", "")).strip()
    required = str(ks_cfg.get("required_code", "")).strip()
    confirm_phrase = str(ks_cfg.get("confirm_phrase", "YES")).strip()
    if required:
        return confirm == required
    return confirm == confirm_phrase


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


def _project_containers(client, project: str) -> list:
    label = f"com.docker.compose.project={project}"
    return client.containers.list(all=True, filters={"label": label})


def _stop_services(client, project: str, services: set[str]) -> None:
    if not services:
        return
    for container in _project_containers(client, project):
        service = container.labels.get("com.docker.compose.service", "")
        if service in services:
            try:
                container.stop(timeout=30)
            except Exception as exc:
                logging.warning("Healthwatch stop failed for %s: %s", service, exc)


def _start_services(client, project: str, services: set[str]) -> None:
    if not services:
        return
    for container in _project_containers(client, project):
        service = container.labels.get("com.docker.compose.service", "")
        if service in services:
            try:
                container.start()
            except Exception as exc:
                logging.warning("Healthwatch start failed for %s: %s", service, exc)


def _run_market_scheduler(cfg: dict) -> None:
    ms_cfg = _market_shutdown_cfg(cfg)
    if not ms_cfg.get("enabled", False):
        return
    if docker is None:
        logging.warning("Healthwatch market shutdown requires docker SDK; skipping.")
        return
    project = str(ms_cfg.get("project_name", "autotrader"))
    interval = int(ms_cfg.get("check_interval_seconds", 60))
    start_before = int(ms_cfg.get("start_before_minutes", 15))
    heartbeat_minutes = int(ms_cfg.get("heartbeat_minutes", 15))
    keep = set(ms_cfg.get("keep_services", ["healthwatch", "autoheal"]))
    stop_list = ms_cfg.get("stop_services")
    client = docker.DockerClient(base_url="unix://var/run/docker.sock")

    last_state: str | None = None
    last_heartbeat: datetime | None = None
    while True:
        now = datetime.now(timezone.utc)
        market_open = is_market_open(cfg, now=now)
        next_open = next_market_open(cfg, now=now)
        should_run = market_open
        if not should_run and next_open is not None:
            delta = (next_open - now).total_seconds()
            should_run = delta <= start_before * 60
        force_sleep = bool(_kill_switch_cfg(cfg).get("force_sleep", False))
        if force_sleep:
            if _kill_switch_armed(cfg):
                should_run = False
            else:
                logging.warning("Kill switch force_sleep requested but interlock not armed.")
        containers = _project_containers(client, project)
        services = {c.labels.get("com.docker.compose.service", "") for c in containers}
        services.discard("")
        if stop_list:
            stop_services = set(stop_list)
        else:
            stop_services = services - keep
        if should_run:
            if last_state != "running":
                logging.info("Healthwatch market wake: starting services=%s", sorted(stop_services))
                _start_services(client, project, stop_services)
                last_state = "running"
        else:
            if last_state != "stopped":
                logging.info("Healthwatch market sleep: stopping services=%s", sorted(stop_services))
                _stop_services(client, project, stop_services)
                last_state = "stopped"
        if heartbeat_minutes > 0:
            if last_heartbeat is None or (now - last_heartbeat).total_seconds() >= heartbeat_minutes * 60:
                next_open_str = next_open.isoformat() if next_open else "unknown"
                logging.info(
                    "Healthwatch market scheduler heartbeat; state=%s next_open=%s force_sleep=%s",
                    last_state or "unknown",
                    next_open_str,
                    force_sleep,
                )
                last_heartbeat = now
        time.sleep(interval)


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
    scheduler = threading.Thread(target=_run_market_scheduler, args=(cfg,), daemon=True)
    scheduler.start()

    handler = _Handler
    handler.state = state
    with socketserver.TCPServer(("", port), handler) as httpd:
        logging.info("Healthwatch listening on :%d", port)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
