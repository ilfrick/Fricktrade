# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


class AuditLogger:
    def __init__(
        self,
        output_dir: str,
        retention_days: int = 0,
        enforce_reason_codes: bool = False,
        reason_codes_path: str | None = None,
        signing_secret: str | None = None,
        signing_enabled: bool = False,
    ):
        self._output_dir = output_dir
        self._retention_days = int(retention_days)
        self._last_prune_date: str | None = None
        self._enforce_reason_codes = bool(enforce_reason_codes)
        self._reason_codes = _load_reason_codes(reason_codes_path)
        self._signing_enabled = bool(signing_enabled)
        self._signing_secret = signing_secret or ""
        self._state_path = Path(output_dir) / ".audit_state.json"
        self._hash_state = _load_hash_state(self._state_path)

    def write(self, payload: dict) -> None:
        try:
            date_str = datetime.now(timezone.utc).date().isoformat()
            self._maybe_prune(date_str)
            path = Path(self._output_dir) / f"{date_str}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = _normalize_reason(payload, self._enforce_reason_codes, self._reason_codes)
            if self._signing_enabled and self._signing_secret:
                payload = self._sign_payload(date_str, payload)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        except Exception as exc:
            logging.warning("Audit log write failed: %s", exc)

    def _sign_payload(self, date_str: str, payload: dict) -> dict:
        prev_hash = self._hash_state.get(date_str, "")
        base = _canonical_json(payload)
        entry_hash = hashlib.sha256((prev_hash + base).encode("utf-8")).hexdigest()
        signature = hmac.new(self._signing_secret.encode("utf-8"), entry_hash.encode("utf-8"), hashlib.sha256).hexdigest()
        payload = dict(payload)
        payload["prev_hash"] = prev_hash or None
        payload["entry_hash"] = entry_hash
        payload["signature"] = signature
        payload["signature_alg"] = "hmac-sha256"
        self._hash_state[date_str] = entry_hash
        _save_hash_state(self._state_path, self._hash_state)
        return payload

    def _maybe_prune(self, date_str: str) -> None:
        if self._retention_days <= 0:
            return
        if self._last_prune_date == date_str:
            return
        _prune_old_files(Path(self._output_dir), self._retention_days)
        self._last_prune_date = date_str


class ComplianceLogger:
    def __init__(
        self,
        output_dir: str,
        formats: Iterable[str] | None = None,
        retention_days: int = 0,
        enforce_reason_codes: bool = False,
        reason_codes_path: str | None = None,
        signing_secret: str | None = None,
        signing_enabled: bool = False,
    ):
        self._output_dir = output_dir
        self._formats = [str(fmt).lower() for fmt in (formats or ["jsonl"])]
        self._retention_days = int(retention_days)
        self._last_prune_date: str | None = None
        self._enforce_reason_codes = bool(enforce_reason_codes)
        self._reason_codes = _load_reason_codes(reason_codes_path)
        self._signing_secret = signing_secret or ""
        self._signing_enabled = bool(signing_enabled)

    def write(self, payload: dict) -> None:
        date_str = datetime.now(timezone.utc).date().isoformat()
        self._maybe_prune(date_str)
        payload = _normalize_reason(payload, self._enforce_reason_codes, self._reason_codes)
        for fmt in self._formats:
            if fmt == "jsonl":
                self._write_jsonl(payload, date_str)
            elif fmt == "csv":
                self._write_csv(payload, date_str)

    def _write_jsonl(self, payload: dict, date_str: str) -> None:
        try:
            path = Path(self._output_dir) / f"{date_str}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
            self._write_digest(path)
        except Exception as exc:
            logging.warning("Compliance JSONL write failed: %s", exc)

    def _write_csv(self, payload: dict, date_str: str) -> None:
        fields = [
            "ts",
            "symbol",
            "decision",
            "reason",
            "stage",
            "action",
            "action_strategy",
            "broker",
            "qty",
            "order_type",
            "limit_price",
            "algo",
            "venue",
            "decision_latency_seconds",
            "order_latency_seconds",
        ]
        row = {key: payload.get(key) for key in fields}
        try:
            path = Path(self._output_dir) / f"{date_str}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not path.exists()
            with path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            self._write_digest(path)
        except Exception as exc:
            logging.warning("Compliance CSV write failed: %s", exc)

    def _write_digest(self, path: Path) -> None:
        digest = _file_sha256(path)
        if not digest:
            return
        sig = None
        if self._signing_enabled and self._signing_secret:
            sig = hmac.new(self._signing_secret.encode("utf-8"), digest.encode("utf-8"), hashlib.sha256).hexdigest()
        digest_path = path.with_suffix(path.suffix + ".sha256")
        payload = {"sha256": digest, "signature": sig, "signature_alg": "hmac-sha256" if sig else None}
        digest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _maybe_prune(self, date_str: str) -> None:
        if self._retention_days <= 0:
            return
        if self._last_prune_date == date_str:
            return
        _prune_old_files(Path(self._output_dir), self._retention_days)
        self._last_prune_date = date_str


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _file_sha256(path: Path) -> str | None:
    try:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def _load_reason_codes(path: str | None) -> set[str]:
    if not path:
        return set()
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return set()
    if isinstance(data, list):
        return {str(item) for item in data}
    return set()


def _normalize_reason(payload: dict, enforce: bool, reason_codes: set[str]) -> dict:
    if not enforce or not reason_codes:
        return payload
    reason = payload.get("reason")
    if reason in reason_codes:
        return payload
    updated = dict(payload)
    updated["reason_raw"] = reason
    updated["reason"] = "unknown"
    return updated


def _load_hash_state(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    return {}


def _save_hash_state(path: Path, state: dict[str, str]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        return


def _prune_old_files(output_dir: Path, retention_days: int) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    for path in output_dir.glob("*"):
        if not path.is_file():
            continue
        try:
            if datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) < cutoff:
                path.unlink()
        except Exception:
            continue
