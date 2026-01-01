from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable


class AuditLogger:
    def __init__(self, output_dir: str):
        self._output_dir = output_dir

    def write(self, payload: dict) -> None:
        try:
            date_str = datetime.utcnow().date().isoformat()
            path = Path(self._output_dir) / f"{date_str}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        except Exception as exc:
            logging.warning("Audit log write failed: %s", exc)


class ComplianceLogger:
    def __init__(self, output_dir: str, formats: Iterable[str] | None = None):
        self._output_dir = output_dir
        self._formats = [str(fmt).lower() for fmt in (formats or ["jsonl"])]

    def write(self, payload: dict) -> None:
        for fmt in self._formats:
            if fmt == "jsonl":
                self._write_jsonl(payload)
            elif fmt == "csv":
                self._write_csv(payload)

    def _write_jsonl(self, payload: dict) -> None:
        try:
            date_str = datetime.utcnow().date().isoformat()
            path = Path(self._output_dir) / f"{date_str}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        except Exception as exc:
            logging.warning("Compliance JSONL write failed: %s", exc)

    def _write_csv(self, payload: dict) -> None:
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
            date_str = datetime.utcnow().date().isoformat()
            path = Path(self._output_dir) / f"{date_str}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not path.exists()
            with path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as exc:
            logging.warning("Compliance CSV write failed: %s", exc)
