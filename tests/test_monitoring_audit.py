# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import json
from datetime import datetime

from app.monitoring.audit import AuditLogger, ComplianceLogger


def test_audit_logger_writes_jsonl(tmp_path):
    codes_path = tmp_path / "codes.json"
    codes_path.write_text(json.dumps(["ok"]), encoding="utf-8")
    logger = AuditLogger(
        str(tmp_path),
        enforce_reason_codes=True,
        reason_codes_path=str(codes_path),
        signing_enabled=True,
        signing_secret="secret",
    )
    payload = {"ts": "2025-01-01T00:00:00Z", "symbol": "AAPL", "decision": "hold", "reason": "missing"}
    logger.write(payload)
    date_str = datetime.utcnow().date().isoformat()
    log_path = tmp_path / f"{date_str}.jsonl"
    assert log_path.exists()
    raw = log_path.read_text(encoding="utf-8").strip().splitlines()[0]
    entry = json.loads(raw)
    assert entry["symbol"] == "AAPL"
    assert entry["reason"] == "unknown"
    assert entry["entry_hash"]
    assert entry["signature"]
    assert (tmp_path / ".audit_state.json").exists()


def test_compliance_logger_writes_jsonl_and_csv(tmp_path):
    logger = ComplianceLogger(str(tmp_path), formats=["jsonl", "csv"], signing_enabled=True, signing_secret="secret")
    payload = {"ts": "2025-01-01T00:00:00Z", "symbol": "AAPL", "decision": "hold", "stage": "signal"}
    logger.write(payload)
    date_str = datetime.utcnow().date().isoformat()
    jsonl_path = tmp_path / f"{date_str}.jsonl"
    csv_path = tmp_path / f"{date_str}.csv"
    assert jsonl_path.exists()
    assert csv_path.exists()
    digest = json.loads((tmp_path / f"{date_str}.jsonl.sha256").read_text(encoding="utf-8"))
    assert digest.get("sha256")
    assert digest.get("signature")
