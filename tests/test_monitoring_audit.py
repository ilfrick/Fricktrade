import json
from datetime import datetime

from app.monitoring.audit import AuditLogger, ComplianceLogger


def test_audit_logger_writes_jsonl(tmp_path):
    logger = AuditLogger(str(tmp_path))
    payload = {"ts": "2025-01-01T00:00:00Z", "symbol": "AAPL", "decision": "hold"}
    logger.write(payload)
    date_str = datetime.utcnow().date().isoformat()
    log_path = tmp_path / f"{date_str}.jsonl"
    assert log_path.exists()
    raw = log_path.read_text(encoding="utf-8").strip().splitlines()[0]
    assert json.loads(raw)["symbol"] == "AAPL"


def test_compliance_logger_writes_jsonl_and_csv(tmp_path):
    logger = ComplianceLogger(str(tmp_path), formats=["jsonl", "csv"])
    payload = {"ts": "2025-01-01T00:00:00Z", "symbol": "AAPL", "decision": "hold", "stage": "signal"}
    logger.write(payload)
    date_str = datetime.utcnow().date().isoformat()
    jsonl_path = tmp_path / f"{date_str}.jsonl"
    csv_path = tmp_path / f"{date_str}.csv"
    assert jsonl_path.exists()
    assert csv_path.exists()
