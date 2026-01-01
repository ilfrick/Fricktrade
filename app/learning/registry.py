from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def register_model(
    model_path: str,
    best_model_path: str | None,
    report: dict | None,
    registry_path: str,
    feature_stats: dict | None = None,
    metadata: dict | None = None,
    artifact_dir: str | None = None,
    artifact_prefix: str = "model",
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "ts": datetime.utcnow().isoformat(),
        "model_path": model_path,
        "model_sha256": _hash_file(model_path),
        "model_bytes": _file_size(model_path),
        "report": report or {},
        "feature_stats": feature_stats or {},
        "metadata": metadata or {},
    }
    if artifact_dir:
        artifact_path = _copy_artifact(model_path, artifact_dir, artifact_prefix)
        if artifact_path:
            record["artifact_path"] = artifact_path
    if best_model_path:
        record["best_model_path"] = best_model_path
        record["best_model_sha256"] = _hash_file(best_model_path)
        record["best_model_bytes"] = _file_size(best_model_path)
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = _load_registry(path)
    records.append(record)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    logging.info("Registered model metadata in %s", registry_path)
    return record


def load_latest_feature_stats(registry_path: str) -> dict[str, Any] | None:
    path = Path(registry_path)
    if not path.exists():
        return None
    records = _load_registry(path)
    for record in reversed(records):
        stats = record.get("feature_stats")
        if isinstance(stats, dict) and stats.get("mean") and stats.get("std"):
            return stats
    return None


def _load_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


def _hash_file(path_str: str) -> str | None:
    path = Path(path_str)
    if not path.exists():
        return None
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
    except Exception:
        return None
    return hasher.hexdigest()


def _file_size(path_str: str) -> int | None:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        return path.stat().st_size
    except Exception:
        return None


def _copy_artifact(model_path: str, artifact_dir: str, prefix: str) -> str | None:
    source = Path(model_path)
    if not source.exists():
        return None
    version_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    target_dir = Path(artifact_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{prefix}_{version_id}{source.suffix or '.zip'}"
    try:
        shutil.copy2(source, target)
    except Exception:
        return None
    return str(target)
