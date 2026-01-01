import json

import numpy as np
import pandas as pd

from app.learning.drift import DriftMonitor, compute_feature_stats
from app.learning.features import observation_size
from app.learning.registry import load_active_model, load_latest_feature_stats, register_model, set_active_model


def test_compute_feature_stats_matches_observation_size():
    df = pd.DataFrame(
        {
            "Close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "Volume": [10.0, 12.0, 11.0, 13.0, 12.0, 14.0],
        }
    )
    feature_cfg = {"include_returns": True, "include_signal_features": False}
    stats = compute_feature_stats([df], window_size=3, feature_config=feature_cfg, max_samples=10, stride=1)
    assert stats["count"] > 0
    assert len(stats["mean"]) == observation_size(3, feature_cfg)
    assert len(stats["std"]) == observation_size(3, feature_cfg)


def test_drift_monitor_flags_feature_drift():
    baseline = {"mean": [0.0, 0.0], "std": [1.0, 1.0]}
    monitor = DriftMonitor(
        baseline_stats=baseline,
        window=3,
        feature_zscore_threshold=1.0,
        max_drift_feature_pct=0.5,
        pnl_window=3,
        max_pnl_drop_pct=2.0,
    )
    for vec in (np.array([5.0, 0.0]), np.array([5.0, 0.0]), np.array([5.0, 0.0])):
        monitor.update_features(vec)
    reasons = monitor.check_drift()
    assert "feature_drift" in reasons


def test_drift_monitor_flags_pnl_drift():
    monitor = DriftMonitor(baseline_stats=None, window=3, pnl_window=3, max_pnl_drop_pct=1.0)
    for pnl in (-2.0, -1.5, -1.2):
        monitor.update_pnl(pnl)
    reasons = monitor.check_drift()
    assert "pnl_drift" in reasons


def test_register_model_writes_registry(tmp_path):
    model_path = tmp_path / "model.zip"
    model_path.write_text("model-bytes", encoding="utf-8")
    registry_path = tmp_path / "registry.json"
    artifact_dir = tmp_path / "artifacts"
    stats = {"mean": [0.0], "std": [1.0], "count": 1}
    record = register_model(
        model_path=str(model_path),
        best_model_path=None,
        report={"average": {"return_pct": 1.0}},
        registry_path=str(registry_path),
        feature_stats=stats,
        metadata={"window_size": 10},
        artifact_dir=str(artifact_dir),
        artifact_prefix="ppo_policy",
    )
    assert record["model_path"] == str(model_path)
    assert record["artifact_path"].startswith(str(artifact_dir))
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    assert len(raw) == 1
    assert load_latest_feature_stats(str(registry_path)) == stats


def test_set_active_model_writes_pointer(tmp_path):
    registry_path = tmp_path / "registry.json"
    record = {"model_path": "/app/models/model.zip", "model_sha256": "abc123", "ts": "2026-01-01T00:00:00Z"}
    active_path = tmp_path / "active.json"
    set_active_model(str(active_path), record, reason="latest")
    active = load_active_model(str(active_path))
    assert active["model_path"] == record["model_path"]
    assert active["reason"] == "latest"
