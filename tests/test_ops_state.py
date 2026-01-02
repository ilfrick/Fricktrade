# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import json

from app.utils.ops_state import load_ops_state, ops_state_is_running, ops_state_is_sleeping


def test_load_ops_state_missing(tmp_path):
    path = tmp_path / "missing.json"
    assert load_ops_state(str(path)) == {}


def test_ops_state_helpers(tmp_path):
    payload = {"state": "sleeping"}
    path = tmp_path / "state.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    state = load_ops_state(str(path))
    assert ops_state_is_sleeping(state) is True
    assert ops_state_is_running(state) is False
