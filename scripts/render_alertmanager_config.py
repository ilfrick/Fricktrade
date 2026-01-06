#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import os
import re
from pathlib import Path


VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values.setdefault(key, value)
    return values


def _expand(template: str, env: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in env:
            return env[key]
        raise KeyError(f"Missing required env var: {key}")

    return VAR_PATTERN.sub(repl, template)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    template_path = root / "alertmanager" / "alertmanager.yml"
    output_path = root / "alertmanager" / "alertmanager.generated.yml"
    env_path = root / ".env"

    env = dict(os.environ)
    env.update(_load_env(env_path))
    rendered = _expand(template_path.read_text(), env)
    output_path.write_text(rendered)


if __name__ == "__main__":
    main()
