#!/usr/bin/env python3
"""Generate per-account Grafana dashboard JSON files from a template.

Reads broker account names from .env, applies them to
``grafana/provisioning/dashboards/_template_account.json``, and writes one
dashboard file per account.  Orphan files from removed accounts are deleted.

Usage::

    python3 scripts/generate_grafana_dashboards.py \
        --env-file .env \
        --output-dir grafana/provisioning/dashboards
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

GENERATED_PREFIX = "account_"
GENERATED_PATTERN = re.compile(r"^account_.*\.json$")


def _parse_env(path: Path) -> dict[str, str]:
    """Minimal .env parser (key=value, ignores comments and blanks)."""
    env: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def _split_csv(value: str) -> list[str]:
    """Split a comma-separated env value, stripping whitespace."""
    return [v.strip() for v in value.split(",") if v.strip()]


def _sanitize_uid(name: str) -> str:
    """Lowercase, replace non-alphanumeric with hyphens."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _collect_accounts(env: dict[str, str]) -> list[tuple[str, str, str]]:
    """Return list of (broker_label, account_name, uid_suffix) tuples."""
    accounts: list[tuple[str, str, str]] = []

    # Alpaca accounts
    alpaca_names = _split_csv(env.get("ALPACA_ACCOUNT_NAMES", ""))
    for name in alpaca_names:
        broker_label = f"alpaca:{name}"
        uid_suffix = f"alpaca-{_sanitize_uid(name)}"
        accounts.append((broker_label, name, uid_suffix))

    # IBKR accounts (future)
    ibkr_names = _split_csv(env.get("IBKR_ACCOUNT_NAMES", ""))
    for name in ibkr_names:
        broker_label = f"ibkr:{name}"
        uid_suffix = f"ibkr-{_sanitize_uid(name)}"
        accounts.append((broker_label, name, uid_suffix))

    return accounts


def generate(env_file: Path, output_dir: Path) -> None:
    template_path = output_dir / "_template_account.json.template"
    if not template_path.exists():
        print(f"ERROR: template not found: {template_path}", file=sys.stderr)
        sys.exit(1)

    template = template_path.read_text()
    env = _parse_env(env_file)
    accounts = _collect_accounts(env)

    if not accounts:
        print("WARNING: no broker accounts found in .env", file=sys.stderr)

    # Generate dashboards
    expected_files: set[str] = set()
    for broker_label, account_name, uid_suffix in accounts:
        filename = f"{GENERATED_PREFIX}{uid_suffix.replace('-', '_')}.json"
        expected_files.add(filename)

        content = template
        content = content.replace("{{BROKER_LABEL}}", broker_label)
        content = content.replace("{{ACCOUNT_NAME}}", account_name)
        content = content.replace("{{UID_SUFFIX}}", uid_suffix)

        out_path = output_dir / filename
        out_path.write_text(content)
        print(f"  generated: {filename}  ({broker_label})")

    # Remove orphan generated files
    for existing in sorted(output_dir.iterdir()):
        if GENERATED_PATTERN.match(existing.name) and existing.name not in expected_files:
            existing.unlink()
            print(f"  removed orphan: {existing.name}")

    print(f"Done: {len(accounts)} dashboard(s) generated.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-account Grafana dashboards")
    parser.add_argument("--env-file", type=Path, required=True, help="Path to .env file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Dashboard output directory")
    args = parser.parse_args()

    if not args.env_file.exists():
        print(f"ERROR: .env file not found: {args.env_file}", file=sys.stderr)
        sys.exit(1)
    if not args.output_dir.is_dir():
        print(f"ERROR: output dir not found: {args.output_dir}", file=sys.stderr)
        sys.exit(1)

    generate(args.env_file, args.output_dir)


if __name__ == "__main__":
    main()
