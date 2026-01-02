# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from typing import Any, Iterable
import os


def get_alpaca_account_cfg(cfg_or_brokers: dict, prefer_name: str | None = None) -> dict[str, Any]:
    brokers_cfg = _brokers_cfg(cfg_or_brokers)
    alpaca_cfg = brokers_cfg.get("alpaca", {}) or {}
    accounts = _enabled_accounts(alpaca_cfg.get("accounts"))
    if accounts:
        if prefer_name:
            for acct in accounts:
                if str(acct.get("name", "")).strip() == prefer_name:
                    return _merge_cfg(alpaca_cfg, acct)
        return _merge_cfg(alpaca_cfg, accounts[0])
    return dict(alpaca_cfg)


def get_ibkr_account_cfg(cfg_or_brokers: dict, prefer_name: str | None = None) -> dict[str, Any]:
    brokers_cfg = _brokers_cfg(cfg_or_brokers)
    ibkr_cfg = brokers_cfg.get("ibkr", {}) or {}
    accounts = _enabled_accounts(ibkr_cfg.get("accounts"))
    if accounts:
        if prefer_name:
            for acct in accounts:
                if str(acct.get("name", "")).strip() == prefer_name:
                    return _merge_cfg(ibkr_cfg, acct)
        return _merge_cfg(ibkr_cfg, accounts[0])
    return dict(ibkr_cfg)


def iter_alpaca_accounts(cfg: dict) -> Iterable[dict[str, Any]]:
    brokers_cfg = cfg.get("brokers", {}) if isinstance(cfg, dict) else {}
    alpaca_cfg = brokers_cfg.get("alpaca", {}) or {}
    if not alpaca_cfg.get("enabled", True):
        return []
    accounts = _enabled_accounts(alpaca_cfg.get("accounts"))
    if not accounts:
        accounts = _load_alpaca_env_accounts(alpaca_cfg)
    if not accounts:
        return [
            {
                "name": "alpaca",
                "api_key": alpaca_cfg.get("api_key", ""),
                "api_secret": alpaca_cfg.get("api_secret", ""),
                "base_url": alpaca_cfg.get("base_url", ""),
            }
        ]
    use_suffix = len(accounts) > 1
    results = []
    for idx, acct in enumerate(accounts):
        acct_name = str(acct.get("name") or f"account{idx + 1}")
        broker_name = f"alpaca:{acct_name}" if use_suffix else "alpaca"
        merged = _merge_cfg(alpaca_cfg, acct)
        merged["name"] = broker_name
        results.append(merged)
    return results


def iter_ibkr_accounts(cfg: dict) -> Iterable[dict[str, Any]]:
    brokers_cfg = cfg.get("brokers", {}) if isinstance(cfg, dict) else {}
    ibkr_cfg = brokers_cfg.get("ibkr", {}) or {}
    if not ibkr_cfg.get("enabled", False):
        return []
    accounts = _enabled_accounts(ibkr_cfg.get("accounts"))
    if not accounts:
        accounts = _load_ibkr_env_accounts(ibkr_cfg)
    if not accounts:
        return [
            {
                "name": "ibkr",
                "host": ibkr_cfg.get("host", "127.0.0.1"),
                "port": ibkr_cfg.get("port", 7497),
                "client_id": ibkr_cfg.get("client_id", 1),
                "account_id": ibkr_cfg.get("account_id", ""),
            }
        ]
    use_suffix = len(accounts) > 1
    results = []
    for idx, acct in enumerate(accounts):
        acct_name = str(acct.get("name") or f"account{idx + 1}")
        broker_name = f"ibkr:{acct_name}" if use_suffix else "ibkr"
        merged = _merge_cfg(ibkr_cfg, acct)
        merged["name"] = broker_name
        results.append(merged)
    return results


def _brokers_cfg(cfg_or_brokers: dict) -> dict[str, Any]:
    if "brokers" in cfg_or_brokers:
        return cfg_or_brokers.get("brokers", {}) or {}
    return cfg_or_brokers


def _merge_cfg(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update({k: v for k, v in override.items() if v is not None})
    return merged


def _enabled_accounts(accounts: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not accounts:
        return []
    enabled = []
    for acct in accounts:
        if not isinstance(acct, dict):
            continue
        if acct.get("enabled", True):
            enabled.append(acct)
    return enabled


def _load_alpaca_env_accounts(alpaca_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    keys = _split_env_list("ALPACA_API_KEYS")
    secrets = _split_env_list("ALPACA_API_SECRETS")
    names = _split_env_list("ALPACA_ACCOUNT_NAMES")
    base_urls = _split_env_list("ALPACA_BASE_URLS")
    accounts: list[dict[str, Any]] = []
    if keys and secrets:
        count = min(len(keys), len(secrets))
        for idx in range(count):
            accounts.append(
                {
                    "name": names[idx] if idx < len(names) else f"account{idx + 1}",
                    "api_key": keys[idx],
                    "api_secret": secrets[idx],
                    "base_url": base_urls[idx] if idx < len(base_urls) else alpaca_cfg.get("base_url", ""),
                    "enabled": True,
                }
            )
        return accounts
    for idx in range(1, 21):
        key = os.getenv(f"ALPACA_API_KEY_{idx}", "").strip()
        secret = os.getenv(f"ALPACA_API_SECRET_{idx}", "").strip()
        if not key and not secret:
            continue
        accounts.append(
            {
                "name": os.getenv(f"ALPACA_ACCOUNT_NAME_{idx}", f"account{idx}").strip(),
                "api_key": key,
                "api_secret": secret,
                "base_url": os.getenv(f"ALPACA_BASE_URL_{idx}", alpaca_cfg.get("base_url", "")).strip(),
                "enabled": True,
            }
        )
    return accounts


def _load_ibkr_env_accounts(ibkr_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    client_ids = _split_env_list("IBKR_CLIENT_IDS")
    names = _split_env_list("IBKR_ACCOUNT_NAMES")
    hosts = _split_env_list("IBKR_HOSTS")
    ports = _split_env_list("IBKR_PORTS")
    account_ids = _split_env_list("IBKR_ACCOUNT_IDS")
    accounts: list[dict[str, Any]] = []
    if client_ids:
        for idx, client_id in enumerate(client_ids):
            accounts.append(
                {
                    "name": names[idx] if idx < len(names) else f"account{idx + 1}",
                    "client_id": int(client_id),
                    "host": hosts[idx] if idx < len(hosts) else ibkr_cfg.get("host", "127.0.0.1"),
                    "port": int(ports[idx]) if idx < len(ports) else int(ibkr_cfg.get("port", 7497)),
                    "account_id": account_ids[idx] if idx < len(account_ids) else ibkr_cfg.get("account_id", ""),
                    "enabled": True,
                }
            )
        return accounts
    for idx in range(1, 21):
        client_id = os.getenv(f"IBKR_CLIENT_ID_{idx}", "").strip()
        if not client_id:
            continue
        accounts.append(
            {
                "name": os.getenv(f"IBKR_ACCOUNT_NAME_{idx}", f"account{idx}").strip(),
                "client_id": int(client_id),
                "host": os.getenv(f"IBKR_HOST_{idx}", ibkr_cfg.get("host", "127.0.0.1")).strip(),
                "port": int(os.getenv(f"IBKR_PORT_{idx}", str(ibkr_cfg.get("port", 7497))).strip()),
                "account_id": os.getenv(f"IBKR_ACCOUNT_ID_{idx}", ibkr_cfg.get("account_id", "")).strip(),
                "enabled": True,
            }
        )
    return accounts


def _split_env_list(key: str) -> list[str]:
    raw = os.getenv(key, "")
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]
