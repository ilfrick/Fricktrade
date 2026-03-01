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
                    return merge_cfg(alpaca_cfg, acct)
        return merge_cfg(alpaca_cfg, accounts[0])
    return dict(alpaca_cfg)


def get_ibkr_account_cfg(cfg_or_brokers: dict, prefer_name: str | None = None) -> dict[str, Any]:
    brokers_cfg = _brokers_cfg(cfg_or_brokers)
    ibkr_cfg = brokers_cfg.get("ibkr", {}) or {}
    accounts = _enabled_accounts(ibkr_cfg.get("accounts"))
    if accounts:
        if prefer_name:
            for acct in accounts:
                if str(acct.get("name", "")).strip() == prefer_name:
                    return merge_cfg(ibkr_cfg, acct)
        return merge_cfg(ibkr_cfg, accounts[0])
    return dict(ibkr_cfg)


def iter_alpaca_accounts(cfg: dict) -> Iterable[dict[str, Any]]:
    brokers_cfg = cfg.get("brokers", {}) if isinstance(cfg, dict) else {}
    alpaca_cfg = brokers_cfg.get("alpaca", {}) or {}
    if not alpaca_cfg.get("enabled", True):
        return []
    yaml_accounts = _enabled_accounts(alpaca_cfg.get("accounts"))
    env_accounts = _load_alpaca_env_accounts(alpaca_cfg)
    accounts = _merge_env_and_yaml_accounts(yaml_accounts, env_accounts)
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
        merged = merge_cfg(alpaca_cfg, acct)
        merged["name"] = broker_name
        results.append(merged)
    return results


def iter_binance_accounts(cfg: dict) -> Iterable[dict[str, Any]]:
    brokers_cfg = cfg.get("brokers", {}) if isinstance(cfg, dict) else {}
    binance_cfg = brokers_cfg.get("binance", {}) or {}
    if not binance_cfg.get("enabled", False):
        return []
    accounts = _enabled_accounts(binance_cfg.get("accounts"))
    if not accounts:
        accounts = _load_binance_env_accounts(binance_cfg)
    if not accounts:
        api_key = binance_cfg.get("api_key", "")
        api_secret = binance_cfg.get("api_secret", "")
        if not api_key and not api_secret:
            return []
        return [
            {
                "name": "binance",
                "api_key": api_key,
                "api_secret": api_secret,
                "testnet": bool(binance_cfg.get("testnet", False)),
                "futures": bool(binance_cfg.get("futures", False)),
                "base_url": str(binance_cfg.get("base_url", "") or ""),
            }
        ]
    use_suffix = len(accounts) > 1
    results = []
    for idx, acct in enumerate(accounts):
        acct_name = str(acct.get("name") or f"account{idx + 1}")
        broker_name = f"binance:{acct_name}" if use_suffix else "binance"
        merged = merge_cfg(binance_cfg, acct)
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
        merged = merge_cfg(ibkr_cfg, acct)
        merged["name"] = broker_name
        results.append(merged)
    return results


def _merge_env_and_yaml_accounts(
    yaml_accounts: list[dict[str, Any]],
    env_accounts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if env_accounts and yaml_accounts:
        yaml_by_name = {str(a.get("name", "")).strip(): a for a in yaml_accounts}
        merged: list[dict[str, Any]] = []
        for env_acct in env_accounts:
            name = str(env_acct.get("name", "")).strip()
            yaml_overrides = yaml_by_name.pop(name, None)
            if yaml_overrides:
                merged.append(merge_cfg(env_acct, yaml_overrides))
            else:
                merged.append(env_acct)
        for leftover in yaml_by_name.values():
            merged.append(leftover)
        return merged
    return env_accounts or yaml_accounts


def _brokers_cfg(cfg_or_brokers: dict) -> dict[str, Any]:
    if "brokers" in cfg_or_brokers:
        return cfg_or_brokers.get("brokers", {}) or {}
    return cfg_or_brokers


def merge_cfg(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        if v is None:
            continue
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = merge_cfg(merged[k], v)
        else:
            merged[k] = v
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


def _load_binance_env_accounts(binance_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    keys = _split_env_list("BINANCE_API_KEYS")
    secrets = _split_env_list("BINANCE_API_SECRETS")
    names = _split_env_list("BINANCE_ACCOUNT_NAMES")
    accounts: list[dict[str, Any]] = []
    if keys and secrets:
        count = min(len(keys), len(secrets))
        for idx in range(count):
            accounts.append(
                {
                    "name": names[idx] if idx < len(names) else f"account{idx + 1}",
                    "api_key": keys[idx],
                    "api_secret": secrets[idx],
                    "testnet": bool(binance_cfg.get("testnet", False)),
                    "enabled": True,
                }
            )
        return accounts
    for idx in range(1, 21):
        key = os.getenv(f"BINANCE_API_KEY_{idx}", "").strip()
        secret = os.getenv(f"BINANCE_API_SECRET_{idx}", "").strip()
        if not key and not secret:
            continue
        accounts.append(
            {
                "name": os.getenv(f"BINANCE_ACCOUNT_NAME_{idx}", f"account{idx}").strip(),
                "api_key": key,
                "api_secret": secret,
                "testnet": bool(binance_cfg.get("testnet", False)),
                "enabled": True,
            }
        )
    return accounts


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


def is_crypto_symbol(symbol: str) -> bool:
    """Return True if symbol is a crypto pair (contains '/')."""
    return "/" in symbol


def get_asset_class(symbol: str) -> str:
    """Return 'crypto' or 'equities' based on symbol format."""
    return "crypto" if is_crypto_symbol(symbol) else "equities"


def get_venue_for_symbol(symbol: str) -> str:
    """Return the trading venue name for a symbol."""
    return "Crypto" if is_crypto_symbol(symbol) else "NYSE"


def _split_env_list(key: str) -> list[str]:
    raw = os.getenv(key, "")
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]
