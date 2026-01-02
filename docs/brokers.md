<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Brokers

## Purpose
Provide a unified API for Alpaca and IBKR.

## Implementations
- Alpaca: `app/brokers/alpaca.py`.
- IBKR: `app/brokers/ibkr.py`.
- Interface: `app/brokers/base.py`.

## Selection
- Controlled by `brokers.ibkr.enabled`.
- Alpaca uses `TRADING_MODE=paper` to decide paper vs live.

## Multi-account
Use `brokers.<name>.accounts[]` to configure multiple accounts per broker. Each entry becomes its
own broker instance (e.g., `alpaca:primary`, `ibkr:account2`) and can be routed via
`execution.brokers.routing.*`. Set `execution.brokers.enabled: true` to activate routing. When only
one account is enabled, it remains named `alpaca` or `ibkr` for backward compatibility.
Use `execution.brokers.routing.mode: auto_split` to split symbols across accounts automatically.
If `accounts[]` is empty, the broker will auto-detect accounts from the environment:
- Alpaca: `ALPACA_API_KEYS`/`ALPACA_API_SECRETS` (comma lists) or `ALPACA_API_KEY_1` + `ALPACA_API_SECRET_1`, etc.
- IBKR: `IBKR_CLIENT_IDS` (comma list) or `IBKR_CLIENT_ID_1`, etc (optional `IBKR_ACCOUNT_ID_*`).
Invalid accounts are skipped and the Agent continues with the valid ones.

## Credentials
- Alpaca: `ALPACA_API_KEY`, `ALPACA_API_SECRET`
- IBKR: configured under `brokers.ibkr.*`

## Configuration
`config/config.yaml`:
- `brokers.alpaca.*`
- `brokers.ibkr.*`
- `brokers.<name>.fees.*`
