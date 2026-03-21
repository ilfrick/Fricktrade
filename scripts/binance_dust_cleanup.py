#!/usr/bin/env python3
"""One-time script: convert all Binance dust positions to BNB.

Run inside the trader container:
    docker exec fricktrade-trader-1 python3 /app/scripts/binance_dust_cleanup.py --config /app/config/config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from app.utils.config import load_config
    from app.data.binance_market_data import _build_binance_client_from_cfg

    cfg = load_config(args.config)
    client = _build_binance_client_from_cfg(cfg)
    if client is None:
        logger.error("Could not build Binance client")
        sys.exit(1)

    account = client.get_account()
    balances = account.get("balances", [])

    dust_assets = []
    stablecoins = {"USDT", "BUSD", "USDC", "BNB", "FDUSD", "USDS"}
    for bal in balances:
        asset = bal["asset"]
        free = float(bal.get("free", 0) or 0)
        locked = float(bal.get("locked", 0) or 0)
        total = free + locked
        if total > 0 and asset not in stablecoins:
            dust_assets.append((asset, total))
            logger.info("  %s: %.8f", asset, total)

    if not dust_assets:
        logger.info("No dust positions found.")
        return

    logger.info("Found %d dust assets to convert", len(dust_assets))

    if args.dry_run:
        logger.info("DRY RUN — no conversions performed")
        return

    # Binance transfer_dust converts small balances to BNB
    assets_to_convert = [a for a, _ in dust_assets]
    # Binance API limits to 10 assets per call
    for i in range(0, len(assets_to_convert), 10):
        batch = assets_to_convert[i : i + 10]
        try:
            result = client.transfer_dust(asset=batch)
            logger.info("Dust conversion result for %s: %s", batch, result)
        except Exception as exc:
            logger.warning("Dust conversion failed for %s: %s", batch, exc)
            # Try one by one
            for asset in batch:
                try:
                    result = client.transfer_dust(asset=[asset])
                    logger.info("  Converted %s: %s", asset, result)
                except Exception as inner_exc:
                    logger.warning("  Failed %s: %s", asset, inner_exc)


if __name__ == "__main__":
    main()
