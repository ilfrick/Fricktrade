# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""Stablecoin flow tracker via Etherscan free API.

Tracks large USDT (ERC-20) transfers to known exchange hot wallets.
Large inflows = incoming buy pressure.  Large outflows = selling done.

Signal: stablecoin_inflow in [-1.0, 1.0] (market-wide, not per-symbol)
   +1.0 = large net inflow to exchanges (bullish — money arriving to buy)
   -1.0 = large net outflow from exchanges (bearish — money leaving)
    0.0 = balanced or no significant activity

Requires ETHERSCAN_API_KEY env var (free tier: 5 calls/sec).
Polls every 15 minutes.
"""
from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

# USDT ERC-20 contract address
USDT_CONTRACT = "0xdAC17F958D2ee523a2206206994597C13D831ec7"

# Known exchange hot wallets (Ethereum mainnet)
_EXCHANGE_WALLETS = {
    # Binance
    "0x28c6c06298d514db089934071355e5743bf21d60": "binance",
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549": "binance",
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d": "binance",
    # Coinbase
    "0x71660c4005ba85c37ccec55d0c4493e66fe775d3": "coinbase",
    "0x503828976d22510aad0201ac7ec88293211d23da": "coinbase",
    # Kraken
    "0x2910543af39aba0cd09dbb2d50200b3e800a63d2": "kraken",
    "0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0": "kraken",
    # OKX
    "0x6cc5f688a315f3dc28a7781717a9a798a59fda7b": "okx",
    # Bybit
    "0xf89d7b9c864f589bbf53a82105107622b35eaa40": "bybit",
}
_EXCHANGE_WALLET_SET = {addr.lower() for addr in _EXCHANGE_WALLETS}

# Minimum transfer size to track (in USDT, 6 decimals)
MIN_TRANSFER_USDT = 100_000  # $100k
# Volume threshold for "significant" flow
SIGNIFICANT_FLOW_USDT = 10_000_000  # $10M


class StablecoinFlowTracker:
    """Tracks large USDT transfers to/from exchanges via Etherscan."""

    def __init__(self, poll_interval: int = 900):
        self._api_key = os.environ.get("ETHERSCAN_API_KEY", "")
        self._poll_interval = poll_interval
        self._net_inflow: float = 0.0  # positive = inflow, negative = outflow (in USD)
        self._inflow_score: float = 0.0  # normalized [-1, 1]
        self._last_block: int = 0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if not _REQUESTS_AVAILABLE:
            return
        if not self._api_key:
            logger.info("ETHERSCAN_API_KEY not set; stablecoin flow tracking disabled")
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="stablecoin-flow-tracker")
        self._thread.start()
        logger.info("StablecoinFlowTracker started")

    def stop(self) -> None:
        self._running = False

    def get_inflow_score(self) -> float:
        """Net stablecoin inflow score in [-1.0, 1.0]. Market-wide signal."""
        with self._lock:
            return self._inflow_score

    def _run(self) -> None:
        while self._running:
            try:
                self._poll()
            except Exception as exc:
                logger.warning("StablecoinFlowTracker error: %s", exc)
            for _ in range(self._poll_interval):
                if not self._running:
                    break
                time.sleep(1)

    def _poll(self) -> None:
        # Get recent USDT transfers (last ~75 blocks ≈ 15 min)
        base_url = "https://api.etherscan.io/api"

        # Get latest block number
        resp = requests.get(base_url, params={
            "module": "proxy",
            "action": "eth_blockNumber",
            "apikey": self._api_key,
        }, timeout=15)
        if resp.status_code != 200:
            return
        latest_block = int(resp.json().get("result", "0x0"), 16)
        if latest_block == 0:
            return

        start_block = max(self._last_block + 1, latest_block - 75) if self._last_block > 0 else latest_block - 75

        # Get USDT token transfers in block range
        resp = requests.get(base_url, params={
            "module": "account",
            "action": "tokentx",
            "contractaddress": USDT_CONTRACT,
            "startblock": start_block,
            "endblock": latest_block,
            "sort": "desc",
            "apikey": self._api_key,
        }, timeout=30)
        if resp.status_code != 200:
            return
        data = resp.json()
        if data.get("status") != "1":
            # No transfers or error
            self._last_block = latest_block
            return

        transfers = data.get("result", [])
        net_inflow = 0.0

        for tx in transfers:
            value_raw = int(tx.get("value", "0"))
            value_usdt = value_raw / 1e6  # USDT has 6 decimals
            if value_usdt < MIN_TRANSFER_USDT:
                continue
            to_addr = tx.get("to", "").lower()
            from_addr = tx.get("from", "").lower()
            to_exchange = to_addr in _EXCHANGE_WALLET_SET
            from_exchange = from_addr in _EXCHANGE_WALLET_SET
            if to_exchange and not from_exchange:
                net_inflow += value_usdt  # inflow
            elif from_exchange and not to_exchange:
                net_inflow -= value_usdt  # outflow

        with self._lock:
            # Exponential decay: blend new reading with previous
            self._net_inflow = self._net_inflow * 0.5 + net_inflow * 0.5
            # Normalize to [-1, 1]
            if abs(self._net_inflow) < SIGNIFICANT_FLOW_USDT * 0.1:
                self._inflow_score = 0.0
            else:
                self._inflow_score = round(
                    max(-1.0, min(1.0, self._net_inflow / SIGNIFICANT_FLOW_USDT)),
                    4,
                )

        self._last_block = latest_block
