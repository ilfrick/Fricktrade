# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import Any


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _floor_time(dt: datetime, interval_minutes: float) -> datetime:
    if interval_minutes <= 0:
        return dt.replace(second=0, microsecond=0)
    minutes = int(interval_minutes)
    if minutes <= 0:
        return dt.replace(second=0, microsecond=0)
    minute_bucket = dt.minute - (dt.minute % minutes)
    return dt.replace(minute=minute_bucket, second=0, microsecond=0)


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def _reward_paths(cfg: dict) -> tuple[Path, Path]:
    lr_cfg = cfg.get("learning", {}).get("live_rewards", {}) if isinstance(cfg, dict) else {}
    base_dir = lr_cfg.get("path", "/data/learning/live_rewards.jsonl")
    positions_path = lr_cfg.get("positions_path", "/data/learning/live_positions.json")
    return Path(base_dir), Path(positions_path)


def _reward_mode(cfg: dict) -> tuple[str, float]:
    learning_cfg = cfg.get("learning", {}) if isinstance(cfg, dict) else {}
    mode = str(learning_cfg.get("reward_pnl_mode", "abs") or "abs").lower()
    scale = float(learning_cfg.get("reward_pnl_scale", 1.0))
    return mode, scale


def _reward_value(realized_pnl: float, entry_price: float, qty: float, cfg: dict) -> float:
    mode, scale = _reward_mode(cfg)
    if mode == "pct":
        denom = entry_price * abs(qty)
        if denom <= 0:
            return 0.0
        return (realized_pnl / denom) * scale
    return realized_pnl * scale


@dataclass
class LivePosition:
    qty: float
    avg_entry: float


class LiveRewardTracker:
    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._rewards_path, self._positions_path = _reward_paths(cfg)
        self._positions: dict[str, LivePosition] = {}
        self._load_positions()

    def _key(self, broker: str | None, symbol: str) -> str:
        broker_name = str(broker or "")
        return f"{broker_name}:{symbol}"

    def _load_positions(self) -> None:
        if not self._positions_path.exists():
            return
        try:
            data = json.loads(self._positions_path.read_text())
        except Exception:
            return
        if not isinstance(data, dict):
            return
        for key, payload in data.items():
            if not isinstance(payload, dict):
                continue
            qty = float(payload.get("qty", 0.0) or 0.0)
            avg_entry = float(payload.get("avg_entry", 0.0) or 0.0)
            if qty == 0:
                continue
            self._positions[str(key)] = LivePosition(qty=qty, avg_entry=avg_entry)

    def _save_positions(self) -> None:
        payload = {key: {"qty": pos.qty, "avg_entry": pos.avg_entry} for key, pos in self._positions.items()}
        self._positions_path.parent.mkdir(parents=True, exist_ok=True)
        self._positions_path.write_text(json.dumps(payload))

    def sync_positions(self, portfolio: dict) -> None:
        brokers = portfolio.get("brokers") if isinstance(portfolio, dict) else None
        if not isinstance(brokers, dict):
            return
        updated = False
        for broker_name, broker_portfolio in brokers.items():
            positions = broker_portfolio.get("positions", {}) if isinstance(broker_portfolio, dict) else {}
            for symbol, pos in positions.items():
                qty = float(pos.get("qty", 0.0) or 0.0)
                avg_entry = float(pos.get("avg_entry", 0.0) or 0.0)
                if qty == 0 or avg_entry == 0:
                    continue
                key = self._key(broker_name, symbol)
                if key not in self._positions:
                    self._positions[key] = LivePosition(qty=qty, avg_entry=avg_entry)
                    updated = True
        if updated:
            self._save_positions()

    def on_order_response(self, response: dict) -> None:
        status = str(response.get("status", "")).lower()
        if status not in {"completed", "filled"}:
            return
        symbol = response.get("symbol")
        if not symbol:
            return
        broker = response.get("broker")
        side = str(response.get("side", "")).lower()
        if side not in {"buy", "sell"}:
            return
        qty = response.get("filled_qty") or response.get("qty")
        price = response.get("filled_avg_price") or response.get("price")
        try:
            qty = float(qty or 0.0)
            price = float(price or 0.0)
        except (TypeError, ValueError):
            return
        if qty <= 0 or price <= 0:
            return

        key = self._key(broker, str(symbol))
        pos = self._positions.get(key, LivePosition(qty=0.0, avg_entry=0.0))
        realized_pnl = 0.0
        entry_price = pos.avg_entry
        entry_qty = 0.0
        ts = _parse_ts(response.get("received_at")) or datetime.utcnow()

        if side == "buy":
            if pos.qty >= 0:
                new_qty = pos.qty + qty
                if new_qty > 0:
                    if pos.qty > 0:
                        entry_price = ((pos.avg_entry * pos.qty) + (price * qty)) / new_qty
                    else:
                        entry_price = price
                    pos = LivePosition(qty=new_qty, avg_entry=entry_price)
            else:
                cover_qty = min(abs(pos.qty), qty)
                realized_pnl = (pos.avg_entry - price) * cover_qty
                entry_price = pos.avg_entry
                entry_qty = cover_qty
                pos_qty = pos.qty + cover_qty
                pos = LivePosition(qty=pos_qty, avg_entry=pos.avg_entry if pos_qty < 0 else 0.0)
                remaining = qty - cover_qty
                if remaining > 0:
                    pos = LivePosition(qty=remaining, avg_entry=price)
        else:  # sell
            if pos.qty > 0:
                sell_qty = min(pos.qty, qty)
                realized_pnl = (price - pos.avg_entry) * sell_qty
                entry_price = pos.avg_entry
                entry_qty = sell_qty
                pos_qty = pos.qty - sell_qty
                pos = LivePosition(qty=pos_qty, avg_entry=pos.avg_entry if pos_qty > 0 else 0.0)
                remaining = qty - sell_qty
                if remaining > 0:
                    pos = LivePosition(qty=-remaining, avg_entry=price)
            else:
                new_qty = pos.qty - qty
                if new_qty < 0:
                    if pos.qty < 0:
                        entry_price = ((abs(pos.avg_entry * pos.qty)) + (price * qty)) / abs(new_qty)
                    else:
                        entry_price = price
                    pos = LivePosition(qty=new_qty, avg_entry=entry_price)

        self._positions[key] = pos
        self._save_positions()

        if realized_pnl == 0.0 or entry_qty == 0.0:
            return

        reward = _reward_value(realized_pnl, entry_price, entry_qty, self._cfg)
        record = {
            "ts": ts.isoformat(),
            "symbol": str(symbol),
            "broker": str(broker or ""),
            "side": side,
            "qty": entry_qty,
            "entry_price": entry_price,
            "exit_price": price,
            "realized_pnl": realized_pnl,
            "reward": reward,
        }
        self._append_reward(record)

    def _append_reward(self, record: dict) -> None:
        try:
            self._rewards_path.parent.mkdir(parents=True, exist_ok=True)
            with self._rewards_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except Exception as exc:
            logging.warning("Failed to append live reward: %s", exc)


def load_live_reward_overrides(cfg: dict, interval_minutes: float) -> dict[str, dict[str, float]]:
    lr_cfg = cfg.get("learning", {}).get("live_rewards", {}) if isinstance(cfg, dict) else {}
    if not lr_cfg.get("enabled", False):
        return {}
    rewards_path, _ = _reward_paths(cfg)
    if not rewards_path.exists():
        return {}
    max_days = int(lr_cfg.get("max_days", 7))
    cutoff = datetime.utcnow() - timedelta(days=max_days)
    overrides: dict[str, dict[str, float]] = {}
    try:
        lines = rewards_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return {}
    for line in lines:
        try:
            payload = json.loads(line)
        except Exception:
            continue
        ts = _parse_ts(payload.get("ts"))
        if ts is None or ts < cutoff:
            continue
        symbol = str(payload.get("symbol") or "")
        if not symbol:
            continue
        reward_val = payload.get("reward")
        try:
            reward_val = float(reward_val)
        except (TypeError, ValueError):
            continue
        bucket = _floor_time(ts, interval_minutes)
        key = bucket.isoformat()
        per_symbol = overrides.setdefault(symbol, {})
        per_symbol[key] = per_symbol.get(key, 0.0) + reward_val
    return overrides
