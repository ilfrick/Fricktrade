# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.monitoring.metrics import (
    STRATEGY_TRADES_REALIZED,
    STRATEGY_WIN_RATE,
    STRATEGY_AVG_PNL_PCT,
    STRATEGY_DRAWDOWN_PCT,
    STRATEGY_DISABLED,
    STRATEGY_SHARPE_RATIO,
    STRATEGY_PROFIT_FACTOR,
)


class PerformanceTracker:
    """Tracks per-strategy and per-symbol trade performance, manages the kill switch."""

    def __init__(self, cfg: dict, strategy_names: list[str]):
        perf_cfg = cfg.get("strategy", {}).get("performance", {})
        self._enabled = bool(perf_cfg.get("enabled", False))
        self._window_days = int(perf_cfg.get("window_days", 30))
        self._min_trades = int(perf_cfg.get("min_trades", 20))
        self._min_win_rate = float(perf_cfg.get("min_win_rate", 0.4))
        self._max_drawdown = float(perf_cfg.get("max_drawdown_pct", 12.0))
        self._report_interval = int(perf_cfg.get("report_interval_minutes", 10))
        self._report_path = str(
            perf_cfg.get("report_path", "/data/reports/strategy_performance.json")
        )
        self._kill_switch_enabled = bool(
            perf_cfg.get("kill_switch", {}).get("enabled", True)
        )
        self._strategy_names = list(strategy_names)
        self._last_report_at: datetime | None = None
        # Consecutive negative-Sharpe tracking for weight penalty
        self._neg_sharpe_threshold = int(perf_cfg.get("negative_sharpe_days", 3))
        self._neg_sharpe_count: dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def consume_pending_entry_strategy(
        self, symbol: str, broker_state, strategy_names: list[str]
    ) -> str | None:
        data = broker_state.pending_entry_strategy.pop(symbol, None)
        if not data:
            return None
        strategy = data.get("strategy")
        if strategy and strategy in strategy_names:
            return str(strategy)
        return None

    def prune_pending_entry_strategies(
        self, now: datetime, broker_state
    ) -> None:
        if not broker_state.pending_entry_strategy:
            return
        cutoff = now - timedelta(days=1)
        stale = [
            symbol
            for symbol, data in broker_state.pending_entry_strategy.items()
            if isinstance(data, dict) and data.get("ts") and data["ts"] < cutoff
        ]
        for symbol in stale:
            broker_state.pending_entry_strategy.pop(symbol, None)

    def record_trade(
        self,
        broker_state,
        strategy: str | None,
        symbol: str,
        pnl_pct: float,
        ts: datetime,
    ) -> None:
        record = {"ts": ts, "pnl_pct": pnl_pct}
        broker_state.symbol_trades.setdefault(symbol, []).append(record)
        if strategy and strategy in self._strategy_names:
            broker_state.strategy_trades.setdefault(strategy, []).append(record)
            STRATEGY_TRADES_REALIZED.labels(strategy=strategy).inc()

    @staticmethod
    def prune_trade_records(
        records: list[dict], cutoff: datetime
    ) -> list[dict]:
        if not records:
            return []
        return [r for r in records if r.get("ts") and r["ts"] >= cutoff]

    @staticmethod
    def compute_trade_stats(records: list[dict]) -> dict:
        if not records:
            return {
                "trades": 0, "win_rate": 0.0, "avg_pnl_pct": 0.0,
                "drawdown_pct": 0.0, "sharpe": 0.0, "profit_factor": 0.0,
            }
        ordered = sorted(records, key=lambda r: r.get("ts") or datetime.min)
        total = len(ordered)
        pnls = [float(r.get("pnl_pct", 0.0) or 0.0) for r in ordered]
        wins = sum(1 for p in pnls if p > 0.0)
        avg_pnl = sum(pnls) / total
        equity = 100.0
        peak = 100.0
        max_dd = 0.0
        for pnl_pct in pnls:
            equity *= 1.0 + pnl_pct / 100.0
            if equity > peak:
                peak = equity
            if peak > 0:
                drawdown = (peak - equity) / peak * 100.0
                if drawdown > max_dd:
                    max_dd = drawdown
        # Sharpe ratio (annualised, assuming daily trades ~252 trading days)
        import math
        if total > 1:
            variance = sum((p - avg_pnl) ** 2 for p in pnls) / (total - 1)
            std_dev = math.sqrt(variance) if variance > 0 else 0.0
            sharpe = (avg_pnl / std_dev * math.sqrt(252)) if std_dev > 0 else 0.0
        else:
            sharpe = 0.0
        # Profit factor = gross_win / gross_loss
        gross_win = sum(p for p in pnls if p > 0.0)
        gross_loss = sum(-p for p in pnls if p < 0.0)
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (gross_win if gross_win > 0 else 0.0)
        return {
            "trades": total,
            "win_rate": wins / total,
            "avg_pnl_pct": avg_pnl,
            "drawdown_pct": max_dd,
            "sharpe": sharpe,
            "profit_factor": profit_factor,
        }

    def update_from_positions(
        self, broker_state, portfolio: dict, broker_name: str,
        strategy_names: list[str], last_prices: dict[str, float],
    ) -> None:
        if not self._enabled:
            return
        now = datetime.now(timezone.utc)
        self.prune_pending_entry_strategies(now, broker_state)
        positions = portfolio.get("positions", {}) or {}
        symbols = set(positions.keys()) | set(broker_state.position_state.keys())
        for symbol in symbols:
            current = positions.get(symbol) or {}
            curr_qty = float(current.get("qty", 0.0) or 0.0)
            curr_avg_entry = current.get("avg_entry")
            if curr_avg_entry is not None:
                try:
                    curr_avg_entry = float(curr_avg_entry)
                except (TypeError, ValueError):
                    curr_avg_entry = None
            prev = broker_state.position_state.get(symbol)
            if prev is None:
                if curr_qty != 0:
                    strategy = self.consume_pending_entry_strategy(
                        symbol, broker_state, strategy_names
                    )
                    # Use last_prices as fallback when broker doesn't provide avg_entry
                    # (e.g. Binance Spot) so that stop-loss and trailing-stop logic works.
                    effective_entry = curr_avg_entry or last_prices.get(symbol)
                    broker_state.position_state[symbol] = {
                        "qty": curr_qty,
                        "avg_entry": effective_entry,
                        "strategy": strategy,
                        "opened_at": now,
                        "peak_price": effective_entry,
                        "took_partial": False,
                    }
                continue
            prev_qty = float(prev.get("qty", 0.0) or 0.0)
            prev_avg = prev.get("avg_entry")
            strategy = prev.get("strategy")
            if curr_qty > prev_qty:
                # Detect fresh re-entry after dust: previous qty was < 1% of new qty
                # (e.g. 0.007 FIL dust → 2050 FIL). Reset opened_at so time_exit
                # doesn't fire immediately using the stale timestamp from the dust era.
                _is_fresh_reentry = prev_qty > 0 and prev_qty < 0.01 * curr_qty
                if strategy is None or _is_fresh_reentry:
                    strategy = self.consume_pending_entry_strategy(
                        symbol, broker_state, strategy_names
                    )
                prev["qty"] = curr_qty
                if curr_avg_entry is not None:
                    prev["avg_entry"] = curr_avg_entry
                elif _is_fresh_reentry:
                    # Binance Spot: no cost basis — fall back to last_price on fresh entry
                    _lp_entry = last_prices.get(symbol)
                    if _lp_entry:
                        prev["avg_entry"] = _lp_entry
                if strategy is not None:
                    prev["strategy"] = strategy
                if _is_fresh_reentry:
                    prev["opened_at"] = now
                    prev["took_partial"] = False
                _lp = last_prices.get(symbol)
                if _lp is not None:
                    prev["peak_price"] = max(float(prev.get("peak_price") or 0), _lp)
                continue
            if curr_qty < prev_qty:
                exit_price = last_prices.get(symbol)
                entry_price = prev_avg or curr_avg_entry
                if entry_price and exit_price:
                    direction = 1.0 if prev_qty > 0 else -1.0
                    pnl_pct = (exit_price - entry_price) / entry_price * 100.0 * direction
                    self.record_trade(broker_state, strategy, symbol, pnl_pct, now)
                if curr_qty == 0:
                    broker_state.position_state.pop(symbol, None)
                else:
                    prev["qty"] = curr_qty
                    if curr_avg_entry is not None:
                        prev["avg_entry"] = curr_avg_entry
                continue
            # qty unchanged — update peak price for trailing stop
            _lp = last_prices.get(symbol)
            if _lp is not None:
                prev["peak_price"] = max(float(prev.get("peak_price") or 0), _lp)

    def maybe_report(
        self,
        broker_states: dict,
        strategy_disabled_globally_fn,
    ) -> str | None:
        if not self._enabled:
            return None
        now = datetime.now(timezone.utc)
        interval_seconds = self._report_interval * 60
        if self._last_report_at and (now - self._last_report_at).total_seconds() < interval_seconds:
            return None
        cutoff = now - timedelta(days=self._window_days)
        strategy_report: dict[str, dict] = {}
        symbol_report: dict[str, dict] = {}
        brokers_report: dict[str, dict] = {}
        aggregate_strategy_records: dict[str, list[dict]] = {
            name: [] for name in self._strategy_names
        }
        aggregate_symbol_records: dict[str, list[dict]] = {}

        for broker_name, broker_state in broker_states.items():
            broker_strategy_report: dict[str, dict] = {}
            broker_symbol_report: dict[str, dict] = {}
            for name in self._strategy_names:
                records = self.prune_trade_records(
                    broker_state.strategy_trades.get(name, []), cutoff
                )
                broker_state.strategy_trades[name] = records
                stats = self.compute_trade_stats(records)
                broker_strategy_report[name] = stats | {
                    "disabled": name in broker_state.disabled_strategies
                }
                aggregate_strategy_records[name].extend(records)
                if (
                    self._kill_switch_enabled
                    and name not in broker_state.disabled_strategies
                    and stats["trades"] >= self._min_trades
                    and (
                        stats["win_rate"] < self._min_win_rate
                        or stats["drawdown_pct"] > self._max_drawdown
                    )
                ):
                    broker_state.disabled_strategies.add(name)
                    logging.warning(
                        "Strategy %s disabled by kill switch (broker=%s trades=%d win_rate=%.2f drawdown=%.2f)",
                        name,
                        broker_name,
                        stats["trades"],
                        stats["win_rate"],
                        stats["drawdown_pct"],
                    )
            for symbol, records in list(broker_state.symbol_trades.items()):
                trimmed = self.prune_trade_records(records, cutoff)
                if trimmed:
                    broker_state.symbol_trades[symbol] = trimmed
                    broker_symbol_report[symbol] = self.compute_trade_stats(trimmed)
                    aggregate_symbol_records.setdefault(symbol, []).extend(trimmed)
                else:
                    broker_state.symbol_trades.pop(symbol, None)
            brokers_report[broker_name] = {
                "strategies": broker_strategy_report,
                "symbols": broker_symbol_report,
                "disabled_strategies": sorted(broker_state.disabled_strategies),
            }
            broker_state.performance_last_report_at = now

        for name in self._strategy_names:
            records = aggregate_strategy_records.get(name, [])
            stats = self.compute_trade_stats(records)
            STRATEGY_WIN_RATE.labels(strategy=name).set(stats["win_rate"])
            STRATEGY_AVG_PNL_PCT.labels(strategy=name).set(stats["avg_pnl_pct"])
            STRATEGY_DRAWDOWN_PCT.labels(strategy=name).set(stats["drawdown_pct"])
            STRATEGY_DISABLED.labels(strategy=name).set(
                1 if strategy_disabled_globally_fn(name) else 0
            )
            # Asset-class breakdown for new metrics
            crypto_records = [r for r in records if "/" in str(r.get("symbol", ""))]
            equity_records = [r for r in records if "/" not in str(r.get("symbol", ""))]
            for asset_class, ac_records in (("equities", equity_records), ("crypto", crypto_records)):
                ac_stats = self.compute_trade_stats(ac_records) if ac_records else {"sharpe": 0.0, "profit_factor": 0.0}
                STRATEGY_SHARPE_RATIO.labels(strategy=name, asset_class=asset_class).set(ac_stats["sharpe"])
                STRATEGY_PROFIT_FACTOR.labels(strategy=name, asset_class=asset_class).set(ac_stats["profit_factor"])
            # Consecutive negative-Sharpe tracking: increment counter when Sharpe < 0,
            # reset when positive. Weight penalty applied via get_neg_sharpe_weight_mult().
            if stats["trades"] >= self._min_trades:
                if stats["sharpe"] < 0:
                    self._neg_sharpe_count[name] = self._neg_sharpe_count.get(name, 0) + 1
                    if self._neg_sharpe_count[name] >= self._neg_sharpe_threshold:
                        logging.warning(
                            "Strategy %s Sharpe < 0 for %d consecutive reports "
                            "(sharpe=%.2f trades=%d) — weight halved",
                            name, self._neg_sharpe_count[name], stats["sharpe"], stats["trades"],
                        )
                else:
                    self._neg_sharpe_count[name] = 0
            strategy_report[name] = stats | {
                "disabled": strategy_disabled_globally_fn(name)
            }

        for symbol, records in aggregate_symbol_records.items():
            symbol_report[symbol] = self.compute_trade_stats(records)

        report = {
            "generated_at": now.isoformat(),
            "window_days": self._window_days,
            "min_trades": self._min_trades,
            "min_win_rate": self._min_win_rate,
            "max_drawdown_pct": self._max_drawdown,
            "strategies": strategy_report,
            "symbols": symbol_report,
            "disabled_strategies": sorted(
                [name for name in self._strategy_names if strategy_disabled_globally_fn(name)]
            ),
            "brokers": brokers_report,
        }
        try:
            report_path = Path(self._report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2))
        except Exception as exc:
            logging.warning("Performance report write failed: %s", exc)
        self._last_report_at = now
        return json.dumps(report)

    def get_neg_sharpe_weight_mult(self, strategy_name: str) -> float:
        """Returns 0.5 weight multiplier when strategy has N consecutive negative-Sharpe reports."""
        if self._neg_sharpe_count.get(strategy_name, 0) >= self._neg_sharpe_threshold:
            return 0.5
        return 1.0
