# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import logging
from datetime import datetime, date, timezone

from app.utils.account import extract_equity_cash
from app.monitoring.metrics import (
    PNL,
    DRAWDOWN,
    ACCOUNT_TOTAL,
    ACCOUNT_CASH,
    ACCOUNT_BUYING_POWER,
    ACCOUNT_INVESTED,
    ACCOUNT_TOTAL_BY_BROKER,
    ACCOUNT_CASH_BY_BROKER,
    ACCOUNT_BUYING_POWER_BY_BROKER,
    ACCOUNT_INVESTED_BY_BROKER,
    PNL_BY_BROKER,
    DRAWDOWN_BY_BROKER,
    BROKER_MARKET_OPEN,
)


def _var_cvar_from_history(history: list[float], confidence: float) -> dict[str, float]:
    if len(history) < 2:
        return {"var_pct": 0.0, "cvar_pct": 0.0}
    returns = []
    for idx in range(1, len(history)):
        prev = history[idx - 1]
        curr = history[idx]
        if not prev:
            continue
        returns.append((curr - prev) / prev * 100.0)
    if not returns:
        return {"var_pct": 0.0, "cvar_pct": 0.0}
    returns = sorted(returns)
    cutoff = max(int((1.0 - confidence) * len(returns)) - 1, 0)
    var_val = returns[cutoff]
    tail = [r for r in returns if r <= var_val]
    cvar_val = sum(tail) / len(tail) if tail else var_val
    return {"var_pct": abs(var_val), "cvar_pct": abs(cvar_val)}


class AccountMetricsUpdater:
    """Tracks account equity, drawdown, PnL, VaR/CVaR, and drift."""

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._equity_start: float | None = None
        self._equity_peak: float | None = None
        self._current_drawdown_pct: float = 0.0
        self._day_start_date: date | None = None
        self._day_start_equity: float | None = None
        self._equity_history: list[float] = []
        self._var_cvar: dict[str, float] = {}

    @property
    def equity_start(self) -> float | None:
        return self._equity_start

    @equity_start.setter
    def equity_start(self, value: float | None) -> None:
        self._equity_start = value

    @property
    def equity_peak(self) -> float | None:
        return self._equity_peak

    @equity_peak.setter
    def equity_peak(self, value: float | None) -> None:
        self._equity_peak = value

    @property
    def current_drawdown_pct(self) -> float:
        return self._current_drawdown_pct

    @property
    def var_cvar(self) -> dict[str, float]:
        return self._var_cvar

    def update_broker_equity(
        self, broker_name: str, broker_state, equity: float,
        last_equity: float | None = None, today_deposits: float = 0.0
    ) -> None:
        if broker_state.equity_start is None:
            broker_state.equity_start = equity
        if broker_state.equity_peak is None or equity > broker_state.equity_peak:
            broker_state.equity_peak = equity
        base_equity = last_equity if last_equity else broker_state.equity_start
        if base_equity:
            pnl_pct = (equity - base_equity) / base_equity * 100.0
            PNL_BY_BROKER.labels(broker=broker_name).set(pnl_pct)
        if broker_state.equity_peak:
            drawdown_pct = (broker_state.equity_peak - equity) / broker_state.equity_peak * 100.0
            broker_state.current_drawdown_pct = max(drawdown_pct, 0.0)
            DRAWDOWN_BY_BROKER.labels(broker=broker_name).set(max(drawdown_pct, 0.0))
        today = datetime.now(timezone.utc).date()
        if broker_state.day_start_date != today or broker_state.day_start_equity is None:
            broker_state.day_start_date = today
            broker_state.day_start_equity = equity
            # Capture deposits already included in today's opening equity so they
            # are not counted as trading profit later in the day.
            broker_state.day_deposits_baseline = today_deposits
        if broker_state.day_start_equity:
            # Only count deposits that arrived AFTER the day baseline was set.
            new_deposits = max(0.0, today_deposits - broker_state.day_deposits_baseline)
            trading_equity = equity - new_deposits
            day_pnl_pct = (trading_equity - broker_state.day_start_equity) / broker_state.day_start_equity * 100.0
            if new_deposits > 0:
                logging.info(
                    "%s: excluding %.2f intraday deposit from day P&L (raw day P&L would be %.2f%%)",
                    broker_name, new_deposits, (equity - broker_state.day_start_equity) / broker_state.day_start_equity * 100.0,
                )
            broker_state.risk.update_daily_loss(day_pnl_pct)
            broker_state.day_pnl_pct = day_pnl_pct

    def update(self, broker, broker_states: dict, broker_name: str, account: dict | None = None) -> dict | None:
        if account is None:
            try:
                account = broker.get_account()
            except (ConnectionError, TimeoutError, OSError) as exc:
                logging.warning("Account metrics update failed: %s", exc)
                return None
        total_val = cash_val = buying_power_val = None
        broker_equities: dict[str, float] = {}
        if isinstance(account, dict):
            if "brokers" in account and isinstance(account["brokers"], dict):
                total_val = float(account.get("equity") or 0.0)
                cash_val = float(account.get("cash") or 0.0)
                buying_power_val = float(account.get("buying_power") or 0.0)
                for name, details in account["brokers"].items():
                    equity, cash, buying_power = extract_equity_cash(details)
                    broker_last_equity = None
                    if details.get("last_equity") is not None:
                        try:
                            broker_last_equity = float(details["last_equity"])
                        except (TypeError, ValueError):
                            broker_last_equity = None
                    ACCOUNT_TOTAL_BY_BROKER.labels(broker=name).set(equity)
                    ACCOUNT_CASH_BY_BROKER.labels(broker=name).set(cash)
                    ACCOUNT_BUYING_POWER_BY_BROKER.labels(broker=name).set(buying_power)
                    ACCOUNT_INVESTED_BY_BROKER.labels(broker=name).set(equity - cash)
                    broker_equities[str(name)] = equity
                    broker_today_deposits = float(details.get("today_deposits", 0) or 0)
                    bstate = broker_states.get(str(name))
                    if bstate:
                        self.update_broker_equity(
                            str(name), bstate, equity,
                            last_equity=broker_last_equity,
                            today_deposits=broker_today_deposits,
                        )
                        bstate.buying_power = buying_power
            else:
                total_val, cash_val, buying_power_val = extract_equity_cash(account)
        if total_val is None or cash_val is None:
            return account
        if not broker_equities:
            broker_equities[broker_name] = total_val
            single_deposits = float(account.get("today_deposits", 0) or 0) if isinstance(account, dict) else 0.0
            bstate = broker_states.get(broker_name)
            if bstate:
                self.update_broker_equity(broker_name, bstate, total_val, today_deposits=single_deposits)
        if self._equity_start is None:
            self._equity_start = total_val
        if self._equity_peak is None or total_val > self._equity_peak:
            self._equity_peak = total_val
        last_equity = None
        if isinstance(account, dict):
            last_equity = account.get("last_equity")
        base_equity = self._equity_start
        if last_equity is not None:
            try:
                last_equity_val = float(last_equity)
            except (TypeError, ValueError):
                last_equity_val = None
            if last_equity_val:
                base_equity = last_equity_val
        if base_equity:
            pnl_pct = (total_val - base_equity) / base_equity * 100.0
            PNL.set(pnl_pct)
        if self._equity_peak:
            drawdown_pct = (self._equity_peak - total_val) / self._equity_peak * 100.0
            DRAWDOWN.set(max(drawdown_pct, 0.0))
            self._current_drawdown_pct = max(drawdown_pct, 0.0)
        ACCOUNT_TOTAL.set(total_val)
        ACCOUNT_CASH.set(cash_val)
        if buying_power_val is not None:
            ACCOUNT_BUYING_POWER.set(buying_power_val)
        ACCOUNT_INVESTED.set(total_val - cash_val)
        today = datetime.now(timezone.utc).date()
        if self._day_start_date != today or self._day_start_equity is None:
            self._day_start_date = today
            self._day_start_equity = total_val
        self._update_var_cvar(total_val, broker_equities, broker_states)
        return account

    def get_day_pnl_pct(self) -> float | None:
        if self._day_start_equity is None or self._equity_start is None:
            return None
        # We need current equity but don't store it separately
        return None

    def update_var_cvar(self, broker_state) -> dict:
        return broker_state.var_cvar

    def _update_var_cvar(
        self, total_val: float, broker_equities: dict[str, float], broker_states: dict
    ) -> None:
        var_cfg = self._cfg.get("risk", {}).get("var", {}) or {}
        if not var_cfg.get("enabled", False):
            return
        window = int(var_cfg.get("window", 60))
        confidence = float(var_cfg.get("confidence", 0.95))
        self._equity_history.append(total_val)
        if len(self._equity_history) > window:
            self._equity_history = self._equity_history[-window:]
        self._var_cvar = _var_cvar_from_history(self._equity_history, confidence)
        for name, equity in broker_equities.items():
            broker_state = broker_states.get(name)
            if not broker_state:
                continue
            broker_state.equity_history.append(float(equity))
            if len(broker_state.equity_history) > window:
                broker_state.equity_history = broker_state.equity_history[-window:]
            broker_state.var_cvar = _var_cvar_from_history(broker_state.equity_history, confidence)

    def var_limit_reason(self, broker_state) -> str | None:
        var_cfg = self._cfg.get("risk", {}).get("var", {}) or {}
        if not var_cfg.get("enabled", False):
            return None
        max_var = float(var_cfg.get("max_var_pct", 0.0) or 0.0)
        max_cvar = float(var_cfg.get("max_cvar_pct", 0.0) or 0.0)
        if max_var and self._var_cvar.get("var_pct", 0.0) > max_var:
            return "var_limit"
        if max_cvar and self._var_cvar.get("cvar_pct", 0.0) > max_cvar:
            return "cvar_limit"
        broker_stats = broker_state.var_cvar
        if max_var and broker_stats.get("var_pct", 0.0) > max_var:
            return "var_limit_broker"
        if max_cvar and broker_stats.get("cvar_pct", 0.0) > max_cvar:
            return "cvar_limit_broker"
        return None

    def update_market_open_metrics(
        self, cfg: dict, broker_names: list[str], market_open: bool, last_market_open: bool | None
    ) -> bool | None:
        brokers_cfg = cfg.get("brokers", {})
        names = [
            name
            for name, c in brokers_cfg.items()
            if not isinstance(c, dict) or c.get("enabled", True)
        ]
        if not names:
            names = broker_names
        for name in names:
            BROKER_MARKET_OPEN.labels(broker=name).set(1 if market_open else 0)
        if market_open != last_market_open:
            state = "open" if market_open else "closed"
            logging.info("Market is %s; %s trading loop.", state, "starting" if market_open else "waiting")
        return market_open
