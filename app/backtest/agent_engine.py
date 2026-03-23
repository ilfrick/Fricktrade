# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd

from app.agents.trader import TradingAgent
from app.data.downloader import download_alpaca_bars
from app.brokers.config_utils import get_alpaca_account_cfg
from app.backtest.sampling import BacktestWindow, build_backtest_plan


@dataclass
class BacktestResult:
    start_value: float
    end_value: float
    return_pct: float
    trades: int
    start: str = ""
    end: str = ""
    symbols: list[str] = field(default_factory=list)
    trade_details: list[dict] = field(default_factory=list)  # per-trade records for analysis
    signal_log: list[dict] = field(default_factory=list)  # per-bar strategy signals vs final action


@dataclass
class BacktestPlanResult:
    runs: list[BacktestResult]
    average_return_pct: float
    total_trades: int


@dataclass
class _FrameData:
    values: list[list[float]]
    indexer: list[int]


class SimBroker:
    def __init__(self, initial_cash: float, commission_pct: float, slippage_bps: float = 0.0, spread_bps: float = 0.0):
        self.cash = float(initial_cash)
        self.commission_pct = float(commission_pct)
        self.slippage_bps = float(slippage_bps)
        self.spread_bps = float(spread_bps)
        self.positions: dict[str, float] = {}
        self.avg_entry_prices: dict[str, float] = {}
        self.current_prices: dict[str, float] = {}
        self._order_id = 0
        self.trades = 0
        self.trade_log: list[dict] = []  # detailed per-trade records
        self.signal_log: list[dict] = []  # per-bar strategy signals

    def get_account(self) -> dict:
        equity = self.cash + sum(self.positions.get(sym, 0.0) * self.current_prices.get(sym, 0.0) for sym in self.positions)
        return {"equity": equity, "cash": self.cash}

    def get_positions(self) -> list[dict]:
        results = []
        for symbol, qty in self.positions.items():
            if qty == 0.0:
                continue
            price = self.current_prices.get(symbol, 0.0)
            results.append(
                {
                    "symbol": symbol,
                    "qty": qty,
                    "market_value": qty * price,
                    "current_price": price,
                    "avg_entry_price": self.avg_entry_prices.get(symbol, 0.0),
                }
            )
        return results

    def get_open_orders(self) -> list[dict]:
        return []

    def place_order(self, symbol: str, side: str, qty: float, order_type: str, **kwargs) -> str:
        price = self.current_prices.get(symbol)
        if price is None:
            raise ValueError(f"Missing price for {symbol}")
        exec_price = _apply_slippage(price, side, self.slippage_bps, self.spread_bps)
        cost = qty * exec_price
        commission = cost * (self.commission_pct / 100.0)
        # Capture avg_entry BEFORE position update (needed for sell P&L calc)
        _pre_avg_entry = self.avg_entry_prices.get(symbol, 0.0)
        if side.lower() == "buy":
            self.cash -= cost + commission
            prev_qty = self.positions.get(symbol, 0.0)
            prev_avg = _pre_avg_entry
            new_qty = prev_qty + qty
            if new_qty > 0:
                self.avg_entry_prices[symbol] = (prev_avg * prev_qty + exec_price * qty) / new_qty
            self.positions[symbol] = new_qty
        else:
            self.cash += cost - commission
            self.positions[symbol] = self.positions.get(symbol, 0.0) - qty
            # Clean up dust: if remaining position is <$1 notional, zero it out
            remaining = self.positions.get(symbol, 0.0)
            if remaining <= 0 or (remaining * exec_price < 1.0):
                self.positions.pop(symbol, None)
                self.avg_entry_prices.pop(symbol, None)
        self._order_id += 1
        self.trades += 1
        trade_rec = {
            "id": self._order_id,
            "symbol": symbol,
            "side": side.lower(),
            "qty": qty,
            "price": exec_price,
            "commission": commission,
            "notional": cost,
        }
        if side.lower() == "sell":
            avg_entry = _pre_avg_entry if _pre_avg_entry > 0 else exec_price
            trade_rec["avg_entry"] = avg_entry
            trade_rec["pnl_pct"] = (exec_price - avg_entry) / avg_entry * 100.0 if avg_entry > 0 else 0.0
        self.trade_log.append(trade_rec)
        return f"sim-{self._order_id}"

    def close_position(self, symbol: str, **kwargs) -> None:
        qty = self.positions.get(symbol, 0.0)
        if qty == 0.0:
            return
        side = "sell" if qty > 0 else "buy"
        self.place_order(symbol, side, abs(qty), "market")
        self.positions.pop(symbol, None)


class SimBrokerRouter:
    def __init__(self, brokers: dict[str, SimBroker], routing: dict | None = None):
        self._brokers = brokers
        self._routing = routing or {}
        self.signal_log: list[dict] = []  # per-bar strategy signals (set by backtest loop)

    @property
    def brokers(self) -> dict[str, SimBroker]:
        return dict(self._brokers)

    def get_account(self) -> dict:
        total_equity = 0.0
        total_cash = 0.0
        per_broker = {}
        for name, broker in self._brokers.items():
            account = broker.get_account()
            equity = float(account.get("equity") or 0.0)
            cash = float(account.get("cash") or 0.0)
            total_equity += equity
            total_cash += cash
            per_broker[name] = {"equity": equity, "cash": cash, "raw": account}
        return {"equity": total_equity, "cash": total_cash, "brokers": per_broker}

    def get_positions(self) -> list[dict]:
        results: list[dict] = []
        for name, broker in self._brokers.items():
            for pos in broker.get_positions():
                item = dict(pos)
                item["broker"] = name
                results.append(item)
        return results

    def get_open_orders(self) -> list[dict]:
        return []

    def place_order(self, symbol: str, side: str, qty: float, order_type: str, **kwargs) -> str:
        broker_name = kwargs.get("broker") or self._resolve_broker(symbol, kwargs.get("strategy"))
        broker = self._brokers.get(str(broker_name))
        if broker is None:
            raise ValueError(f"Unknown broker {broker_name!r} for {symbol}")
        return broker.place_order(symbol, side, qty, order_type, **kwargs)

    def close_position(self, symbol: str, **kwargs) -> None:
        broker_name = kwargs.get("broker")
        if broker_name:
            broker = self._brokers.get(str(broker_name))
            if broker:
                broker.close_position(symbol)
            return
        for name, broker in self._brokers.items():
            if symbol in broker.positions:
                broker.close_position(symbol)

    def cancel_order(self, order_id: str, **kwargs) -> None:
        return

    def _resolve_broker(self, symbol: str, strategy: str | None) -> str:
        symbol_map = self._routing.get("symbols", {}) or {}
        if symbol in symbol_map:
            return str(symbol_map[symbol])
        strategy_map = self._routing.get("strategies", {}) or {}
        if strategy and strategy in strategy_map:
            return str(strategy_map[strategy])
        default = self._routing.get("default")
        if default:
            return str(default)
        return next(iter(self._brokers.keys()))


def run_agent_backtest(cfg: dict) -> BacktestResult | BacktestPlanResult:
    data_cfg = cfg["data"]
    backtest_cfg = cfg["backtest"]
    interval = data_cfg.get("interval")
    data_dir = Path(backtest_cfg["data_dir"])
    plan = build_backtest_plan(cfg)
    if plan:
        results = []
        for window in plan:
            result = _run_agent_backtest_single(cfg, window.symbols, window.start, window.end)
            results.append(result)
        total_trades = sum(r.trades for r in results)
        avg_return = sum(r.return_pct for r in results) / len(results)
        return BacktestPlanResult(runs=results, average_return_pct=avg_return, total_trades=total_trades)

    symbols_source = str(backtest_cfg.get("symbols_source", "data"))
    symbols = data_cfg.get("symbols", [])
    if symbols_source == "config":
        symbols = backtest_cfg.get("symbols", []) or data_cfg.get("symbols", [])
    elif symbols_source == "dynamic":
        symbols = _resolve_dynamic_symbols(cfg)
    elif symbols_source == "data_dir":
        symbols = _symbols_from_data_dir(data_dir, interval)
    if not symbols:
        raise ValueError("No symbols configured for backtest.")
    start = datetime.strptime(backtest_cfg["start"], "%Y-%m-%d")
    end = datetime.strptime(backtest_cfg["end"], "%Y-%m-%d")
    return _run_agent_backtest_single(cfg, symbols, start, end)


def _run_agent_backtest_single(cfg: dict, symbols: list[str], start: datetime, end: datetime) -> BacktestResult:
    data_cfg = cfg["data"]
    backtest_cfg = cfg["backtest"]
    interval = data_cfg.get("interval")
    data_dir = Path(backtest_cfg["data_dir"])

    if backtest_cfg.get("download_missing_symbols", False):
        _download_missing_bars(cfg, symbols, interval, data_dir)

    cache_cfg = backtest_cfg.get("cache", {})
    if not isinstance(cache_cfg, dict):
        cache_cfg = {}
    cache_enabled = bool(cache_cfg.get("enabled", True))
    cache_format = str(cache_cfg.get("format", "npz")).lower()
    cache_compress = bool(cache_cfg.get("compress", False))
    cache_dir = None
    if cache_enabled:
        if cache_format != "npz":
            logging.warning("Unsupported backtest cache format %s; caching disabled", cache_format)
            cache_enabled = False
        else:
            cache_dir_value = cache_cfg.get("dir") or (data_dir / "backtest_cache")
            cache_dir = Path(cache_dir_value)
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                logging.warning("Failed to create backtest cache dir %s: %s", cache_dir, exc)
                cache_enabled = False

    frames = {}
    for symbol in symbols:
        path = data_dir / f"{symbol.replace('/', '_').replace('.', '_')}_{interval}.csv"
        if not path.exists():
            continue
        frames[symbol] = _load_csv(
            path,
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
            cache_format=cache_format,
            cache_compress=cache_compress,
        )
    if not frames:
        raise FileNotFoundError(f"No CSV data found for symbols in {data_dir}")

    timeline_index, prepared_frames = _prepare_backtest_frames(frames, start, end)
    if not prepared_frames:
        raise FileNotFoundError(f"No CSV data found for symbols in {data_dir}")
    sim_cfg = _backtest_cfg_override(cfg)
    broker = _build_sim_broker(cfg, backtest_cfg)
    agent = TradingAgent(broker, sim_cfg)

    interval_minutes = _interval_minutes(interval or "1m")
    lookback_minutes = int(sim_cfg["strategy"]["params"].get("lookback_minutes", 30))
    lookback_bars = max(2, int(lookback_minutes / interval_minutes))

    # Strategies need sufficient history. At 1m bars, crypto_momentum slow_window=300,
    # trend_following EMA-150 (needs ~450 bars to converge), indicators need ~100.
    # Use at least 500 bars to ensure all strategies have enough data at any interval.
    state = {sym: _SymbolState(max_len=max(lookback_bars, 500)) for sym in prepared_frames}
    start_value = broker.get_account()["equity"]

    news_cache = _load_backtest_news(backtest_cfg)
    timeline = timeline_index.to_pydatetime()
    _bt_signal_counts: dict[str, int] = {}
    _bt_total_bars = 0
    for ts_idx, ts in enumerate(timeline):
        _apply_news_cache(agent, news_cache, ts)
        for symbol, frame_data in prepared_frames.items():
            pos = frame_data.indexer[ts_idx]
            if pos < 0:
                continue
            row = frame_data.values[pos]
            sym_state = state[symbol]
            sym_state.update_from_values(ts, row)
            market_state = sym_state.market_state()
            if market_state["last_price"] is None:
                continue
            broker.current_prices[symbol] = market_state["last_price"]
            portfolio = agent._get_portfolio_snapshot()
            agent._enrich_market_state(market_state, portfolio, symbol)
            market_state["strategy_symbols"] = getattr(agent, "_symbols_by_strategy", {})
            agent.run_once(symbol, market_state)
            # Capture per-strategy signals when any strategy has a non-hold vote
            _bar_sigs = getattr(agent, "_last_bar_signals", None)
            if _bar_sigs and any(
                s.get("action") != "hold" for s in _bar_sigs.get("signals", [])
            ):
                _bar_sigs["ts"] = str(ts)
                _bar_sigs["price"] = market_state["last_price"]
                broker.signal_log.append(_bar_sigs)
            _bt_total_bars += 1
        if ts_idx > 0 and ts_idx % 2000 == 0:
            logging.info("Backtest progress: bar %d/%d, trades=%d", ts_idx, len(timeline), broker.trades)
        # Drain order queue: SimBroker completes instantly but queue processes one per update()
        _drain_order_queues(agent)
        agent._flush_order_responses()
        # Update position_state so stops/trailing stops can fire next bar
        _update_backtest_positions(agent, broker)

    # Final flush for any remaining orders
    _drain_order_queues(agent)
    agent._flush_order_responses()
    end_value = broker.get_account()["equity"]
    return BacktestResult(
        start_value=start_value,
        end_value=end_value,
        return_pct=(end_value - start_value) / start_value * 100.0,
        trades=broker.trades,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        symbols=sorted(prepared_frames.keys()),
        trade_details=_collect_trade_log(broker),
        signal_log=broker.signal_log,
    )


def _collect_trade_log(broker) -> list[dict]:
    """Collect trade_log from SimBroker or all sub-brokers in SimBrokerRouter."""
    if hasattr(broker, "trade_log"):
        return broker.trade_log
    if hasattr(broker, "_brokers"):
        logs: list[dict] = []
        for name, sub in broker._brokers.items():
            for rec in sub.trade_log:
                logs.append({**rec, "broker": name})
        return sorted(logs, key=lambda x: x.get("id", 0))
    return []


def _update_backtest_positions(agent: TradingAgent, broker) -> None:
    """Sync position_state from SimBroker so stops/trailing stops can fire."""
    portfolio = agent._get_portfolio_snapshot()
    # Update last_prices on broker_state from SimBroker's current_prices
    broker_state = agent._broker_state(agent._broker_name)
    if hasattr(broker, "current_prices"):
        broker_state.last_prices.update(broker.current_prices)
    elif hasattr(broker, "_brokers"):
        # SimBrokerRouter: merge all broker prices
        for name, sub_broker in broker._brokers.items():
            bs = agent._broker_state(name)
            bs.last_prices.update(sub_broker.current_prices)
    # Populate position_state with entry prices, peak prices, etc.
    if agent._perf_tracker.enabled:
        brokers_data = portfolio.get("brokers", {})
        if isinstance(brokers_data, dict) and brokers_data:
            for broker_name in brokers_data.keys():
                bp = agent._portfolio_for_broker(portfolio, broker_name)
                bs = agent._broker_state(broker_name)
                agent._perf_tracker.update_from_positions(
                    bs, bp, broker_name, agent._strategy_names, bs.last_prices,
                )
        else:
            agent._perf_tracker.update_from_positions(
                broker_state, portfolio, agent._broker_name,
                agent._strategy_names, broker_state.last_prices,
            )
    # Cleanup: remove stale position_state entries for closed positions.
    # update_from_positions should handle this, but in backtest the order
    # queue drain + portfolio snapshot timing can leave stale entries that
    # cause the circuit breaker to spam on every bar.
    positions = portfolio.get("positions", {})
    for bs in agent._broker_states.values():
        stale = [s for s in bs.position_state if float(positions.get(s, {}).get("qty", 0)) == 0]
        for s in stale:
            bs.position_state.pop(s, None)


def _drain_order_queues(agent: TradingAgent, max_rounds: int = 100) -> None:
    """Repeatedly flush order queues until all SimBroker orders are processed."""
    for _ in range(max_rounds):
        has_pending = False
        for queue in agent._order_queues.values():
            with queue._lock:
                if queue._active is not None or queue._queue:
                    has_pending = True
        if not has_pending:
            break
        agent._update_open_order_queues()


def _load_backtest_news(backtest_cfg: dict) -> dict[str, set[str]]:
    source = str(backtest_cfg.get("news_source", "none")).lower()
    if source != "local":
        return {}
    path = Path(backtest_cfg.get("news_path", ""))
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    cache = {}
    for day, symbols in payload.items():
        if isinstance(symbols, list):
            cache[str(day)] = set(str(sym) for sym in symbols)
    return cache


def _apply_news_cache(agent: TradingAgent, news_cache: dict[str, set[str]], ts: datetime) -> None:
    if not news_cache:
        return
    day_key = ts.strftime("%Y-%m-%d")
    symbols = news_cache.get(day_key, set())
    agent._news_cache = {symbol: True for symbol in symbols}
    agent._news_cache_at = ts


def _symbols_from_data_dir(data_dir: Path, interval: str | None) -> list[str]:
    if not interval:
        return []
    pattern = f"*_{interval}.csv"
    symbols = []
    for path in data_dir.glob(pattern):
        name = path.stem
        suffix = f"_{interval}"
        if not name.endswith(suffix):
            continue
        raw = name[: -len(suffix)]
        if not raw:
            continue
        # Detect crypto: exactly TWO parts separated by _ where second is a
        # known quote currency (USD, USDT, etc.) → restore "/"
        parts = raw.split("_")
        _crypto_quotes = {"USD", "USDT", "USDC", "BUSD"}
        if len(parts) == 2 and parts[1] in _crypto_quotes:
            symbols.append(f"{parts[0]}/{parts[1]}")
        else:
            symbols.append(raw.replace("_", "."))
    return sorted(set(symbols))


def _build_sim_broker(cfg: dict, backtest_cfg: dict):
    exec_cfg = cfg.get("execution", {}).get("brokers", {})
    brokers_cfg = cfg.get("brokers", {})
    slippage_bps = float(backtest_cfg.get("slippage_bps", 0.0) or 0.0)
    spread_bps = float(backtest_cfg.get("spread_bps", 0.0) or 0.0)
    enabled = []
    if brokers_cfg.get("alpaca", {}).get("enabled", True):
        enabled.append("alpaca")
    if brokers_cfg.get("ibkr", {}).get("enabled", False):
        enabled.append("ibkr")
    if exec_cfg.get("enabled", False) and len(enabled) > 1:
        per_cash = float(backtest_cfg["initial_cash"]) / len(enabled)
        brokers = {
            name: SimBroker(per_cash, backtest_cfg["commission_pct"], slippage_bps, spread_bps)
            for name in enabled
        }
        return SimBrokerRouter(brokers, exec_cfg.get("routing", {}))
    return SimBroker(backtest_cfg["initial_cash"], backtest_cfg["commission_pct"], slippage_bps, spread_bps)


def _apply_slippage(price: float, side: str, slippage_bps: float, spread_bps: float) -> float:
    if price <= 0:
        return price
    slippage = slippage_bps / 10000.0
    half_spread = (spread_bps / 10000.0) / 2.0
    bump = slippage + half_spread
    if side.lower() == "sell":
        return max(price * (1.0 - bump), 0.0)
    return price * (1.0 + bump)


def _load_csv_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[0])
    df.rename(columns={df.columns[0]: "Datetime"}, inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["Datetime"])
    df["Datetime"] = df["Datetime"].dt.tz_convert(None)
    df = df.set_index("Datetime").sort_index()
    return df


def _cache_path_for_csv(path: Path, cache_dir: Path) -> Path:
    return cache_dir / f"{path.stem}.npz"


def _load_cached_npz(path: Path) -> pd.DataFrame:
    payload = np.load(path, allow_pickle=False)
    index_ns = payload["index_ns"]
    values = payload["values"]
    columns = payload["columns"].tolist()
    df = pd.DataFrame(values, columns=columns)
    df.index = pd.to_datetime(index_ns)
    df.index.name = "Datetime"
    return df


def _write_cached_npz(path: Path, df: pd.DataFrame, compress: bool) -> None:
    index_ns = df.index.view("int64")
    values = df.to_numpy()
    columns = np.asarray(df.columns, dtype=str)
    if compress:
        np.savez_compressed(path, index_ns=index_ns, values=values, columns=columns)
    else:
        np.savez(path, index_ns=index_ns, values=values, columns=columns)


def _load_csv(
    path: Path,
    *,
    cache_dir: Path | None = None,
    cache_enabled: bool = False,
    cache_format: str = "npz",
    cache_compress: bool = False,
) -> pd.DataFrame:
    if not cache_enabled or cache_dir is None:
        return _load_csv_raw(path)
    if cache_format != "npz":
        return _load_csv_raw(path)
    cache_path = _cache_path_for_csv(path, cache_dir)
    try:
        if cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
            return _load_cached_npz(cache_path)
    except Exception as exc:
        logging.warning("Failed to load backtest cache %s: %s", cache_path, exc)
    df = _load_csv_raw(path)
    try:
        _write_cached_npz(cache_path, df, cache_compress)
    except Exception as exc:
        logging.warning("Failed to write backtest cache %s: %s", cache_path, exc)
    return df


def _build_timeline(frames: dict[str, pd.DataFrame], start: datetime, end: datetime) -> list[datetime]:
    times = set()
    for frame in frames.values():
        subset = frame.loc[(frame.index >= start) & (frame.index <= end)]
        times.update(subset.index.to_pydatetime().tolist())
    return sorted(times)


def _prepare_backtest_frames(
    frames: dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
) -> tuple[pd.DatetimeIndex, dict[str, _FrameData]]:
    timeline = None
    subsets: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        subset = frame.loc[(frame.index >= start) & (frame.index <= end)]
        if subset.empty:
            continue
        subsets[symbol] = subset
        timeline = subset.index if timeline is None else timeline.union(subset.index)
    if not subsets:
        return pd.DatetimeIndex([]), {}
    timeline = timeline.sort_values()
    prepared: dict[str, _FrameData] = {}
    columns = ["Open", "High", "Low", "Close", "Volume"]
    for symbol, subset in subsets.items():
        missing = [col for col in columns if col not in subset.columns]
        if missing:
            subset = subset.copy()
            for col in missing:
                subset[col] = 0.0
        subset = subset[columns]
        prepared[symbol] = _FrameData(
            values=subset.to_numpy(),
            indexer=subset.index.get_indexer(timeline),
        )
    return timeline, prepared


def _interval_minutes(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    if interval.endswith("d"):
        return 60 * 24
    return 1


def _backtest_cfg_override(cfg: dict) -> dict:
    new_cfg = dict(cfg)
    new_cfg["data"] = dict(cfg["data"])
    new_cfg["news"] = dict(cfg.get("news", {}))
    new_cfg["data"]["dynamic_symbols"] = dict(cfg["data"].get("dynamic_symbols", {}))
    backtest_cfg = cfg.get("backtest", {})
    dynamic_enabled = bool(backtest_cfg.get("dynamic_symbols_enabled", False))
    news_enabled = bool(backtest_cfg.get("news_enabled", False))
    news_source = str(backtest_cfg.get("news_source", "none")).lower()
    new_cfg["data"]["dynamic_symbols"]["enabled"] = dynamic_enabled
    if dynamic_enabled:
        new_cfg["data"]["dynamic_symbols"]["provider"] = "data"
        ai_cfg = dict(new_cfg["data"]["dynamic_symbols"].get("ai_filter", {}))
        ai_cfg["enabled"] = False
        new_cfg["data"]["dynamic_symbols"]["ai_filter"] = ai_cfg
    if news_source == "local":
        new_cfg["news"]["enabled"] = False
    else:
        new_cfg["news"]["enabled"] = news_enabled
    new_cfg["execution"] = dict(cfg.get("execution", {}))
    new_cfg["execution"]["open_orders"] = {"enabled": False}
    # Disable algo slicing in backtest — SimBroker has no market impact
    new_cfg["execution"]["algos"] = dict(cfg.get("execution", {}).get("algos", {}))
    new_cfg["execution"]["algos"]["enabled"] = False
    # Disable LLM components that depend on live data (macro regime fetches
    # current FRED indicators, not historical — would block all entries if
    # current regime is "crisis")
    new_cfg["llm"] = dict(cfg.get("llm", {}))
    new_cfg["llm"]["macro_regime"] = {"enabled": False}
    new_cfg["llm"]["risk_interpreter"] = {"enabled": False}
    new_cfg["llm"]["sentiment"] = {"enabled": False}
    new_cfg["tactical_meta_orchestrator"] = {"enabled": False}
    new_cfg["strategic_meta_orchestrator"] = {"enabled": False}
    orchestrator_cfg = dict(cfg.get("orchestrator", {}))
    ml_cfg = dict(orchestrator_cfg.get("ml", {}))
    pretrain_cfg = dict(ml_cfg.get("pretrain", {}))
    pretrain_cfg["enabled"] = False
    pretrain_cfg["in_trader"] = False
    ml_cfg["pretrain"] = pretrain_cfg
    orchestrator_cfg["ml"] = ml_cfg
    rl_cfg = dict(orchestrator_cfg.get("rl", {}))
    rl_pretrain = dict(rl_cfg.get("pretrain", {}))
    rl_pretrain["enabled"] = False
    rl_pretrain["in_trader"] = False
    rl_cfg["pretrain"] = rl_pretrain
    orchestrator_cfg["rl"] = rl_cfg
    new_cfg["orchestrator"] = orchestrator_cfg
    # Disable per-broker cooldown in backtest — all symbols process in the
    # same tick so the cooldown blocks every symbol after the first trade.
    new_cfg["risk"] = dict(cfg.get("risk", {}))
    new_cfg["risk"]["cooldown_seconds"] = 0
    # Disable min_hold in backtest — all bars process in real-time seconds,
    # so wall-clock elapsed is always ~0 and min_hold blocks every exit.
    new_cfg["strategy"] = dict(cfg.get("strategy", {}))
    new_cfg["strategy"]["params"] = dict(cfg.get("strategy", {}).get("params", {}))
    new_cfg["strategy"]["params"]["min_hold_minutes"] = 0
    # Disable stop-exit re-entry cooldown — uses wall-clock time, blocks all
    # re-entries in backtest since 30 real minutes never elapse between bars.
    new_cfg["execution"] = dict(new_cfg.get("execution", {}))
    new_cfg["execution"]["stop_exit_reentry_cooldown_minutes"] = 0
    # Disable alpha_decay and regime_hold_minutes — both use datetime.now()
    # which is wall-clock, not simulated bar time.  In backtest the entire
    # timeline processes in minutes of real time, causing non-deterministic
    # spurious exits.
    new_cfg["strategy"]["params"]["alpha_decay_exit"] = {"enabled": False}
    new_cfg["strategy"]["params"]["regime_hold_minutes"] = {"enabled": False}
    return new_cfg


def _resolve_dynamic_symbols(cfg: dict) -> list[str]:
    data_cfg = cfg.get("data", {})
    backtest_cfg = cfg.get("backtest", {})
    symbols = data_cfg.get("symbols", [])
    if not backtest_cfg.get("dynamic_symbols_enabled", False):
        return symbols
    interval = data_cfg.get("interval")
    data_dir = Path(backtest_cfg.get("data_dir", "/data"))
    if not symbols:
        symbols = _symbols_from_data_dir(data_dir, interval)
    if not symbols:
        return symbols
    sim_cfg = _backtest_cfg_override(cfg)
    broker = _build_sim_broker(cfg, backtest_cfg)
    agent = TradingAgent(broker, sim_cfg)
    agent._symbols = symbols or []
    now = datetime.strptime(backtest_cfg["start"], "%Y-%m-%d")
    agent._refresh_news_cache(agent._symbols, now=now)
    portfolio = agent._get_portfolio_snapshot()
    agent._refresh_dynamic_symbols(portfolio, now=now)
    return agent._symbols


def _download_missing_bars(cfg: dict, symbols: list[str], interval: str, data_dir: Path) -> None:
    missing = []
    for symbol in symbols:
        path = data_dir / f"{symbol.replace('/', '_').replace('.', '_')}_{interval}.csv"
        if not path.exists():
            missing.append(symbol)
    if not missing:
        return
    alpaca_cfg = get_alpaca_account_cfg(cfg)
    api_key = alpaca_cfg.get("api_key", "")
    api_secret = alpaca_cfg.get("api_secret", "")
    backtest_cfg = cfg.get("backtest", {})
    download_alpaca_bars(
        missing,
        interval=interval,
        out_dir=str(data_dir),
        api_key=api_key,
        api_secret=api_secret,
        start=backtest_cfg.get("start", ""),
        end=backtest_cfg.get("end", ""),
        rate_limit_seconds=int(cfg.get("data", {}).get("rate_limit_seconds", 2)),
    )


class _SymbolState:
    def __init__(self, max_len: int):
        self.max_len = max_len
        self.prices = deque(maxlen=max_len)
        self.volumes = deque(maxlen=max_len)
        self.opens = deque(maxlen=max_len)
        self.highs = deque(maxlen=max_len)
        self.lows = deque(maxlen=max_len)
        self.session_prices = deque(maxlen=max_len)
        self.session_volumes = deque(maxlen=max_len)
        self.session_opens = deque(maxlen=max_len)
        self.session_highs = deque(maxlen=max_len)
        self.session_lows = deque(maxlen=max_len)
        self.session_open = None
        self.prev_close = None
        self.current_date = None

    def update(self, ts: datetime, row: pd.Series) -> None:
        price = float(row.get("Close", 0.0) or 0.0)
        date = ts.date()
        if self.current_date != date:
            if self.prices:
                self.prev_close = self.prices[-1]
            self.session_open = price
            self.current_date = date
            self.session_prices.clear()
            self.session_volumes.clear()
            self.session_opens.clear()
            self.session_highs.clear()
            self.session_lows.clear()
        self.prices.append(price)
        self.opens.append(float(row.get("Open", 0.0) or 0.0))
        self.highs.append(float(row.get("High", 0.0) or 0.0))
        self.lows.append(float(row.get("Low", 0.0) or 0.0))
        self.volumes.append(float(row.get("Volume", 0.0) or 0.0))
        self.session_prices.append(price)
        self.session_opens.append(float(row.get("Open", 0.0) or 0.0))
        self.session_highs.append(float(row.get("High", 0.0) or 0.0))
        self.session_lows.append(float(row.get("Low", 0.0) or 0.0))
        self.session_volumes.append(float(row.get("Volume", 0.0) or 0.0))

    def update_from_values(self, ts: datetime, values) -> None:
        price = float(values[3] or 0.0)
        date = ts.date()
        if self.current_date != date:
            if self.prices:
                self.prev_close = self.prices[-1]
            self.session_open = price
            self.current_date = date
            self.session_prices.clear()
            self.session_volumes.clear()
            self.session_opens.clear()
            self.session_highs.clear()
            self.session_lows.clear()
        self.prices.append(price)
        self.opens.append(float(values[0] or 0.0))
        self.highs.append(float(values[1] or 0.0))
        self.lows.append(float(values[2] or 0.0))
        self.volumes.append(float(values[4] or 0.0))
        self.session_prices.append(price)
        self.session_opens.append(float(values[0] or 0.0))
        self.session_highs.append(float(values[1] or 0.0))
        self.session_lows.append(float(values[2] or 0.0))
        self.session_volumes.append(float(values[4] or 0.0))

    def market_state(self) -> dict:
        prices = list(self.prices)
        volumes = list(self.volumes)
        session_prices = list(self.session_prices) if self.session_prices else prices
        session_volumes = list(self.session_volumes) if self.session_volumes else volumes
        session_opens = list(self.session_opens) if self.session_opens else list(self.opens)
        session_highs = list(self.session_highs) if self.session_highs else list(self.highs)
        session_lows = list(self.session_lows) if self.session_lows else list(self.lows)
        last_price = prices[-1] if prices else None
        rel_volume = 0.0
        if session_volumes:
            avg_volume = sum(session_volumes) / len(session_volumes)
            rel_volume = (session_volumes[-1] / avg_volume) if avg_volume else 0.0
        session_gain_pct = 0.0
        if self.session_open:
            session_gain_pct = (last_price - self.session_open) / self.session_open * 100.0 if last_price else 0.0
        elif self.prev_close:
            session_gain_pct = (last_price - self.prev_close) / self.prev_close * 100.0 if last_price else 0.0
        return {
            "prices": prices,
            "volumes": volumes,
            "session_prices": session_prices,
            "session_volumes": session_volumes,
            "opens": list(self.opens),
            "highs": list(self.highs),
            "lows": list(self.lows),
            "session_opens": session_opens,
            "session_highs": session_highs,
            "session_lows": session_lows,
            "session_bar_count": len(session_prices),
            "last_price": last_price,
            "session_volume": sum(session_volumes) if session_volumes else 0.0,
            "relative_volume": rel_volume,
            "session_gain_pct": session_gain_pct,
            "spread_pct": None,
        }
