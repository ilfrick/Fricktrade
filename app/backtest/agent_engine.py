from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

import pandas as pd

from app.agents.trader import TradingAgent
from app.data.downloader import download_alpaca_bars
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


@dataclass
class BacktestPlanResult:
    runs: list[BacktestResult]
    average_return_pct: float
    total_trades: int


class SimBroker:
    def __init__(self, initial_cash: float, commission_pct: float):
        self.cash = float(initial_cash)
        self.commission_pct = float(commission_pct)
        self.positions: dict[str, float] = {}
        self.current_prices: dict[str, float] = {}
        self._order_id = 0
        self.trades = 0

    def get_account(self) -> dict:
        equity = self.cash + sum(self.positions.get(sym, 0.0) * self.current_prices.get(sym, 0.0) for sym in self.positions)
        return {"equity": equity, "cash": self.cash}

    def get_positions(self) -> list[dict]:
        results = []
        for symbol, qty in self.positions.items():
            price = self.current_prices.get(symbol, 0.0)
            results.append(
                {
                    "symbol": symbol,
                    "qty": qty,
                    "market_value": qty * price,
                    "current_price": price,
                }
            )
        return results

    def get_open_orders(self) -> list[dict]:
        return []

    def place_order(self, symbol: str, side: str, qty: float, order_type: str, **kwargs) -> str:
        price = self.current_prices.get(symbol)
        if price is None:
            raise ValueError(f"Missing price for {symbol}")
        cost = qty * price
        commission = cost * (self.commission_pct / 100.0)
        if side.lower() == "buy":
            self.cash -= cost + commission
            self.positions[symbol] = self.positions.get(symbol, 0.0) + qty
        else:
            self.cash += cost - commission
            self.positions[symbol] = self.positions.get(symbol, 0.0) - qty
        self._order_id += 1
        self.trades += 1
        return f"sim-{self._order_id}"

    def close_position(self, symbol: str) -> None:
        qty = self.positions.get(symbol, 0.0)
        if qty == 0.0:
            return
        side = "sell" if qty > 0 else "buy"
        self.place_order(symbol, side, abs(qty), "market")
        self.positions.pop(symbol, None)


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
    if symbols_source == "dynamic":
        symbols = _resolve_dynamic_symbols(cfg)
    if symbols_source == "data_dir":
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

    frames = {}
    for symbol in symbols:
        path = data_dir / f"{symbol.replace('.', '_')}_{interval}.csv"
        if not path.exists():
            continue
        frames[symbol] = _load_csv(path)
    if not frames:
        raise FileNotFoundError(f"No CSV data found for symbols in {data_dir}")

    timeline = _build_timeline(frames, start, end)
    sim_cfg = _backtest_cfg_override(cfg)
    broker = SimBroker(backtest_cfg["initial_cash"], backtest_cfg["commission_pct"])
    agent = TradingAgent(broker, sim_cfg)

    interval_minutes = _interval_minutes(interval or "1m")
    lookback_minutes = int(sim_cfg["strategy"]["params"].get("lookback_minutes", 30))
    lookback_bars = max(2, int(lookback_minutes / interval_minutes))

    state = {sym: _SymbolState(max_len=max(lookback_bars, 60)) for sym in frames}
    start_value = broker.get_account()["equity"]

    news_cache = _load_backtest_news(backtest_cfg)
    for ts in timeline:
        _apply_news_cache(agent, news_cache, ts)
        for symbol, frame in frames.items():
            if ts not in frame.index:
                continue
            row = frame.loc[ts]
            sym_state = state[symbol]
            sym_state.update(ts, row)
            market_state = sym_state.market_state()
            if market_state["last_price"] is None:
                continue
            broker.current_prices[symbol] = market_state["last_price"]
            portfolio = agent._get_portfolio_snapshot()
            agent._enrich_market_state(market_state, portfolio, symbol)
            market_state["strategy_symbols"] = getattr(agent, "_symbols_by_strategy", {})
            agent.run_once(symbol, market_state)

    end_value = broker.get_account()["equity"]
    return BacktestResult(
        start_value=start_value,
        end_value=end_value,
        return_pct=(end_value - start_value) / start_value * 100.0,
        trades=broker.trades,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        symbols=sorted(frames.keys()),
    )


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
        symbol = name[: -len(suffix)]
        if symbol:
            symbols.append(symbol.replace("_", "."))
    return sorted(set(symbols))


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[0])
    df.rename(columns={df.columns[0]: "Datetime"}, inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["Datetime"])
    df["Datetime"] = df["Datetime"].dt.tz_convert(None)
    df = df.set_index("Datetime").sort_index()
    return df


def _build_timeline(frames: dict[str, pd.DataFrame], start: datetime, end: datetime) -> list[datetime]:
    times = set()
    for frame in frames.values():
        subset = frame.loc[(frame.index >= start) & (frame.index <= end)]
        times.update(subset.index.to_pydatetime().tolist())
    return sorted(times)


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
    broker = SimBroker(backtest_cfg["initial_cash"], backtest_cfg["commission_pct"])
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
        path = data_dir / f"{symbol.replace('.', '_')}_{interval}.csv"
        if not path.exists():
            missing.append(symbol)
    if not missing:
        return
    alpaca_cfg = cfg.get("brokers", {}).get("alpaca", {})
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
        self.prices.append(price)
        self.opens.append(float(row.get("Open", 0.0) or 0.0))
        self.highs.append(float(row.get("High", 0.0) or 0.0))
        self.lows.append(float(row.get("Low", 0.0) or 0.0))
        self.volumes.append(float(row.get("Volume", 0.0) or 0.0))

    def market_state(self) -> dict:
        prices = list(self.prices)
        volumes = list(self.volumes)
        last_price = prices[-1] if prices else None
        rel_volume = 0.0
        if volumes:
            avg_volume = sum(volumes) / len(volumes)
            rel_volume = (volumes[-1] / avg_volume) if avg_volume else 0.0
        session_gain_pct = 0.0
        if self.session_open:
            session_gain_pct = (last_price - self.session_open) / self.session_open * 100.0 if last_price else 0.0
        elif self.prev_close:
            session_gain_pct = (last_price - self.prev_close) / self.prev_close * 100.0 if last_price else 0.0
        return {
            "prices": prices,
            "volumes": volumes,
            "opens": list(self.opens),
            "highs": list(self.highs),
            "lows": list(self.lows),
            "last_price": last_price,
            "session_volume": sum(volumes) if volumes else 0.0,
            "relative_volume": rel_volume,
            "session_gain_pct": session_gain_pct,
            "spread_pct": None,
        }
