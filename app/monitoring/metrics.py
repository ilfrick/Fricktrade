# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
except Exception:
    class _NoopMetric:
        def labels(self, **kwargs):
            return self

        def inc(self, amount: float = 1.0) -> None:
            return None

        def set(self, value: float) -> None:
            return None

        def observe(self, value: float) -> None:
            return None

        def remove(self, *args, **kwargs) -> None:
            return None

    def _noop_metric(*args, **kwargs):
        return _NoopMetric()

    Counter = Gauge = Histogram = _noop_metric

    def start_http_server(*args, **kwargs):
        return None

TRADES = Counter("trades_total", "Total trades executed", ["symbol", "side"])
TRADES_BY_BROKER = Counter(
    "trades_by_broker_total",
    "Total trades executed (per broker)",
    ["broker", "symbol", "side"],
)
SKIPPED_ORDERS = Counter(
    "orders_skipped_total",
    "Orders skipped by safety checks",
    ["symbol", "side", "reason"],
)
SKIPPED_ORDERS_BY_BROKER = Counter(
    "orders_skipped_by_broker_total",
    "Orders skipped by safety checks (per broker)",
    ["broker", "symbol", "side", "reason"],
)
ORDER_REJECTS = Counter(
    "order_rejects_total",
    "Orders rejected by broker",
    ["broker", "side", "reason"],
)
PDT_BLOCKS = Counter(
    "pdt_blocks_total",
    "Orders blocked by PDT protection",
    ["broker", "symbol", "side"],
)
PNL = Gauge("pnl_percent", "Current PnL percent")
DRAWDOWN = Gauge("drawdown_percent", "Current drawdown percent")
PNL_BY_BROKER = Gauge("pnl_percent_by_broker", "Current PnL percent (per broker)", ["broker"])
DRAWDOWN_BY_BROKER = Gauge("drawdown_percent_by_broker", "Current drawdown percent (per broker)", ["broker"])
ACCOUNT_TOTAL = Gauge("account_total", "Account total equity")
ACCOUNT_CASH = Gauge("account_cash", "Account available cash")
ACCOUNT_BUYING_POWER = Gauge("account_buying_power", "Account buying power")
ACCOUNT_INVESTED = Gauge("account_invested", "Account invested value")
ACCOUNT_TOTAL_BY_BROKER = Gauge("account_total_by_broker", "Account total equity", ["broker"])
ACCOUNT_CASH_BY_BROKER = Gauge("account_cash_by_broker", "Account available cash", ["broker"])
ACCOUNT_BUYING_POWER_BY_BROKER = Gauge("account_buying_power_by_broker", "Account buying power", ["broker"])
ACCOUNT_INVESTED_BY_BROKER = Gauge("account_invested_by_broker", "Account invested value", ["broker"])
SYMBOL_ACTIVE = Gauge("symbol_active", "Configured trading symbols", ["symbol"])
SYMBOL_ACTIVE_BY_BROKER = Gauge("symbol_active_by_broker", "Configured trading symbols by broker", ["broker", "symbol"])
STRATEGY_ACTIVE = Gauge("strategy_active", "Configured trading strategies", ["strategy"])
ORCHESTRATOR_STRATEGY_ACTIVE = Gauge(
    "orchestrator_strategy_active",
    "Strategies selected by orchestrator",
    ["symbol", "strategy"],
)
ORCHESTRATOR_STRATEGY_SELECTED = Counter(
    "orchestrator_strategy_selected_total",
    "Total orchestrator selections by strategy",
    ["strategy"],
)
STRATEGY_TRADES_REALIZED = Counter(
    "strategy_trades_realized_total",
    "Total realized trades by strategy",
    ["strategy"],
)
STRATEGY_WIN_RATE = Gauge(
    "strategy_win_rate",
    "Rolling win rate for strategy",
    ["strategy"],
)
STRATEGY_AVG_PNL_PCT = Gauge(
    "strategy_avg_pnl_pct",
    "Rolling average PnL percent for strategy",
    ["strategy"],
)
STRATEGY_DRAWDOWN_PCT = Gauge(
    "strategy_drawdown_pct",
    "Rolling drawdown percent for strategy",
    ["strategy"],
)
STRATEGY_DISABLED = Gauge(
    "strategy_disabled",
    "Strategy disabled by kill switch",
    ["strategy"],
)
STRATEGY_SHARPE_RATIO = Gauge(
    "strategy_sharpe_ratio",
    "Rolling 30-day Sharpe ratio for strategy",
    ["strategy", "asset_class"],
)
STRATEGY_PROFIT_FACTOR = Gauge(
    "strategy_profit_factor",
    "Rolling 30-day profit factor (gross_win / gross_loss) for strategy",
    ["strategy", "asset_class"],
)
TRADE_COST_BPS = Gauge(
    "trade_cost_bps",
    "Average effective trade cost in basis points",
    ["broker", "asset_class"],
)
OPEN_ORDERS = Gauge("open_orders", "Open orders", ["symbol", "side"])
OPEN_ORDERS_BY_BROKER = Gauge("open_orders_by_broker", "Open orders", ["broker", "symbol", "side"])
BROKER_ACTIVE = Gauge("broker_active", "Active broker", ["broker"])
BROKER_MARKET_OPEN = Gauge("broker_market_open", "Market open status for broker", ["broker"])
POSITION_QTY = Gauge("position_qty", "Position quantity", ["symbol"])
POSITION_VALUE = Gauge("position_value", "Position market value", ["symbol"])
POSITION_QTY_BY_BROKER = Gauge("position_qty_by_broker", "Position quantity", ["broker", "symbol"])
POSITION_VALUE_BY_BROKER = Gauge("position_value_by_broker", "Position market value", ["broker", "symbol"])
SIGNAL_RETURN_30M = Gauge("signal_return_30m_pct", "30m return percent", ["symbol"])
SIGNAL_RETURN_60M = Gauge("signal_return_60m_pct", "60m return percent", ["symbol"])
SIGNAL_EARLY_VOL = Gauge("signal_early_volume_pct", "Early volume percent", ["symbol"])
SIGNAL_RUNUP = Gauge("signal_runup_pct", "Runup percent", ["symbol"])
SIGNAL_DRAWDOWN = Gauge("signal_drawdown_pct", "Drawdown percent", ["symbol"])
SIGNAL_ABS_MOVE = Gauge("signal_abs_move", "Absolute move", ["symbol"])
SIGNAL_RUNUP_ABS = Gauge("signal_runup_abs", "Runup absolute", ["symbol"])
SIGNAL_DRAWDOWN_ABS = Gauge("signal_drawdown_abs", "Drawdown absolute", ["symbol"])
BROKER_REQUESTS = Counter(
    "broker_requests_total",
    "Broker API requests",
    ["broker", "method", "status"],
)
BROKER_LAST_SUCCESS = Gauge(
    "broker_last_success_timestamp_seconds",
    "Last successful broker API call",
    ["broker", "method"],
)
BROKER_LATENCY = Histogram(
    "broker_request_latency_seconds",
    "Broker API request latency",
    ["broker", "method"],
)
DECISION_LATENCY = Histogram(
    "decision_latency_seconds",
    "Time spent generating a decision per symbol",
    ["symbol"],
)
ORDER_LATENCY = Histogram(
    "order_enqueue_latency_seconds",
    "Time to enqueue an order for execution",
    ["symbol", "side"],
)
MARKET_CACHE_STALE_BARS = Counter(
    "market_cache_stale_bars_total",
    "Stale market cache bars used",
    ["interval", "source"],
)
MARKET_CACHE_STALE_FILTERED = Counter(
    "market_cache_stale_filtered_total",
    "Stale market cache filtered symbols used",
    ["interval", "source"],
)
PORTFOLIO_SCALE_FALLBACK = Counter(
    "portfolio_scale_fallback_total",
    "Portfolio position scale fallback to 1.0 on error",
    ["symbol"],
)
TAKE_PROFIT_EXITS = Counter(
    "take_profit_exits_total",
    "Take-profit exit events",
    ["symbol", "reason"],
)


def start_metrics_server(port: int):
    start_http_server(port)
