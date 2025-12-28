from prometheus_client import Counter, Gauge, Histogram, start_http_server

TRADES = Counter("trades_total", "Total trades executed", ["symbol", "side"])
SKIPPED_ORDERS = Counter(
    "orders_skipped_total",
    "Orders skipped by safety checks",
    ["symbol", "side", "reason"],
)
ORDER_REJECTS = Counter(
    "order_rejects_total",
    "Orders rejected by broker",
    ["broker", "symbol", "side", "code", "reason"],
)
PNL = Gauge("pnl_percent", "Current PnL percent")
DRAWDOWN = Gauge("drawdown_percent", "Current drawdown percent")
ACCOUNT_TOTAL = Gauge("account_total", "Account total equity")
ACCOUNT_CASH = Gauge("account_cash", "Account available cash")
ACCOUNT_INVESTED = Gauge("account_invested", "Account invested value")
SYMBOL_ACTIVE = Gauge("symbol_active", "Configured trading symbols", ["symbol"])
STRATEGY_ACTIVE = Gauge("strategy_active", "Configured trading strategies", ["strategy"])
ORCHESTRATOR_STRATEGY_ACTIVE = Gauge(
    "orchestrator_strategy_active",
    "Strategies selected by orchestrator",
    ["symbol", "strategy"],
)
OPEN_ORDERS = Gauge("open_orders", "Open orders", ["symbol", "side"])
BROKER_ACTIVE = Gauge("broker_active", "Active broker", ["broker"])
BROKER_MARKET_OPEN = Gauge("broker_market_open", "Market open status for broker", ["broker"])
POSITION_QTY = Gauge("position_qty", "Position quantity", ["symbol"])
POSITION_VALUE = Gauge("position_value", "Position market value", ["symbol"])
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


def start_metrics_server(port: int):
    start_http_server(port)
