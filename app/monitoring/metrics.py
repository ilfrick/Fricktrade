from prometheus_client import Gauge, Counter, start_http_server

TRADES = Counter("trades_total", "Total trades executed", ["symbol", "side"])
SKIPPED_ORDERS = Counter(
    "orders_skipped_total",
    "Orders skipped by safety checks",
    ["symbol", "side", "reason"],
)
PNL = Gauge("pnl_percent", "Current PnL percent")
DRAWDOWN = Gauge("drawdown_percent", "Current drawdown percent")
ACCOUNT_TOTAL = Gauge("account_total", "Account total equity")
ACCOUNT_CASH = Gauge("account_cash", "Account available cash")
ACCOUNT_INVESTED = Gauge("account_invested", "Account invested value")
SYMBOL_ACTIVE = Gauge("symbol_active", "Configured trading symbols", ["symbol"])
STRATEGY_ACTIVE = Gauge("strategy_active", "Configured trading strategies", ["strategy"])
OPEN_ORDERS = Gauge("open_orders", "Open orders", ["symbol", "side"])
BROKER_ACTIVE = Gauge("broker_active", "Active broker", ["broker"])


def start_metrics_server(port: int):
    start_http_server(port)
