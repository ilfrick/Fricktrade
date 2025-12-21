from prometheus_client import Gauge, Counter, start_http_server

TRADES = Counter("trades_total", "Total trades executed", ["symbol", "side"])
PNL = Gauge("pnl_percent", "Current PnL percent")
DRAWDOWN = Gauge("drawdown_percent", "Current drawdown percent")


def start_metrics_server(port: int):
    start_http_server(port)
