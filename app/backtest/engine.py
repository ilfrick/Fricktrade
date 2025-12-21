from datetime import datetime
from pathlib import Path
import backtrader as bt

try:
    import cupy as cp
except Exception:
    cp = None


class MomentumStrategy(bt.Strategy):
    params = dict(period=15)

    def __init__(self):
        self.sma = bt.ind.SMA(period=self.p.period)

    def next(self):
        if not self.position and self.data.close[0] > self.sma[0]:
            self.buy()
        elif self.position and self.data.close[0] < self.sma[0]:
            self.sell()


def _gpu_sma(series, period: int):
    if cp is None:
        return None
    arr = cp.asarray(series)
    if arr.size < period:
        return None
    cumsum = cp.cumsum(arr, dtype=cp.float64)
    cumsum[period:] = cumsum[period:] - cumsum[:-period]
    return cp.asnumpy(cumsum[period - 1 :] / period)


def run_backtest(
    data_dir: str,
    start: str,
    end: str,
    initial_cash: float,
    commission_pct: float,
    interval: str | None = None,
) -> dict:
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission_pct / 100.0)
    cerebro.addstrategy(MomentumStrategy)

    data_dir_path = Path(data_dir)
    pattern = f"*_{interval}.csv" if interval else "*.csv"
    data_path = next(data_dir_path.glob(pattern), None)
    if not data_path:
        data_path = next(data_dir_path.glob("*.csv"), None)
    if not data_path:
        raise FileNotFoundError(f"No CSV data found in {data_dir}")

    dtformat = "%Y-%m-%d %H:%M:%S"
    if interval and interval.endswith("d"):
        dtformat = "%Y-%m-%d"
    data = bt.feeds.GenericCSVData(
        dataname=str(data_path),
        fromdate=datetime.strptime(start, "%Y-%m-%d"),
        todate=datetime.strptime(end, "%Y-%m-%d"),
        dtformat=dtformat,
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1,
        timeframe=bt.TimeFrame.Minutes,
    )
    cerebro.adddata(data)

    start_value = cerebro.broker.getvalue()
    cerebro.run()
    end_value = cerebro.broker.getvalue()

    return {
        "start_value": start_value,
        "end_value": end_value,
        "return_pct": (end_value - start_value) / start_value * 100.0,
        "gpu_enabled": cp is not None,
    }
