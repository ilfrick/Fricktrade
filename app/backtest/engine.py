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


def run_backtest(data_dir: str, start: str, end: str, initial_cash: float, commission_pct: float) -> dict:
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission_pct / 100.0)
    cerebro.addstrategy(MomentumStrategy)

    data_path = next(Path(data_dir).glob("*_1m.csv"), None)
    if not data_path:
        raise FileNotFoundError("No 1m CSV data found in data_dir")

    data = bt.feeds.GenericCSVData(
        dataname=str(data_path),
        fromdate=bt.date2num(datetime.strptime(start, "%Y-%m-%d")),
        todate=bt.date2num(datetime.strptime(end, "%Y-%m-%d")),
        dtformat="%Y-%m-%d %H:%M:%S",
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=6,
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
