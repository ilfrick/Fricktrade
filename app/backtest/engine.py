from datetime import datetime
import logging
from pathlib import Path
import backtrader as bt
import pandas as pd

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
    use_gpu: bool = True,
) -> dict:
    gpu_available = cp is not None
    gpu_enabled = bool(use_gpu) and gpu_available
    if use_gpu and not gpu_available:
        logging.warning("GPU requested for backtest but CuPy is unavailable.")
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

    dtformat = _infer_dtformat(data_path, interval)
    req_start = datetime.strptime(start, "%Y-%m-%d")
    req_end = datetime.strptime(end, "%Y-%m-%d")
    data_start, data_end, data_count = _scan_csv_range(data_path)
    if data_count == 0:
        raise ValueError(f"No rows found in CSV data at {data_path}")
    if req_end < data_start or req_start > data_end:
        logging.warning(
            "Backtest range %s-%s outside data range %s-%s; using data range.",
            req_start.date(),
            req_end.date(),
            data_start.date(),
            data_end.date(),
        )
        req_start, req_end = data_start, data_end
    else:
        if req_start < data_start:
            req_start = data_start
        if req_end > data_end:
            req_end = data_end
    if req_end < req_start:
        raise ValueError(
            f"Backtest date range invalid after clamping: {req_start.date()} > {req_end.date()}"
        )
    data = bt.feeds.GenericCSVData(
        dataname=str(data_path),
        fromdate=req_start,
        todate=req_end,
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
        "gpu_enabled": gpu_enabled,
    }


def _infer_dtformat(data_path: Path, interval: str | None) -> str:
    default_format = "%Y-%m-%d %H:%M:%S"
    if interval and interval.endswith("d"):
        default_format = "%Y-%m-%d"
    first_dt = None
    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            dt_field = line.split(",", 1)[0]
            if dt_field.lower() in {"date", "datetime", "timestamp"}:
                continue
            first_dt = dt_field
            break
    if not first_dt:
        return default_format
    if "T" in first_dt:
        first_dt = first_dt.replace("T", " ")
    if "+" in first_dt:
        return "%Y-%m-%d %H:%M:%S%z"
    if len(first_dt) == 10:
        return "%Y-%m-%d"
    return "%Y-%m-%d %H:%M:%S"


def _scan_csv_range(data_path: Path) -> tuple[datetime, datetime, int]:
    data = pd.read_csv(data_path, usecols=[0])
    if data.empty:
        return datetime.min, datetime.min, 0
    dates = pd.to_datetime(data.iloc[:, 0], utc=True, errors="coerce")
    dates = dates.dropna()
    if dates.empty:
        return datetime.min, datetime.min, 0
    dates = dates.dt.tz_convert(None)
    return dates.min().to_pydatetime(), dates.max().to_pydatetime(), int(dates.size)
