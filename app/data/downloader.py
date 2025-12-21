from pathlib import Path
import yfinance as yf


def download_yfinance(symbols: list[str], interval: str, lookback_days: int, out_dir: str) -> list[Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    files = []
    period = f"{lookback_days}d"
    for symbol in symbols:
        data = yf.download(tickers=symbol, period=period, interval=interval, auto_adjust=True, progress=False)
        if data.empty:
            continue
        file_path = out_path / f"{symbol.replace('.', '_')}_{interval}.csv"
        data.to_csv(file_path)
        files.append(file_path)
    return files
