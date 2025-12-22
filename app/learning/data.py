from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_csv_data(data_dir: str, interval: str | None = None) -> list[pd.DataFrame]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pattern = f"*_{interval}.csv" if interval else "*.csv"
    files = sorted(data_path.glob(pattern))
    if not files:
        files = sorted(data_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV data found in {data_dir}")

    datasets = []
    for file_path in files:
        df = pd.read_csv(file_path)
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "datetime"})
        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(df.columns):
            continue
        df = df.sort_values("datetime") if "datetime" in df.columns else df
        datasets.append(df)
    if not datasets:
        raise ValueError("No valid OHLCV CSVs found in data directory")
    return datasets
