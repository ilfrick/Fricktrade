# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""Training module for the return ranker model.

Reads all CSV files from data_dir (written by scripts/collect_return_ranker_data.py),
trains a sklearn GradientBoostingRegressor to predict 60-minute forward log-return,
and saves the fitted model as a joblib pickle.

Can be called standalone or imported by the AI filter for auto-retraining.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from app.signals.return_ranker import FEATURE_NAMES

logger = logging.getLogger(__name__)

# Minimum rows required to attempt training
MIN_ROWS = 200
# Rows sampled per CSV file to cap memory usage (0 = all)
MAX_ROWS_PER_FILE = 50_000


def train_return_ranker(
    data_dir: str,
    model_path: str,
    *,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    min_samples_leaf: int = 20,
) -> bool:
    """Train and save the return ranker model.

    Returns True on success, False if there is insufficient data or sklearn
    is not available.
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        import joblib
    except ImportError:
        logger.error("sklearn / joblib not available — cannot train return ranker")
        return False

    X, y = _load_training_data(data_dir)
    if X is None or len(X) < MIN_ROWS:
        logger.warning(
            "return_ranker: insufficient training data (%d rows, need %d)",
            0 if X is None else len(X),
            MIN_ROWS,
        )
        return False

    logger.info(
        "return_ranker: training on %d rows, %d features; n_estimators=%d depth=%d",
        len(X),
        X.shape[1],
        n_estimators,
        max_depth,
    )

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        min_samples_leaf=min_samples_leaf,
        loss="huber",          # robust to return outliers
        random_state=42,
    )
    model.fit(X, y)

    # Log feature importances for interpretability
    importances = model.feature_importances_
    ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda t: -t[1])
    top = ", ".join(f"{n}={v:.3f}" for n, v in ranked[:10])
    logger.info("return_ranker feature importances (top 10): %s", top)

    # Holdout evaluation (last 20% of data, time-ordered)
    split = int(len(X) * 0.8)
    if split < MIN_ROWS // 2:
        split = len(X)
    if split < len(X):
        y_pred = model.predict(X[split:])
        y_true = y[split:]
        mae = float(np.mean(np.abs(y_pred - y_true)))
        corr = float(np.corrcoef(y_pred, y_true)[0, 1]) if len(y_true) > 2 else float("nan")
        logger.info("return_ranker holdout MAE=%.6f corr=%.4f n=%d", mae, corr, len(y_true))

    out_path = Path(model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    logger.info("return_ranker model saved to %s", out_path)
    return True


def _load_training_data(data_dir: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Read all return_ranker_*.csv files, return (X, y) numpy arrays."""
    import csv

    pattern = str(Path(data_dir) / "return_ranker_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        logger.warning("return_ranker: no training files found in %s", data_dir)
        return None, None

    all_rows: list[list[float]] = []
    all_labels: list[float] = []

    for fpath in files:
        try:
            rows, labels = _read_csv(fpath)
        except Exception as exc:
            logger.warning("return_ranker: skipping %s — %s", fpath, exc)
            continue
        if MAX_ROWS_PER_FILE and len(rows) > MAX_ROWS_PER_FILE:
            # Uniform subsample (preserve time order within file)
            idx = np.linspace(0, len(rows) - 1, MAX_ROWS_PER_FILE, dtype=int)
            rows = [rows[i] for i in idx]
            labels = [labels[i] for i in idx]
        all_rows.extend(rows)
        all_labels.extend(labels)
        logger.debug("return_ranker: loaded %d rows from %s", len(rows), fpath)

    if not all_rows:
        return None, None

    X = np.array(all_rows, dtype=np.float64)
    y = np.array(all_labels, dtype=np.float64)

    # Drop rows with any non-finite value
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    # Winsorise labels at 1st/99th percentile to remove fat-tail distortion
    p1, p99 = np.percentile(y, [1, 99])
    y = np.clip(y, p1, p99)

    logger.info("return_ranker: %d valid rows from %d files", len(X), len(files))
    return X, y


def _read_csv(fpath: str) -> tuple[list[list[float]], list[float]]:
    import csv

    rows: list[list[float]] = []
    labels: list[float] = []
    with open(fpath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            try:
                label = float(rec["forward_ret_60m"])
            except (KeyError, ValueError):
                continue
            feat = []
            valid = True
            for col in FEATURE_NAMES:
                v = rec.get(col, "")
                if v == "" or v is None:
                    valid = False
                    break
                try:
                    feat.append(float(v))
                except ValueError:
                    valid = False
                    break
            if valid:
                rows.append(feat)
                labels.append(label)
    return rows, labels


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Train return ranker model")
    p.add_argument("--data-dir", default="data/training", help="Directory with return_ranker_*.csv files")
    p.add_argument("--model-path", default="data/return_ranker.pkl", help="Output model path")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.05)
    args = p.parse_args()

    ok = train_return_ranker(
        args.data_dir,
        args.model_path,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )
    sys.exit(0 if ok else 1)
