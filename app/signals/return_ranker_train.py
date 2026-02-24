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
    max_age_days: int = 3,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    min_samples_leaf: int = 20,
    report: bool = True,
) -> bool:
    """Train and save the return ranker model.

    Returns True on success, False if there is insufficient data or sklearn
    is not available.  When ``report=True`` (default), writes a markdown
    training report next to the model file.
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        import joblib
    except ImportError:
        logger.error("sklearn / joblib not available — cannot train return ranker")
        return False

    X, y, file_stats = _load_training_data(data_dir, max_age_days=max_age_days)
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

    # Collect metrics for the report
    metrics: dict = {
        "n_rows": len(X),
        "n_features": X.shape[1],
        "n_files": file_stats["n_files"],
        "rows_per_file": file_stats.get("rows_per_file", {}),
        "hyperparams": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "min_samples_leaf": min_samples_leaf,
            "loss": "huber",
        },
        "label_stats": {
            "mean": float(np.mean(y)),
            "std": float(np.std(y)),
            "min": float(np.min(y)),
            "max": float(np.max(y)),
            "p5": float(np.percentile(y, 5)),
            "p25": float(np.percentile(y, 25)),
            "p50": float(np.percentile(y, 50)),
            "p75": float(np.percentile(y, 75)),
            "p95": float(np.percentile(y, 95)),
        },
    }

    # Feature summary statistics
    feat_stats = {}
    for i, name in enumerate(FEATURE_NAMES):
        col = X[:, i]
        feat_stats[name] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "zeros_pct": float(np.sum(col == 0) / len(col) * 100),
        }
    metrics["feature_stats"] = feat_stats

    # Time-ordered train/test split — evaluate BEFORE training on full data
    split = int(len(X) * 0.8)
    has_holdout = split >= MIN_ROWS // 2 and split < len(X)

    if has_holdout:
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        eval_model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            min_samples_leaf=min_samples_leaf,
            loss="huber",
            random_state=42,
        )
        eval_model.fit(X_train, y_train)
        y_pred = eval_model.predict(X_test)
        mae = float(np.mean(np.abs(y_pred - y_test)))
        corr = float(np.corrcoef(y_pred, y_test)[0, 1]) if len(y_test) > 2 else float("nan")
        rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
        median_ae = float(np.median(np.abs(y_pred - y_test)))
        # Directional accuracy: did we predict the sign correctly?
        sign_correct = float(np.mean(np.sign(y_pred) == np.sign(y_test))) if len(y_test) > 0 else float("nan")
        # Top-quintile precision: of the symbols the model ranked highest, how often were actual returns positive?
        top_k = max(len(y_test) // 5, 1)
        top_idx = np.argsort(y_pred)[-top_k:]
        top_quintile_precision = float(np.mean(y_test[top_idx] > 0)) if top_k > 0 else float("nan")
        # Bottom-quintile (worst predicted) — should have negative actual returns
        bottom_idx = np.argsort(y_pred)[:top_k]
        bottom_quintile_neg_rate = float(np.mean(y_test[bottom_idx] < 0)) if top_k > 0 else float("nan")

        metrics["holdout"] = {
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "mae": mae,
            "rmse": rmse,
            "median_ae": median_ae,
            "correlation": corr,
            "directional_accuracy": sign_correct,
            "top_quintile_precision": top_quintile_precision,
            "bottom_quintile_neg_rate": bottom_quintile_neg_rate,
        }
        logger.info("return_ranker holdout MAE=%.6f corr=%.4f dir_acc=%.2f%% n=%d",
                     mae, corr, sign_correct * 100, len(y_test))

    # Train final model on ALL data for deployment
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

    # Feature importances
    importances = model.feature_importances_
    ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda t: -t[1])
    metrics["feature_importances"] = ranked
    top = ", ".join(f"{n}={v:.3f}" for n, v in ranked[:10])
    logger.info("return_ranker feature importances (top 10): %s", top)

    # In-sample residual stats (sanity check, not for evaluation)
    y_pred_full = model.predict(X)
    residuals = y - y_pred_full
    metrics["in_sample"] = {
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "r_squared": float(1 - np.sum(residuals ** 2) / np.sum((y - np.mean(y)) ** 2)),
    }

    out_path = Path(model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    logger.info("return_ranker model saved to %s", out_path)

    if report:
        _write_training_report(out_path.parent, metrics)

    return True


def _date_from_filename(path: str) -> str:
    """Extract YYYY-MM-DD from return_ranker_YYYY-MM-DD.csv."""
    import re
    m = re.search(r"(\d{4}-\d{2}-\d{2})", Path(path).name)
    return m.group(1) if m else ""


def _load_training_data(data_dir: str, max_age_days: int = 0) -> tuple[np.ndarray, np.ndarray, dict] | tuple[None, None, dict]:
    """Read return_ranker_*.csv files, return (X, y, file_stats) numpy arrays.

    When *max_age_days* > 0, only files whose date suffix is within the
    last *max_age_days* days are included (rolling window).
    """
    import csv

    pattern = str(Path(data_dir) / "return_ranker_*.csv")
    files = sorted(glob.glob(pattern))

    if max_age_days > 0:
        from datetime import datetime as _dt, timedelta as _td
        cutoff = (_dt.now() - _td(days=max_age_days)).strftime("%Y-%m-%d")
        files = [f for f in files if _date_from_filename(f) >= cutoff]

    file_stats: dict = {"n_files": 0, "rows_per_file": {}}
    if not files:
        logger.warning("return_ranker: no training files found in %s", data_dir)
        return None, None, file_stats

    all_rows: list[list[float]] = []
    all_labels: list[float] = []
    rows_per_file: dict[str, int] = {}

    for fpath in files:
        try:
            rows, labels = _read_csv(fpath)
        except Exception as exc:
            logger.warning("return_ranker: skipping %s — %s", fpath, exc)
            continue
        rows_per_file[Path(fpath).name] = len(rows)
        if MAX_ROWS_PER_FILE and len(rows) > MAX_ROWS_PER_FILE:
            # Uniform subsample (preserve time order within file)
            idx = np.linspace(0, len(rows) - 1, MAX_ROWS_PER_FILE, dtype=int)
            rows = [rows[i] for i in idx]
            labels = [labels[i] for i in idx]
        all_rows.extend(rows)
        all_labels.extend(labels)
        logger.debug("return_ranker: loaded %d rows from %s", len(rows), fpath)

    file_stats = {"n_files": len(rows_per_file), "rows_per_file": rows_per_file}

    if not all_rows:
        return None, None, file_stats

    X = np.array(all_rows, dtype=np.float64)
    y = np.array(all_labels, dtype=np.float64)

    # Drop rows with any non-finite value
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    # Winsorise labels at 1st/99th percentile to remove fat-tail distortion
    p1, p99 = np.percentile(y, [1, 99])
    y = np.clip(y, p1, p99)

    logger.info("return_ranker: %d valid rows from %d files", len(X), len(files))
    return X, y, file_stats


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
# Training report
# ---------------------------------------------------------------------------

def _write_training_report(report_dir: Path, metrics: dict) -> None:
    """Write a markdown training report to *report_dir*/return_ranker_report.md."""
    from datetime import datetime, timezone as tz

    lines: list[str] = []
    now = datetime.now(tz.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines += [
        "# Return Ranker — Training Report",
        f"_Generated: {now}_",
        "",
        "## Dataset",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total rows | {metrics['n_rows']:,} |",
        f"| Features | {metrics['n_features']} |",
        f"| Training files | {metrics['n_files']} |",
        "",
    ]

    # Per-file row counts
    rpf = metrics.get("rows_per_file", {})
    if rpf:
        lines += ["**Rows per file:**", ""]
        lines += ["| File | Rows |", "|------|------|"]
        for fname, cnt in sorted(rpf.items()):
            lines.append(f"| {fname} | {cnt:,} |")
        lines.append("")

    # Label distribution
    ls = metrics.get("label_stats", {})
    if ls:
        lines += [
            "## Label Distribution (forward_ret_60m, log-return)",
            "",
            "| Stat | Value |",
            "|------|-------|",
        ]
        for key in ("mean", "std", "min", "p5", "p25", "p50", "p75", "p95", "max"):
            lines.append(f"| {key} | {ls[key]:.6f} |")
        lines.append("")

    # Hyperparameters
    hp = metrics.get("hyperparams", {})
    if hp:
        lines += ["## Hyperparameters", "", "| Param | Value |", "|-------|-------|"]
        for k, v in hp.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    # Holdout evaluation
    ho = metrics.get("holdout")
    if ho:
        lines += [
            "## Holdout Evaluation (time-ordered 80/20 split)",
            "",
            f"Train rows: **{ho['train_rows']:,}** | Test rows: **{ho['test_rows']:,}**",
            "",
            "| Metric | Value | Interpretation |",
            "|--------|-------|----------------|",
            f"| MAE | {ho['mae']:.6f} | Mean absolute prediction error |",
            f"| RMSE | {ho['rmse']:.6f} | Root mean squared error |",
            f"| Median AE | {ho['median_ae']:.6f} | Robust central error |",
            f"| Correlation | {ho['correlation']:.4f} | Predicted vs actual rank agreement |",
            f"| Directional accuracy | {ho['directional_accuracy']*100:.1f}% | Predicted correct sign of return |",
            f"| Top-quintile precision | {ho['top_quintile_precision']*100:.1f}% | % of top-predicted that had positive return |",
            f"| Bottom-quintile neg rate | {ho['bottom_quintile_neg_rate']*100:.1f}% | % of bottom-predicted that had negative return |",
            "",
        ]
        # Quality assessment
        corr = ho["correlation"]
        da = ho["directional_accuracy"]
        tqp = ho["top_quintile_precision"]
        lines.append("**Assessment:**")
        flags = []
        if corr > 0.15:
            flags.append(f"Correlation {corr:.4f} > 0.15 — meaningful predictive signal")
        elif corr > 0.05:
            flags.append(f"Correlation {corr:.4f} — weak but non-zero signal")
        else:
            flags.append(f"Correlation {corr:.4f} — very weak, model may not add value yet")
        if da > 0.55:
            flags.append(f"Directional accuracy {da*100:.1f}% > 55% — better than coin flip")
        else:
            flags.append(f"Directional accuracy {da*100:.1f}% — near random, needs more data")
        if tqp > 0.60:
            flags.append(f"Top-quintile precision {tqp*100:.1f}% — model picks winners above baseline")
        else:
            flags.append(f"Top-quintile precision {tqp*100:.1f}% — not yet reliably picking winners")
        for f in flags:
            lines.append(f"- {f}")
        lines.append("")
    else:
        lines += ["## Holdout Evaluation", "", "_Insufficient data for holdout split._", ""]

    # In-sample fit
    ins = metrics.get("in_sample")
    if ins:
        lines += [
            "## In-Sample Fit (sanity check only — not for evaluation)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| MAE | {ins['mae']:.6f} |",
            f"| RMSE | {ins['rmse']:.6f} |",
            f"| R-squared | {ins['r_squared']:.4f} |",
            "",
        ]

    # Feature importances
    fi = metrics.get("feature_importances", [])
    if fi:
        lines += [
            "## Feature Importances (final model)",
            "",
            "| Rank | Feature | Importance | Cumulative |",
            "|------|---------|------------|------------|",
        ]
        cumsum = 0.0
        for rank, (name, imp) in enumerate(fi, 1):
            cumsum += imp
            lines.append(f"| {rank} | {name} | {imp:.4f} | {cumsum:.4f} |")
        lines.append("")

    # Feature statistics
    fs = metrics.get("feature_stats", {})
    if fs:
        lines += [
            "## Feature Statistics",
            "",
            "| Feature | Mean | Std | Min | Max | Zeros% |",
            "|---------|------|-----|-----|-----|--------|",
        ]
        for name in FEATURE_NAMES:
            s = fs.get(name, {})
            lines.append(
                f"| {name} | {s.get('mean', 0):.4f} | {s.get('std', 0):.4f} | "
                f"{s.get('min', 0):.4f} | {s.get('max', 0):.4f} | {s.get('zeros_pct', 0):.1f}% |"
            )
        lines.append("")

    # Recommendations
    lines += ["## Recommendations", ""]
    if ho:
        if ho["correlation"] > 0.10 and ho["directional_accuracy"] > 0.53:
            lines.append("- Model shows useful signal. Consider enabling `return_ranker.enabled: true` in config for live A/B testing alongside Keras.")
        else:
            lines.append("- Model signal is weak. Collect more sessions of training data before enabling in production.")
        if ho["top_quintile_precision"] < 0.55:
            lines.append("- Top-quintile precision is low. Consider adding more features or increasing `n_estimators`.")
    lines.append("- Compare this report with previous reports to track model improvement over time.")
    lines.append("- Cross-reference with session analysis to verify model predictions match actual trading outcomes.")
    lines.append("")

    report_path = report_dir / "return_ranker_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("return_ranker training report written to %s", report_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Train return ranker model")
    p.add_argument("--data-dir", default="data/training", help="Directory with return_ranker_*.csv files")
    p.add_argument("--model-path", default="data/return_ranker.pkl", help="Output model path")
    p.add_argument("--max-age-days", type=int, default=3, help="Only use last N days of data (0 = all)")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.05)
    args = p.parse_args()

    ok = train_return_ranker(
        args.data_dir,
        args.model_path,
        max_age_days=args.max_age_days,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )
    sys.exit(0 if ok else 1)
