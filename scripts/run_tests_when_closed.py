# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import argparse
import logging
import subprocess
import time

from app.utils.config import load_config
from app.utils.market import is_market_open
from app.utils.ops_state import load_ops_state, ops_state_is_running, ops_state_is_sleeping


def _run_pytest() -> int:
    result = subprocess.run(
        ["pytest", "-q", "--disable-warnings", "--maxfail=1"],
        check=False,
    )
    return result.returncode


def _run_backtest(config_path: str) -> int:
    result = subprocess.run(
        ["python", "-m", "app.main", "backtest", "--config", config_path],
        check=False,
    )
    return result.returncode


def _run_benchmarks(cfg: dict, config_path: str) -> int:
    bench_cfg = cfg.get("benchmarking", {}) or {}
    args = [
        "python",
        "scripts/benchmark_runner.py",
        "--config",
        config_path,
        "--output",
        str(bench_cfg.get("output_path", "/data/reports/benchmark_report.json")),
    ]
    if bench_cfg.get("use_plan", True):
        args.append("--use-plan")
        args.extend(["--plan-window-days", str(bench_cfg.get("plan_window_days", 60))])
        args.extend(["--plan-step-days", str(bench_cfg.get("plan_step_days", 30))])
        args.extend(["--plan-liquidity-tiers", str(bench_cfg.get("plan_liquidity_tiers", 3))])
        args.extend(["--plan-sample-per-tier", str(bench_cfg.get("plan_sample_per_tier", 10))])
    plot_dir = bench_cfg.get("plot_dir")
    if plot_dir:
        args.extend(["--plot-dir", str(plot_dir)])
    pdf_path = bench_cfg.get("pdf_path")
    if pdf_path:
        args.extend(["--pdf-path", str(pdf_path)])
    result = subprocess.run(args, check=False)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/app/config/config.yaml")
    parser.add_argument("--interval-minutes", type=int, default=15)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    interval = max(args.interval_minutes, 1)

    while True:
        cfg = load_config(args.config)
        ms_cfg = cfg.get("healthwatch", {}).get("market_shutdown", {}) or {}
        if ms_cfg.get("write_state", False):
            ops_state = load_ops_state(ms_cfg.get("state_path", "/data/system_state.json"))
            if ops_state_is_running(ops_state):
                logging.info("Ops state running; skipping tests.")
                if args.once:
                    break
                time.sleep(interval * 60)
                continue
            if ops_state_is_sleeping(ops_state):
                logging.info("Ops state sleeping; running tests.")
            else:
                logging.info("Ops state unknown; falling back to market check.")
        if is_market_open(cfg):
            logging.info("Market open; skipping tests.")
        else:
            logging.info("Market closed; running tests.")
            code = _run_pytest()
            logging.info("Pytest finished with exit code %d.", code)
            backtest_cfg = cfg.get("backtest", {})
            if backtest_cfg.get("run_when_closed", False):
                logging.info("Market closed; running backtest.")
                backtest_code = _run_backtest(args.config)
                logging.info("Backtest finished with exit code %d.", backtest_code)
            bench_cfg = cfg.get("benchmarking", {})
            if bench_cfg.get("run_when_closed", False):
                logging.info("Market closed; running benchmarks.")
                bench_code = _run_benchmarks(cfg, args.config)
                logging.info("Benchmarks finished with exit code %d.", bench_code)
        if args.once:
            break
        time.sleep(interval * 60)


if __name__ == "__main__":
    main()
