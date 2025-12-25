import argparse
import logging
import subprocess
import time

from app.utils.config import load_config
from app.utils.market import is_market_open


def _run_pytest() -> int:
    result = subprocess.run(
        ["pytest", "-q", "--disable-warnings", "--maxfail=1"],
        check=False,
    )
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
        if is_market_open(cfg):
            logging.info("Market open; skipping tests.")
        else:
            logging.info("Market closed; running tests.")
            code = _run_pytest()
            logging.info("Pytest finished with exit code %d.", code)
        if args.once:
            break
        time.sleep(interval * 60)


if __name__ == "__main__":
    main()
