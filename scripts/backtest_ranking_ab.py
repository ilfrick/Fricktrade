# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""A/B backtest: entry ranking enabled vs disabled."""
from __future__ import annotations

import copy
import json
import logging
import sys
import time
from pathlib import Path

from app.backtest.agent_engine import run_agent_backtest
from app.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

PERIODS = [
    ("crash", "2026-01-28", "2026-02-09"),
    ("recovery", "2026-02-21", "2026-03-03"),
]


def _run_one(cfg: dict, label: str, start: str, end: str) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["backtest"]["start"] = start
    cfg["backtest"]["end"] = end
    cfg["backtest"].pop("walk_forward", None)
    t0 = time.time()
    result = run_agent_backtest(cfg)
    elapsed = time.time() - t0
    r = getattr(result, "__dict__", {})
    return {
        "label": label,
        "start": start,
        "end": end,
        "return_pct": round(getattr(result, "return_pct", r.get("return_pct", 0)), 4),
        "trades": getattr(result, "trades", r.get("trades", 0)),
        "elapsed": round(elapsed, 1),
        "trade_details": getattr(result, "trade_details", []),
    }


def _stats(details: list) -> dict:
    sells = [t for t in details if t.get("side") == "sell"]
    if not sells:
        return {"sells": 0, "wins": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0}
    wins = [t for t in sells if (t.get("pnl_pct") or 0) > 0]
    losses = [t for t in sells if (t.get("pnl_pct") or 0) <= 0]
    return {
        "sells": len(sells),
        "wins": len(wins),
        "win_rate": round(len(wins) / len(sells) * 100, 1) if sells else 0,
        "avg_win": round(sum(t.get("pnl_pct", 0) for t in wins) / len(wins), 3) if wins else 0,
        "avg_loss": round(sum(t.get("pnl_pct", 0) for t in losses) / len(losses), 3) if losses else 0,
    }


def main():
    cfg = load_config("config/config.yaml")

    results = []
    for period_name, start, end in PERIODS:
        # A: ranking enabled
        cfg_a = copy.deepcopy(cfg)
        cfg_a.setdefault("strategy", {}).setdefault("entry_ranking", {})["enabled"] = True
        r_a = _run_one(cfg_a, f"ranking_ON_{period_name}", start, end)
        s_a = _stats(r_a.pop("trade_details", []))
        r_a.update(s_a)
        results.append(r_a)
        print(f"  {r_a['label']}: ret={r_a['return_pct']:+.2f}% trades={r_a['trades']} "
              f"sells={s_a['sells']} wr={s_a['win_rate']:.1f}% ({r_a['elapsed']:.0f}s)")

        # B: ranking disabled
        cfg_b = copy.deepcopy(cfg)
        cfg_b.setdefault("strategy", {}).setdefault("entry_ranking", {})["enabled"] = False
        r_b = _run_one(cfg_b, f"ranking_OFF_{period_name}", start, end)
        s_b = _stats(r_b.pop("trade_details", []))
        r_b.update(s_b)
        results.append(r_b)
        print(f"  {r_b['label']}: ret={r_b['return_pct']:+.2f}% trades={r_b['trades']} "
              f"sells={s_b['sells']} wr={s_b['win_rate']:.1f}% ({r_b['elapsed']:.0f}s)")

    print("\n=== A/B SUMMARY ===")
    for r in results:
        print(f"  {r['label']:>30}: ret={r['return_pct']:+.3f}% trades={r['trades']} "
              f"sells={r['sells']} wins={r['wins']} wr={r['win_rate']:.1f}%")

    out = Path("/data/reports/ranking_ab_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
