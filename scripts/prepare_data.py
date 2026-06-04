#!/usr/bin/env python3
"""Build the per-reaction cache, apply Phase-1 filters, and persist splits.

Usage:
    python scripts/prepare_data.py data=t1x_core
    python scripts/prepare_data.py data=synthetic
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from katabasis.config import load_config, to_container
from katabasis.data import load_reactions


def main(argv: list[str]) -> int:
    cfg = load_config("config", overrides=argv)
    data_cfg = to_container(cfg)["data"]

    report_path = Path("runs") / f"filter_report_{data_cfg['subset']}.json"
    splits, report = load_reactions(data_cfg, report_path=report_path)

    print(f"subset={data_cfg['subset']}")
    print("filter report:", json.dumps(report.as_dict(), indent=2))
    print("split sizes:", {k: len(v) for k, v in splits.items()})
    if data_cfg.get("split_path"):
        print("split persisted to:", data_cfg["split_path"])
    print("filter report written to:", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
