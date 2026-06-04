#!/usr/bin/env python3
"""Run a training phase from a composed config.

Examples
--------
    # Phase 3 (Network B) on t1x_core:
    python scripts/run.py data=t1x_core model=default train=phase3

    # Phase 2 direct baseline:
    python scripts/run.py data=t1x_core train=phase2

    # Phase 4 (needs a trained B checkpoint):
    python scripts/run.py train=phase4 ckpt_b=runs/phase3/net_b.pt

    # Tiny smoke run on synthetic data (no download):
    python scripts/run.py data=synthetic model=small train=smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from katabasis.config import load_config, to_container
from katabasis.data import load_reactions
from katabasis.train import (
    RunLogger,
    get_device,
    joint_finetune,
    train_network_a,
    train_network_b,
)
from katabasis.train.common import build_network_a, build_network_b
from katabasis.utils import seed_everything


def main(argv: list[str]) -> int:
    cfg_dc = load_config("config", overrides=argv)
    cfg = to_container(cfg_dc)
    seed_everything(cfg.get("seed", 0))
    device = get_device(cfg.get("device", "auto"))
    run_dir = Path(cfg.get("run_dir", "runs/default"))
    phase = cfg["train"]["phase"]

    splits, report = load_reactions(cfg["data"], report_path=run_dir / "filter_report.json")
    print(
        f"phase={phase} subset={cfg['data']['subset']} "
        f"sizes={ {k: len(v) for k, v in splits.items()} }"
    )

    with RunLogger(run_dir, config=cfg, use_wandb=cfg.get("wandb", False)) as logger:
        if phase == 3:
            net_b, summary = train_network_b(
                cfg, splits["train"], splits["val"], device=device, logger=logger
            )
            torch.save(net_b.state_dict(), run_dir / "net_b.pt")
        elif phase in (2, 4):
            net_b = None
            if phase == 4:
                net_b = build_network_b(cfg["model"])
                net_b.load_state_dict(torch.load(cfg["ckpt_b"], map_location=device))
            net_a, summary = train_network_a(
                cfg,
                splits["train"],
                splits["val"],
                device=device,
                net_b=net_b,
                logger=logger,
                phase=phase,
            )
            torch.save(net_a.state_dict(), run_dir / "net_a.pt")
        elif phase == 5:
            net_a = build_network_a(cfg["model"])
            net_a.load_state_dict(torch.load(cfg["ckpt_a"], map_location=device))
            net_b = build_network_b(cfg["model"])
            net_b.load_state_dict(torch.load(cfg["ckpt_b"], map_location=device))
            net_a, net_b, summary = joint_finetune(
                cfg,
                net_a,
                net_b,
                splits["train"],
                splits["val"],
                device=device,
                phase4_saddle_rmsd=cfg.get("phase4_saddle_rmsd", float("inf")),
                logger=logger,
            )
            torch.save(net_a.state_dict(), run_dir / "net_a_joint.pt")
            torch.save(net_b.state_dict(), run_dir / "net_b_joint.pt")
        else:
            raise SystemExit(f"unknown phase {phase}; set train=phase2|phase3|phase4|phase5")

    print("summary:", {k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
