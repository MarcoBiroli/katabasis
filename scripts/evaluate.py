#!/usr/bin/env python3
"""Phase-6 evaluation: saddle RMSD + cycle-consistency confidence analysis.

Loads a trained Network A (and optionally B) and reports, on the test split:
saddle RMSD vs the zero-displacement trivial baseline, and -- if B is provided
-- the correlation between cycle-consistency error (descent endpoints vs R/P)
and the true saddle error. Positive correlation => a usable confidence score.

Examples
--------
    python scripts/evaluate.py data=t1x_core ckpt_a=runs/phase4/net_a.pt \
        ckpt_b=runs/phase3/net_b.pt
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from katabasis.config import load_config, to_container
from katabasis.data import load_reactions
from katabasis.data.batching import GraphBatch, build_batch_index
from katabasis.data.midpoint import midpoint
from katabasis.eval import (
    cycle_consistency_correlation,
    cycle_consistency_error,
    integrate_flow,
    saddle_rmsd,
)
from katabasis.train import get_device
from katabasis.train.common import build_network_a, build_network_b


def _graph(z_np, device):
    z = torch.as_tensor(z_np, dtype=torch.long, device=device)
    batch_idx, ptr = build_batch_index([len(z_np)])
    return GraphBatch(z=z, batch=batch_idx.to(device), ptr=ptr.to(device), n_graphs=1)


@torch.no_grad()
def main(argv: list[str]) -> int:
    cfg = to_container(load_config("config", overrides=argv))
    device = get_device(cfg.get("device", "auto"))
    splits, _ = load_reactions(cfg["data"])
    test = splits["test"]

    net_a = build_network_a(cfg["model"])
    net_a.load_state_dict(torch.load(cfg["ckpt_a"], map_location=device))
    net_a = net_a.to(device).eval()

    net_b = None
    if cfg.get("ckpt_b"):
        net_b = build_network_b(cfg["model"])
        net_b.load_state_dict(torch.load(cfg["ckpt_b"], map_location=device))
        net_b = net_b.to(device).eval()

    saddle_errs, trivial_errs, cycle_errs = [], [], []
    for r in test:
        g = _graph(r.atomic_numbers, device)
        R = torch.as_tensor(r.reactant, dtype=torch.float32, device=device)
        P = torch.as_tensor(r.product, dtype=torch.float32, device=device)
        M = torch.as_tensor(midpoint(r.reactant, r.product), dtype=torch.float32, device=device)
        pred = net_a(R, P, M, g)
        se = saddle_rmsd(pred.cpu().numpy(), r.saddle, r.atomic_numbers)
        saddle_errs.append(se)
        trivial_errs.append(saddle_rmsd(M.cpu().numpy(), r.saddle, r.atomic_numbers))
        if net_b is not None:
            end_r = integrate_flow(net_b, pred, R, g, n_steps=cfg["train"].get("eval_steps", 32))
            end_p = integrate_flow(net_b, pred, P, g, n_steps=cfg["train"].get("eval_steps", 32))
            cycle_errs.append(
                cycle_consistency_error(
                    end_r.cpu().numpy(),
                    end_p.cpu().numpy(),
                    r.reactant,
                    r.product,
                    r.atomic_numbers,
                )
            )

    print("=" * 60)
    print(f"Test reactions: {len(test)}  (subset={cfg['data']['subset']})")
    print(f"  saddle RMSD  : mean={np.mean(saddle_errs):.4f}  median={np.median(saddle_errs):.4f}")
    print(f"  trivial RMSD : mean={np.mean(trivial_errs):.4f}  (predict M)")
    if cycle_errs:
        stats = cycle_consistency_correlation(np.array(cycle_errs), np.array(saddle_errs))
        print(
            f"  cycle-consistency confidence: pearson={stats.pearson:.3f} "
            f"spearman={stats.spearman:.3f} (n={stats.n})"
        )
        print("  -> positive correlation means cycle error is a usable confidence score.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
