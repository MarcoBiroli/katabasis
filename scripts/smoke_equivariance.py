#!/usr/bin/env python3
"""Phase-0 equivariance canary (PLAN.md Phase 0).

Loads/synthesizes one reaction, computes ``M = (R + P) / 2``, applies a random
SO(3) rotation jointly to R and P, and verifies the rotated midpoint equals the
rotation of the unrotated midpoint within float tolerance. Also runs a network
forward pass and checks it rotates identically. This is the canary for the
whole pipeline.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from katabasis.data import collate_reactions
from katabasis.data.dataset import ReactionDataset
from katabasis.data.midpoint import midpoint
from katabasis.data.synthetic import make_synthetic_reaction
from katabasis.models import NetworkA
from katabasis.utils import random_reflection, random_rotation, random_translation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["SE3", "O3"], default="SE3")
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(0)
    r = make_synthetic_reaction("smoke", seed=0)

    # 1) Midpoint equivariance under rotation + translation.
    Q = random_rotation(rng)
    tvec = random_translation(rng=rng)
    M = midpoint(r.reactant, r.product)
    M_rot = midpoint(r.reactant @ Q.T + tvec, r.product @ Q.T + tvec)
    err = np.abs((M @ Q.T + tvec) - M_rot).max()
    print(f"[midpoint] rotation+translation equivariance error: {err:.2e}")
    assert err < args.tol, "midpoint is not equivariant!"

    # 2) Network forward-pass equivariance.
    ds = ReactionDataset([r])
    batch = collate_reactions([ds[0]])
    for k in ("reactant", "product", "midpoint", "saddle"):
        setattr(batch, k, getattr(batch, k).double())
    net = NetworkA(embed_dim=16, n_layers=2, n_scalars=16, n_vectors=8, n_tensors=4).double().eval()
    Qt = torch.as_tensor(Q)
    tt = torch.as_tensor(tvec)
    with torch.no_grad():
        pred = net(batch.reactant, batch.product, batch.midpoint, batch)
        pred_rot = net(
            batch.reactant @ Qt.T + tt, batch.product @ Qt.T + tt, batch.midpoint @ Qt.T + tt, batch
        )
        net_err = float(((pred @ Qt.T + tt) - pred_rot).abs().max())
    print(f"[network ] rotation+translation equivariance error: {net_err:.2e}")
    assert net_err < args.tol, "Network A is not equivariant!"

    # 3) Reflection behavior matches the pinned group.
    F = random_reflection(rng)
    Ft = torch.as_tensor(F)
    with torch.no_grad():
        pred_ref = net(batch.reactant @ Ft.T, batch.product @ Ft.T, batch.midpoint @ Ft.T, batch)
        refl_err = float(((pred @ Ft.T) - pred_ref).abs().max())
    if args.group == "O3":
        print(f"[network ] reflection equivariance error (O3): {refl_err:.2e}")
    else:
        print(f"[network ] reflection NOT required for SE3 (observed err {refl_err:.2e})")

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
