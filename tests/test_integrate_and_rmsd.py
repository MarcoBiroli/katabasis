"""Flow integration and differentiable aligned RMSD."""

from __future__ import annotations

import pytest
import torch

from katabasis.data import collate_reactions
from katabasis.data.dataset import ReactionDataset
from katabasis.eval.integrate import integrate_flow, steepest_descent_on_forces
from katabasis.losses.rmsd import batched_aligned_rmsd, kabsch_rmsd_torch
from katabasis.models import NetworkB
from katabasis.utils import random_rotation


@pytest.fixture(autouse=True)
def _f64():
    torch.set_default_dtype(torch.float64)


def test_kabsch_rmsd_zero_for_rotated_copy(reaction, rng):
    p = torch.as_tensor(reaction.reactant)
    Q = torch.as_tensor(random_rotation(rng))
    q = p @ Q.T
    # The loss floors at ~1e-6 due to the sqrt epsilon that protects gradients.
    assert float(kabsch_rmsd_torch(p, q)) < 1e-5


def test_kabsch_rmsd_differentiable(reaction):
    p = torch.tensor(reaction.reactant, requires_grad=True)
    q = torch.as_tensor(reaction.product)
    loss = kabsch_rmsd_torch(p, q)
    loss.backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()


def test_batched_aligned_rmsd(reaction):
    ds = ReactionDataset([reaction, reaction])
    b = collate_reactions([ds[0], ds[1]])
    for k in ("midpoint", "saddle"):
        setattr(b, k, getattr(b, k).double())
    r = batched_aligned_rmsd(b.midpoint, b.saddle, b.ptr, b.z, b.groups)
    assert r > 0 and torch.isfinite(r)


def test_integrate_flow_shapes(reaction):
    net = (
        NetworkB(embed_dim=8, time_dim=4, n_layers=2, n_scalars=8, n_vectors=4, n_tensors=2)
        .double()
        .eval()
    )
    ds = ReactionDataset([reaction])
    b = collate_reactions([ds[0]])
    x0 = b.midpoint.double()
    target = b.product.double()
    final, traj = integrate_flow(net, x0, target, b, n_steps=8, return_trajectory=True)
    assert final.shape == x0.shape
    assert traj.shape == (9, *x0.shape)


def test_steepest_descent_baseline(reaction):
    saddle = torch.as_tensor(reaction.saddle)

    def forces(x):
        return -(x - torch.as_tensor(reaction.product))  # pull toward product

    out = steepest_descent_on_forces(forces, saddle, n_steps=200, step_size=0.05)
    # Should move closer to the product than the starting saddle.
    d_start = float((saddle - torch.as_tensor(reaction.product)).norm())
    d_end = float((out - torch.as_tensor(reaction.product)).norm())
    assert d_end < d_start
