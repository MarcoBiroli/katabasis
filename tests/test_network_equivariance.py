"""End-to-end SE(3) equivariance and R<->P symmetry of the networks.

Critical correctness requirements 2 and 6. Run in float64 so the tolerance
reflects true equivariance, not float32 accumulation.
"""

from __future__ import annotations

import pytest
import torch

from katabasis.data import collate_flow, collate_reactions
from katabasis.data.dataset import DescentFlowDataset, ReactionDataset
from katabasis.models import NetworkA, NetworkB
from katabasis.utils import random_rotation, random_translation

TOL = 1e-9


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _batch_a(reaction):
    ds = ReactionDataset([reaction])
    b = collate_reactions([ds[0]])
    for k in ("reactant", "product", "midpoint", "saddle"):
        setattr(b, k, getattr(b, k).double())
    return b


def _small_a():
    return NetworkA(embed_dim=8, n_layers=2, n_scalars=8, n_vectors=4, n_tensors=2).double().eval()


def test_network_a_rotation_translation_equivariance(reaction, rng):
    net = _small_a()
    b = _batch_a(reaction)
    Q = torch.as_tensor(random_rotation(rng))
    t = torch.as_tensor(random_translation(rng=rng))
    with torch.no_grad():
        pred = net(b.reactant, b.product, b.midpoint, b)
        pred2 = net(b.reactant @ Q.T + t, b.product @ Q.T + t, b.midpoint @ Q.T + t, b)
    assert float(((pred @ Q.T + t) - pred2).abs().max()) < TOL


def test_network_a_rp_symmetry(reaction):
    net = _small_a()
    b = _batch_a(reaction)
    with torch.no_grad():
        pred = net(b.reactant, b.product, b.midpoint, b)
        pred_swap = net(b.product, b.reactant, b.midpoint, b)
    assert float((pred - pred_swap).abs().max()) < TOL


def test_network_b_rotation_equivariance(reaction, rng):
    net = (
        NetworkB(embed_dim=8, time_dim=4, n_layers=2, n_scalars=8, n_vectors=4, n_tensors=2)
        .double()
        .eval()
    )
    ds = DescentFlowDataset([reaction])
    fb = collate_flow([ds[0], ds[1]])
    for k in ("x_t", "target", "dxdt", "unit_tangent", "t_node"):
        setattr(fb, k, getattr(fb, k).double())
    Q = torch.as_tensor(random_rotation(rng))
    t = torch.as_tensor(random_translation(rng=rng))
    with torch.no_grad():
        v = net(fb.x_t, fb.target, fb.t_node, fb)
        # velocity is a vector: rotates with Q, invariant to translation.
        v2 = net(fb.x_t @ Q.T + t, fb.target @ Q.T + t, fb.t_node, fb)
    assert float(((v @ Q.T) - v2).abs().max()) < TOL


def test_network_b_target_conditioning(reaction):
    """B must actually respond to the target (not ignore the conditioning)."""
    net = (
        NetworkB(embed_dim=8, time_dim=4, n_layers=2, n_scalars=8, n_vectors=4, n_tensors=2)
        .double()
        .eval()
    )
    ds = DescentFlowDataset([reaction])
    fb = collate_flow([ds[0]])
    for k in ("x_t", "target", "t_node"):
        setattr(fb, k, getattr(fb, k).double())
    with torch.no_grad():
        v_to_target = net(fb.x_t, fb.target, fb.t_node, fb)
        v_to_other = net(fb.x_t, fb.target + 1.0, fb.t_node, fb)
    assert float((v_to_target - v_to_other).abs().max()) > 1e-6
