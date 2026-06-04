"""No saddle leakage at inference (Critical req. 4).

Network A's inference signature must consume only (R, P) -> M and never the
ground-truth saddle. We assert this structurally on the forward signature and
behaviorally by predicting without any saddle in scope.
"""

from __future__ import annotations

import inspect

import torch

from katabasis.data import collate_reactions
from katabasis.data.dataset import ReactionDataset
from katabasis.data.midpoint import midpoint
from katabasis.models import NetworkA


def test_network_a_forward_takes_no_saddle():
    params = set(inspect.signature(NetworkA.forward).parameters)
    assert "saddle" not in params, "Network A.forward must not accept a saddle argument"
    assert {"reactant", "product", "midpoint"} <= params


def test_inference_pathway_uses_only_rp(reaction):
    """Predict a saddle from R, P, M alone -- no ground-truth saddle referenced."""
    net = NetworkA(embed_dim=8, n_layers=2, n_scalars=8, n_vectors=4, n_tensors=2).eval()
    ds = ReactionDataset([reaction])
    b = collate_reactions([ds[0]])
    M = torch.as_tensor(midpoint(reaction.reactant, reaction.product), dtype=torch.float32)
    with torch.no_grad():
        pred = net(b.reactant, b.product, M, b)
    assert pred.shape == (reaction.n_atoms, 3)
