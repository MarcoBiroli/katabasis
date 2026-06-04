"""End-to-end smoke: data pipeline + a few steps of every training phase.

Guards against integration regressions across the whole stack on CPU in
seconds. Not a quality check -- just that all four phases run and produce finite
metrics."""

from __future__ import annotations

import math

import torch

from katabasis.data import load_reactions
from katabasis.train import (
    get_device,
    joint_finetune,
    train_network_a,
    train_network_b,
)

MODEL = dict(
    embed_dim=8,
    n_layers=2,
    l_max=2,
    n_scalars=8,
    n_vectors=4,
    n_tensors=2,
    r_max=4.0,
    num_basis=8,
    radial_hidden=[16],
    time_dim=4,
)
SMOKE = dict(
    lr=2e-3,
    batch_size=4,
    epochs=1,
    seed=0,
    t_alpha=0.5,
    t_beta=1.5,
    noise_sigma=0.0,
    lambda_c=0.05,
    lambda_e=0.0,
    endpoint_start_frac=0.8,
    contractivity_gate=0.2,
    contractivity_iters=2,
    endpoint_steps=2,
    eval_steps=2,
    grad_clip=10.0,
    aux_enabled=True,
    aux_weight=0.1,
    aux_steps=2,
    joint_epoch_frac=1.0,
)
DATA = dict(
    subset="synthetic",
    n_reactions=8,
    split_seed=0,
    fractions=[0.5, 0.25, 0.25],
    filter=dict(
        min_barrier=5.0,
        min_rp_rmsd=0.3,
        monotonic_noise_frac=0.05,
        require_single_barrier=True,
        max_removed_frac=0.6,
    ),
)


def test_all_phases_run():
    dev = get_device("cpu")
    splits, _ = load_reactions(DATA)
    assert splits["train"] and splits["val"]
    cfg = dict(model=MODEL, train=SMOKE)

    net_b, sb = train_network_b(cfg, splits["train"], splits["val"], device=dev)
    assert math.isfinite(sb["val_endpoint_rmsd"])

    cfg2 = dict(model=MODEL, train={**SMOKE, "aux_enabled": False})
    _, s2 = train_network_a(cfg2, splits["train"], splits["val"], device=dev, net_b=None, phase=2)
    assert math.isfinite(s2["val_saddle_rmsd"])

    net_a, s4 = train_network_a(
        cfg, splits["train"], splits["val"], device=dev, net_b=net_b, phase=4
    )
    assert math.isfinite(s4["val_saddle_rmsd"])

    _, _, s5 = joint_finetune(
        cfg,
        net_a,
        net_b,
        splits["train"],
        splits["val"],
        device=dev,
        phase4_saddle_rmsd=s4["val_saddle_rmsd"],
    )
    assert "degraded_vs_phase4" in s5


def test_residual_init_predicts_near_midpoint():
    """At init, Network A's prediction should equal M (zero displacement)."""
    from katabasis.data import collate_reactions
    from katabasis.data.dataset import ReactionDataset
    from katabasis.data.synthetic import make_synthetic_reaction
    from katabasis.models import NetworkA

    r = make_synthetic_reaction("init", seed=0)
    net = NetworkA(**{k: MODEL[k] for k in MODEL if k != "time_dim"}).eval()
    b = collate_reactions([ReactionDataset([r])[0]])
    with torch.no_grad():
        pred = net(b.reactant, b.product, b.midpoint, b)
    assert float((pred - b.midpoint).abs().max()) < 1e-5
