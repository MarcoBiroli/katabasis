"""The phase training loops (PLAN.md Phases 2-5).

- :func:`train_network_b`         Phase 3: descent flow in isolation.
- :func:`train_network_a`         Phase 2 (direct baseline) and Phase 4 (with
                                  frozen B auxiliary loss), selected by config.
- :func:`joint_finetune`          Phase 5: unfreeze both, LR / 10, short run.

Each loop logs per-epoch validation metrics and returns the model plus a
summary dict. They run on the synthetic dataset for CPU smoke tests and on
Halo8 for real runs -- the only difference is the reaction list passed in.
"""

from __future__ import annotations

import numpy as np
import torch

from katabasis.data.types import Reaction
from katabasis.eval.integrate import integrate_flow
from katabasis.eval.metrics import saddle_rmsd
from katabasis.models.network_a import NetworkA
from katabasis.models.network_b import NetworkB
from katabasis.train.common import (
    AuxConfig,
    build_network_a,
    build_network_b,
    flow_loader,
    network_a_step,
    network_b_step,
    reaction_loader,
)
from katabasis.train.logging import RunLogger


# --------------------------------------------------------------------------- #
# Phase 3: Network B
# --------------------------------------------------------------------------- #
def train_network_b(
    cfg: dict,
    train_reactions: list[Reaction],
    val_reactions: list[Reaction],
    *,
    device: torch.device,
    logger: RunLogger | None = None,
) -> tuple[NetworkB, dict]:
    tr = cfg["train"]
    net = build_network_b(cfg["model"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=tr["lr"])
    loader = flow_loader(
        train_reactions,
        tr["batch_size"],
        shuffle=True,
        t_alpha=tr["t_alpha"],
        t_beta=tr["t_beta"],
        noise_sigma=tr["noise_sigma"],
        seed=tr["seed"],
    )
    epochs = tr["epochs"]
    summary = {}
    for epoch in range(epochs):
        net.train()
        # Endpoint correction turns on only in the late phase (~80% through).
        lambda_e = tr["lambda_e"] if epoch >= int(tr["endpoint_start_frac"] * epochs) else 0.0
        running = 0.0
        n = 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = network_b_step(
                net,
                batch,
                lambda_c=tr["lambda_c"],
                lambda_e=lambda_e,
                contractivity_gate=tr["contractivity_gate"],
                contractivity_iters=tr["contractivity_iters"],
                endpoint_steps=tr["endpoint_steps"],
            )
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), tr["grad_clip"])
            opt.step()
            running += float(out["loss"].detach()) * batch.n_graphs
            n += batch.n_graphs
        val = evaluate_network_b(net, val_reactions, device, n_steps=tr["eval_steps"])
        if logger:
            logger.log({"phase": 3, "train_loss": running / max(n, 1), **val}, step=epoch)
        summary = {"train_loss": running / max(n, 1), **val}
    return net, summary


@torch.no_grad()
def evaluate_network_b(
    net: NetworkB, reactions: list[Reaction], device: torch.device, *, n_steps: int = 32
) -> dict:
    """Endpoint RMSD + path fidelity by integrating B from the true saddle."""
    from katabasis.data.arclength import build_legs
    from katabasis.data.batching import GraphBatch, build_batch_index
    from katabasis.eval.metrics import path_fidelity

    net.eval()
    endpoint_errs, path_errs = [], []
    for r in reactions:
        legs = build_legs(r)
        for leg in legs:
            z = torch.as_tensor(r.atomic_numbers, dtype=torch.long, device=device)
            batch_idx, ptr = build_batch_index([r.n_atoms])
            graph = GraphBatch(z=z, batch=batch_idx.to(device), ptr=ptr.to(device), n_graphs=1)
            x0 = torch.as_tensor(leg.positions[0], dtype=torch.float32, device=device)
            target = torch.as_tensor(leg.endpoint, dtype=torch.float32, device=device)
            final, traj = integrate_flow(
                net, x0, target, graph, n_steps=n_steps, return_trajectory=True
            )
            err = saddle_rmsd(final.cpu().numpy(), leg.endpoint, r.atomic_numbers)
            endpoint_errs.append(err)
            pf = path_fidelity(traj.cpu().numpy(), leg, r.atomic_numbers)
            path_errs.append(pf["path_rmsd_mean"])
    return {
        "val_endpoint_rmsd": float(np.mean(endpoint_errs)) if endpoint_errs else float("nan"),
        "val_path_rmsd": float(np.mean(path_errs)) if path_errs else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Phase 2 (direct baseline) and Phase 4 (with frozen B)
# --------------------------------------------------------------------------- #
def train_network_a(
    cfg: dict,
    train_reactions: list[Reaction],
    val_reactions: list[Reaction],
    *,
    device: torch.device,
    net_b: NetworkB | None = None,
    logger: RunLogger | None = None,
    phase: int = 4,
) -> tuple[NetworkA, dict]:
    tr = cfg["train"]
    net = build_network_a(cfg["model"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=tr["lr"])
    aux = AuxConfig(
        enabled=bool(tr.get("aux_enabled", False)) and net_b is not None,
        weight=tr.get("aux_weight", 0.1),
        n_steps=tr.get("aux_steps", 16),
    )
    if net_b is not None:
        net_b = net_b.to(device).eval()
        for p in net_b.parameters():
            p.requires_grad_(False)

    loader = reaction_loader(train_reactions, tr["batch_size"], shuffle=True)
    epochs = tr["epochs"]
    summary = {}
    for epoch in range(epochs):
        net.train()
        running = 0.0
        n = 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = network_a_step(net, batch, net_b=net_b, aux=aux)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), tr["grad_clip"])
            opt.step()
            running += float(out["primary"].detach()) * batch.n_graphs
            n += batch.n_graphs
        val = evaluate_network_a(net, val_reactions, device)
        if logger:
            logger.log(
                {"phase": phase, "train_saddle_rmsd": running / max(n, 1), **val}, step=epoch
            )
        summary = {"train_saddle_rmsd": running / max(n, 1), **val}
    return net, summary


@torch.no_grad()
def evaluate_network_a(net: NetworkA, reactions: list[Reaction], device: torch.device) -> dict:
    """Mean connectivity-constrained saddle RMSD on a held-out set."""
    from katabasis.data.batching import GraphBatch, build_batch_index
    from katabasis.data.midpoint import midpoint

    net.eval()
    errs = []
    trivial_errs = []  # zero-displacement (predict M) trivial baseline
    for r in reactions:
        z = torch.as_tensor(r.atomic_numbers, dtype=torch.long, device=device)
        batch_idx, ptr = build_batch_index([r.n_atoms])
        graph = GraphBatch(z=z, batch=batch_idx.to(device), ptr=ptr.to(device), n_graphs=1)
        R = torch.as_tensor(r.reactant, dtype=torch.float32, device=device)
        P = torch.as_tensor(r.product, dtype=torch.float32, device=device)
        M = torch.as_tensor(midpoint(r.reactant, r.product), dtype=torch.float32, device=device)
        pred = net(R, P, M, graph).cpu().numpy()
        errs.append(saddle_rmsd(pred, r.saddle, r.atomic_numbers))
        trivial_errs.append(saddle_rmsd(M.cpu().numpy(), r.saddle, r.atomic_numbers))
    return {
        "val_saddle_rmsd": float(np.mean(errs)) if errs else float("nan"),
        "val_trivial_rmsd": float(np.mean(trivial_errs)) if trivial_errs else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Phase 5: joint fine-tuning
# --------------------------------------------------------------------------- #
def joint_finetune(
    cfg: dict,
    net_a: NetworkA,
    net_b: NetworkB,
    train_reactions: list[Reaction],
    val_reactions: list[Reaction],
    *,
    device: torch.device,
    phase4_saddle_rmsd: float,
    logger: RunLogger | None = None,
) -> tuple[NetworkA, NetworkB, dict]:
    """Unfreeze both networks, LR / 10, short run; A's RMSD must not degrade >10%."""
    tr = cfg["train"]
    net_a = net_a.to(device).train()
    net_b = net_b.to(device).train()
    for p in net_b.parameters():
        p.requires_grad_(True)

    opt = torch.optim.Adam(list(net_a.parameters()) + list(net_b.parameters()), lr=tr["lr"] / 10.0)
    aux = AuxConfig(enabled=True, weight=tr.get("aux_weight", 0.1), n_steps=tr.get("aux_steps", 16))
    loader = reaction_loader(train_reactions, tr["batch_size"], shuffle=True)
    epochs = max(1, int(tr["epochs"] * tr.get("joint_epoch_frac", 0.1)))
    summary = {}
    for epoch in range(epochs):
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = network_a_step(net_a, batch, net_b=net_b, aux=aux)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                list(net_a.parameters()) + list(net_b.parameters()), tr["grad_clip"]
            )
            opt.step()
        val = evaluate_network_a(net_a, val_reactions, device)
        degraded = val["val_saddle_rmsd"] > 1.1 * phase4_saddle_rmsd
        if logger:
            logger.log({"phase": 5, "degraded": bool(degraded), **val}, step=epoch)
        summary = {**val, "degraded_vs_phase4": bool(degraded)}
    return net_a, net_b, summary
