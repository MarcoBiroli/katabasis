"""Shared training utilities: model builders, dataloaders, and loss steps."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from katabasis.data.dataset import (
    DescentFlowDataset,
    ReactionBatch,
    ReactionDataset,
    collate_flow,
    collate_reactions,
)
from katabasis.data.types import Reaction
from katabasis.eval.integrate import integrate_flow
from katabasis.losses.flow_matching import flow_matching_loss
from katabasis.losses.rmsd import batched_aligned_rmsd
from katabasis.models.network_a import NetworkA
from katabasis.models.network_b import NetworkB


def get_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _model_kwargs(model_cfg: dict) -> dict:
    keys = (
        "embed_dim",
        "n_layers",
        "l_max",
        "n_scalars",
        "n_vectors",
        "n_tensors",
        "r_max",
        "num_basis",
        "radial_hidden",
    )
    out = {k: model_cfg[k] for k in keys if k in model_cfg}
    if "radial_hidden" in out:
        out["radial_hidden"] = tuple(out["radial_hidden"])
    return out


def build_network_a(model_cfg: dict) -> NetworkA:
    return NetworkA(**_model_kwargs(model_cfg))


def build_network_b(model_cfg: dict) -> NetworkB:
    kwargs = _model_kwargs(model_cfg)
    if "time_dim" in model_cfg:
        kwargs["time_dim"] = model_cfg["time_dim"]
    return NetworkB(**kwargs)


def reaction_loader(
    reactions: list[Reaction], batch_size: int, shuffle: bool, num_workers: int = 0
) -> DataLoader:
    return DataLoader(
        ReactionDataset(reactions),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_reactions,
        num_workers=num_workers,
    )


def flow_loader(
    reactions: list[Reaction],
    batch_size: int,
    shuffle: bool,
    *,
    t_alpha: float,
    t_beta: float,
    noise_sigma: float,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    ds = DescentFlowDataset(
        reactions, t_alpha=t_alpha, t_beta=t_beta, noise_sigma=noise_sigma, seed=seed
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_flow,
        num_workers=num_workers,
    )


# --------------------------------------------------------------------------- #
# Loss steps
# --------------------------------------------------------------------------- #
@dataclass
class AuxConfig:
    """Auxiliary descent-loss settings for Network A (Phase 4/5)."""

    enabled: bool = False
    weight: float = 0.1  # auxiliary weight; primary:aux = 10:1 by default
    n_steps: int = 16  # integration steps during training (cheaper than eval's 32)


def network_a_step(
    net_a: NetworkA,
    batch: ReactionBatch,
    *,
    net_b: NetworkB | None = None,
    aux: AuxConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Compute Network A's loss: primary saddle RMSD (+ optional aux descent).

    The auxiliary term runs two descents from the predicted saddle with a frozen
    ``net_b`` (toward R and toward P) and penalizes endpoint mismatch. Primary
    weight stays >= 10x the auxiliary (never let aux dominate saddle RMSD).
    """
    pred_saddle = net_a(batch.reactant, batch.product, batch.midpoint, batch)
    primary = batched_aligned_rmsd(pred_saddle, batch.saddle, batch.ptr, batch.z, batch.groups)
    out: dict[str, torch.Tensor] = {"primary": primary, "loss": primary}

    if aux is not None and aux.enabled and net_b is not None:
        end_r = integrate_flow(net_b, pred_saddle, batch.reactant, batch, n_steps=aux.n_steps)
        end_p = integrate_flow(net_b, pred_saddle, batch.product, batch, n_steps=aux.n_steps)
        aux_r = batched_aligned_rmsd(end_r, batch.reactant, batch.ptr, batch.z, batch.groups)
        aux_p = batched_aligned_rmsd(end_p, batch.product, batch.ptr, batch.z, batch.groups)
        aux_loss = 0.5 * (aux_r + aux_p)
        out["aux"] = aux_loss
        out["loss"] = primary + aux.weight * aux_loss
    return out


def network_b_step(
    net_b: NetworkB,
    batch,
    *,
    lambda_c: float = 0.0,
    lambda_e: float = 0.0,
    contractivity_gate: float = 0.2,
    contractivity_subsample: int = 0,
    contractivity_iters: int = 6,
    endpoint_steps: int = 16,
) -> dict[str, torch.Tensor]:
    """Network B flow-matching loss (+ optional contractivity & endpoint terms)."""
    direction, speed = net_b.forward_split(batch.x_t, batch.target, batch.t_node, batch)
    fm = flow_matching_loss(direction, speed, batch.unit_tangent, batch.speed, batch.batch)
    loss = fm["loss"]
    out = {"loss": loss, "direction_loss": fm["direction_loss"], "speed_loss": fm["speed_loss"]}

    # Contractivity regularizer, t-gated and applied at the basin target.
    if lambda_c > 0:
        gate_mask = batch.t > contractivity_gate  # (B,) which graphs to penalize
        if gate_mask.any():
            penalty = _contractivity_term(
                net_b, batch, gate_mask, contractivity_subsample, contractivity_iters
            )
            out["contractivity"] = penalty.detach()
            loss = loss + lambda_c * penalty

    # Late-training endpoint correction.
    if lambda_e > 0:
        end = integrate_flow(net_b, batch.x_t, batch.target, batch, n_steps=endpoint_steps)
        endpoint = batched_aligned_rmsd(
            end, batch.target, batch.ptr, batch.z, _trivial_groups(batch.ptr)
        )
        out["endpoint"] = endpoint.detach()
        loss = loss + lambda_e * endpoint

    out["loss"] = loss
    return out


def _trivial_groups(ptr: torch.Tensor) -> list[list[list[int]]]:
    """Per-graph singleton groups (used when interchangeable groups aren't needed)."""
    return [[[i] for i in range(int(ptr[g + 1] - ptr[g]))] for g in range(len(ptr) - 1)]


def _contractivity_term(net_b, batch, gate_mask, subsample, iters):
    """Evaluate the contractivity penalty at the basin target for gated graphs."""
    # Evaluate sym(J_v) at the target (the basin fixed point), per CLAUDE.md.
    target = batch.target

    def f(x):
        t_node = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        return net_b(x, target, t_node, batch)

    n_graphs = batch.n_graphs
    mu = _per_graph_abscissa(f, target, batch.batch, n_graphs, iters)
    relu = torch.relu(mu)
    relu = relu * gate_mask.to(relu.dtype)
    denom = gate_mask.sum().clamp_min(1).to(relu.dtype)
    return relu.sum() / denom


def _per_graph_abscissa(f, x, batch_idx, n_graphs, iters):
    from katabasis.losses.contractivity import numerical_abscissa

    return numerical_abscissa(f, x, batch_idx, n_graphs, n_iter=iters, create_graph=True)
