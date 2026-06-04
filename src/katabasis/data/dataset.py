"""Torch datasets and collate functions for Networks A and B.

``ReactionDataset`` yields per-reaction tensors for Network A (saddle
prediction). ``DescentFlowDataset`` yields flow-matching samples for Network B,
drawing a ``t ~ Beta(a, b)`` (saddle-biased) per item and building the
arc-length bridge target on the fly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from katabasis.data.alignment import permutation_align
from katabasis.data.arclength import DescentLeg, bridge_tangent, build_legs, interpolate_bridge
from katabasis.data.batching import build_batch_index
from katabasis.data.connectivity import interchangeable_groups
from katabasis.data.midpoint import midpoint
from katabasis.data.types import Reaction


def _t(x: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=dtype)


# --------------------------------------------------------------------------- #
# Network A: saddle prediction from (R, P, M)
# --------------------------------------------------------------------------- #
@dataclass
class ReactionSample:
    reaction_id: str
    z: torch.Tensor  # (N,)
    reactant: torch.Tensor  # (N, 3)
    product: torch.Tensor  # (N, 3)
    midpoint: torch.Tensor  # (N, 3)
    saddle: torch.Tensor  # (N, 3)
    groups: list[list[int]]


class ReactionDataset(Dataset):
    """Per-reaction samples for Network A / the direct baseline."""

    def __init__(self, reactions: list[Reaction]):
        self.reactions = reactions
        # Precompute interchangeable groups from the (stable) reactant geometry.
        self._groups = [interchangeable_groups(r.atomic_numbers, r.reactant) for r in reactions]

    def __len__(self) -> int:
        return len(self.reactions)

    def __getitem__(self, idx: int) -> ReactionSample:
        r = self.reactions[idx]
        m = midpoint(r.reactant, r.product)
        return ReactionSample(
            reaction_id=r.reaction_id,
            z=_t(r.atomic_numbers, torch.long),
            reactant=_t(r.reactant),
            product=_t(r.product),
            midpoint=_t(m),
            saddle=_t(r.saddle),
            groups=self._groups[idx],
        )


@dataclass
class ReactionBatch:
    z: torch.Tensor
    reactant: torch.Tensor
    product: torch.Tensor
    midpoint: torch.Tensor
    saddle: torch.Tensor
    batch: torch.Tensor
    ptr: torch.Tensor
    n_graphs: int
    groups: list[list[int]]  # per-graph, atom indices local to that graph
    reaction_ids: list[str]

    def to(self, device) -> ReactionBatch:
        return ReactionBatch(
            z=self.z.to(device),
            reactant=self.reactant.to(device),
            product=self.product.to(device),
            midpoint=self.midpoint.to(device),
            saddle=self.saddle.to(device),
            batch=self.batch.to(device),
            ptr=self.ptr.to(device),
            n_graphs=self.n_graphs,
            groups=self.groups,
            reaction_ids=self.reaction_ids,
        )


def collate_reactions(samples: list[ReactionSample]) -> ReactionBatch:
    counts = [s.z.shape[0] for s in samples]
    batch, ptr = build_batch_index(counts)
    return ReactionBatch(
        z=torch.cat([s.z for s in samples]),
        reactant=torch.cat([s.reactant for s in samples]),
        product=torch.cat([s.product for s in samples]),
        midpoint=torch.cat([s.midpoint for s in samples]),
        saddle=torch.cat([s.saddle for s in samples]),
        batch=batch,
        ptr=ptr,
        n_graphs=len(samples),
        groups=[s.groups for s in samples],
        reaction_ids=[s.reaction_id for s in samples],
    )


# --------------------------------------------------------------------------- #
# Network B: flow-matching descent samples
# --------------------------------------------------------------------------- #
@dataclass
class FlowSample:
    reaction_id: str
    z: torch.Tensor  # (N,)
    x_t: torch.Tensor  # (N, 3) current state on the bridge
    target: torch.Tensor  # (N, 3) basin endpoint (R or P)
    t: torch.Tensor  # scalar in [0, 1]
    unit_tangent: torch.Tensor  # (N, 3) direction dx/ds (Frobenius-normalized)
    speed: torch.Tensor  # scalar leg length L
    dxdt: torch.Tensor  # (N, 3) = speed * unit_tangent


class DescentFlowDataset(Dataset):
    """Flow-matching samples drawn from descent legs with saddle-biased ``t``.

    Each ``__getitem__`` returns one ``(x_t, target, t, tangent)`` sample. A
    reaction contributes both legs; the leg and ``t`` are sampled per access so
    epochs see fresh bridge points (standard flow-matching practice).
    """

    def __init__(
        self,
        reactions: list[Reaction],
        *,
        t_alpha: float = 0.5,
        t_beta: float = 1.5,
        noise_sigma: float = 0.0,
        seed: int = 0,
    ):
        self.legs: list[tuple[str, DescentLeg]] = []
        for r in reactions:
            for leg in build_legs(r):
                self.legs.append((r.reaction_id, leg))
        self._z = {}
        for r in reactions:
            self._z[r.reaction_id] = r.atomic_numbers
        self.t_alpha = t_alpha
        self.t_beta = t_beta
        self.noise_sigma = noise_sigma
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.legs)

    def __getitem__(self, idx: int) -> FlowSample:
        rid, leg = self.legs[idx]
        t = float(self._rng.beta(self.t_alpha, self.t_beta))
        x_t = interpolate_bridge(leg, t)
        unit, dxdt, speed = bridge_tangent(leg, t)
        if self.noise_sigma > 0:
            x_t = x_t + self.noise_sigma * self._rng.standard_normal(x_t.shape)
        return FlowSample(
            reaction_id=rid,
            z=_t(self._z[rid], torch.long),
            x_t=_t(x_t),
            target=_t(leg.endpoint),
            t=torch.tensor(t, dtype=torch.float32),
            unit_tangent=_t(unit),
            speed=torch.tensor(speed, dtype=torch.float32),
            dxdt=_t(dxdt),
        )


@dataclass
class FlowBatch:
    z: torch.Tensor
    x_t: torch.Tensor
    target: torch.Tensor
    t: torch.Tensor  # (B,) per-graph time
    t_node: torch.Tensor  # (sumN,) broadcast to nodes
    unit_tangent: torch.Tensor
    speed: torch.Tensor  # (B,)
    dxdt: torch.Tensor
    batch: torch.Tensor
    ptr: torch.Tensor
    n_graphs: int
    reaction_ids: list[str]

    def to(self, device) -> FlowBatch:
        return FlowBatch(
            **{
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in self.__dict__.items()
            }
        )


def collate_flow(samples: list[FlowSample]) -> FlowBatch:
    counts = [s.z.shape[0] for s in samples]
    batch, ptr = build_batch_index(counts)
    t = torch.stack([s.t for s in samples])
    t_node = torch.cat([s.t.repeat(n) for s, n in zip(samples, counts, strict=False)])
    return FlowBatch(
        z=torch.cat([s.z for s in samples]),
        x_t=torch.cat([s.x_t for s in samples]),
        target=torch.cat([s.target for s in samples]),
        t=t,
        t_node=t_node,
        unit_tangent=torch.cat([s.unit_tangent for s in samples]),
        speed=torch.stack([s.speed for s in samples]),
        dxdt=torch.cat([s.dxdt for s in samples]),
        batch=batch,
        ptr=ptr,
        n_graphs=len(samples),
        reaction_ids=[s.reaction_id for s in samples],
    )


def align_to_reference(
    pred: np.ndarray, ref: np.ndarray, atomic_numbers: np.ndarray, groups
) -> tuple[np.ndarray, float]:
    """Convenience: connectivity-constrained align ``pred`` onto ``ref``."""
    _, aligned, rmsd = permutation_align(pred, ref, atomic_numbers, groups)
    return aligned, rmsd
