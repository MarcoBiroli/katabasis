"""Network B: target-conditioned, contractive descent flow.

Given a current state and a target geometry, B outputs an equivariant velocity
field that flows the current state into the target's basin. Conditioning is via
the **target-displacement vector field** ``(target - current)`` as an l=1 input
feature (automatically SE(3)-equivariant; CLAUDE.md "Conditioning mechanism"),
augmented with the per-atom RMSD-to-target scalar and a time embedding driving
per-layer FiLM. R<->P symmetry is automatic: a single network handles both
descents, distinguished only by which target it is given.
"""

from __future__ import annotations

import math

import torch
from e3nn import o3
from torch import nn

from katabasis.data.batching import GraphBatch
from katabasis.models.backbone import EquivariantBackbone
from katabasis.models.embedding import AtomEmbedding
from katabasis.models.irreps import OUTPUT_VECTOR_IRREPS


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of scalar time ``t`` (N,) -> (N, dim) invariant feat."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half, 1)
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if emb.shape[1] < dim:  # odd dim padding
        emb = torch.cat([emb, torch.zeros(emb.shape[0], dim - emb.shape[1], device=t.device)], 1)
    return emb


class NetworkB(nn.Module):
    """Equivariant velocity field ``v(x, target, t)`` -> per-atom velocity (1o).

    The output is split into a unit direction (1o) and a per-atom scalar speed
    (0e), so the velocity-scale normalization (predict direction + speed) is
    native to the architecture. ``forward`` returns the full velocity
    ``speed * direction``; ``forward_split`` exposes both pieces.
    """

    def __init__(
        self,
        *,
        embed_dim: int = 32,
        time_dim: int = 16,
        n_layers: int = 4,
        l_max: int = 2,
        n_scalars: int = 32,
        n_vectors: int = 16,
        n_tensors: int = 8,
        r_max: float = 4.0,
        num_basis: int = 8,
        radial_hidden: tuple[int, ...] = (64,),
    ):
        super().__init__()
        self.embed = AtomEmbedding(embed_dim)
        self.time_dim = time_dim

        # Node input irreps: atom scalars + RMSD-to-target scalar + time emb (0e)
        # + the target-displacement vector (1o).
        scalar_dim = embed_dim + 1 + time_dim
        self.node_input_irreps = o3.Irreps(f"{scalar_dim}x0e + 1x1o")
        # Output: a direction vector (1o) plus a scalar speed (0e).
        out_irreps = o3.Irreps("1x0e + 1x1o")
        self.backbone = EquivariantBackbone(
            self.node_input_irreps,
            n_layers=n_layers,
            l_max=l_max,
            n_scalars=n_scalars,
            n_vectors=n_vectors,
            n_tensors=n_tensors,
            r_max=r_max,
            num_basis=num_basis,
            radial_hidden=radial_hidden,
            irreps_out=out_irreps,
            film_dim=time_dim + 1,  # time emb + global rmsd
        )
        # Small-scale readout init: the initial velocity field is tiny (so early
        # integration is numerically stable) but still target-responsive.
        with torch.no_grad():
            self.backbone.readout.weight.mul_(1e-2)

    def _node_input(
        self, x: torch.Tensor, target: torch.Tensor, t_node: torch.Tensor, graph: GraphBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        disp = target - x  # (N, 3) l=1 conditioning feature
        rmsd = disp.norm(dim=1, keepdim=True)  # (N, 1) invariant
        z_emb = self.embed(graph.z)  # (N, embed_dim)
        t_emb = time_embedding(t_node, self.time_dim)  # (N, time_dim)
        scalars = torch.cat([z_emb, rmsd, t_emb], dim=1)
        node_input = torch.cat([scalars, disp], dim=1)  # scalars (0e) then 1o
        film_cond = torch.cat([t_emb, rmsd], dim=1)  # (N, time_dim + 1)
        return node_input, film_cond

    def forward_split(
        self,
        x: torch.Tensor,  # (N, 3)
        target: torch.Tensor,  # (N, 3)
        t_node: torch.Tensor,  # (N,)
        graph: GraphBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(direction (N,3) unit-ish, speed (N,1) >= 0)``."""
        node_input, film_cond = self._node_input(x, target, t_node, graph)
        out = self.backbone(x, node_input, graph.batch, film_cond=film_cond)  # (N, 4)
        speed = torch.nn.functional.softplus(out[:, :1])  # (N, 1) non-negative
        direction = out[:, 1:]  # (N, 3)
        return direction, speed

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        t_node: torch.Tensor,
        graph: GraphBatch,
    ) -> torch.Tensor:
        """Full velocity ``v = speed * direction`` (N, 3)."""
        direction, speed = self.forward_split(x, target, t_node, graph)
        return speed * direction

    @property
    def irreps_out(self) -> o3.Irreps:
        return OUTPUT_VECTOR_IRREPS
