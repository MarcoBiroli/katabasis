"""SE(3)-equivariant message-passing backbone (e3nn).

A NequIP-flavoured convolution: edge directions are embedded with spherical
harmonics, edge lengths with a smooth radial basis whose MLP produces the
tensor-product path weights, messages are gathered per node, and a gated
nonlinearity follows. Radius graphs are built per-molecule from the ``batch``
index so messages never cross molecule boundaries.

Default backbone: 3-4 layers, ``l_max = 2``, hidden multiplicities ~32
(PLAN.md Phase 3). All sizes come from config.
"""

from __future__ import annotations

import torch
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate
from torch import nn

from katabasis.models.irreps import feature_irreps, sh_irreps


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum ``src`` rows into ``dim_size`` buckets given by ``index`` (no torch_scatter dep)."""
    out = src.new_zeros((dim_size, src.shape[1]))
    out.index_add_(0, index, src)
    return out


def radius_graph(
    pos: torch.Tensor, batch: torch.Tensor, r_max: float, max_neighbors: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(edge_src, edge_dst)`` within ``r_max`` and the same molecule.

    O(N^2) per molecule -- fine for the small systems here (<~25 atoms). Self
    loops are excluded; ``max_neighbors`` caps degree for pathological cases.
    """
    dist = torch.cdist(pos, pos)  # (N, N)
    same = batch[:, None] == batch[None, :]
    mask = same & (dist <= r_max) & (dist > 1e-8)
    src, dst = torch.where(mask)
    if max_neighbors is not None and src.numel() > 0:
        # Keep nearest `max_neighbors` per destination node.
        keep = []
        for node in torch.unique(dst):
            sel = (dst == node).nonzero(as_tuple=True)[0]
            if sel.numel() > max_neighbors:
                d = dist[node, src[sel]]
                sel = sel[torch.topk(d, max_neighbors, largest=False).indices]
            keep.append(sel)
        idx = torch.cat(keep)
        src, dst = src[idx], dst[idx]
    return src, dst


def _gate_for(irreps_out: o3.Irreps) -> tuple[Gate, o3.Irreps]:
    """Build a Gate producing ``irreps_out`` and return ``(gate, conv_target)``.

    The convolution must emit scalars + one gate scalar per gated (l>0) irrep.
    """
    scalars = o3.Irreps([(mul, ir) for mul, ir in irreps_out if ir.l == 0])
    gated = o3.Irreps([(mul, ir) for mul, ir in irreps_out if ir.l > 0])
    n_gates = sum(mul for mul, _ in gated)
    gates = o3.Irreps(f"{n_gates}x0e") if n_gates else o3.Irreps("")
    gate = Gate(
        scalars,
        [torch.nn.functional.silu] * len(scalars),
        gates,
        [torch.sigmoid] * len(gates),
        gated,
    )
    return gate, gate.irreps_in.simplify()


class Convolution(nn.Module):
    """One equivariant convolution: edge tensor product with radial weights."""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_sh: o3.Irreps,
        irreps_out: o3.Irreps,
        num_basis: int,
        radial_hidden: list[int],
    ):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_out = o3.Irreps(irreps_out)

        # Build "uvu" instructions keeping only paths whose output we want.
        instructions = []
        irreps_mid = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_sh) in enumerate(self.irreps_sh):
                for ir_out in ir_in * ir_sh:
                    if ir_out in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        # Keep the intermediate irreps unsorted: the instruction indices map
        # directly to entries, and TensorProduct does not require sorted output.
        irreps_mid = o3.Irreps(irreps_mid)

        self.tp = o3.TensorProduct(
            self.irreps_in,
            self.irreps_sh,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )
        self.radial = FullyConnectedNet(
            [num_basis, *radial_hidden, self.tp.weight_numel],
            torch.nn.functional.silu,
        )
        self.linear = o3.Linear(irreps_mid, self.irreps_out)
        # Self-interaction (skip connection through a linear map).
        self.self_interaction = o3.Linear(self.irreps_in, self.irreps_out)

    def forward(
        self,
        x: torch.Tensor,  # (N, irreps_in)
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_sh: torch.Tensor,  # (E, irreps_sh)
        edge_radial: torch.Tensor,  # (E, num_basis)
    ) -> torch.Tensor:
        weights = self.radial(edge_radial)  # (E, weight_numel)
        messages = self.tp(x[edge_src], edge_sh, weights)  # (E, irreps_mid)
        agg = _scatter_sum(messages, edge_dst, x.shape[0])
        # Normalize by sqrt(avg degree) for stable scale (approximate).
        out = self.linear(agg) + self.self_interaction(x)
        return out


class EquivariantBackbone(nn.Module):
    """Stacked equivariant convolutions with gated nonlinearities.

    ``forward`` consumes node positions, a scalar node embedding, and a batch
    index, and returns per-node features in ``irreps_out``.
    """

    def __init__(
        self,
        irreps_node_input: o3.Irreps,
        *,
        n_layers: int = 4,
        l_max: int = 2,
        n_scalars: int = 32,
        n_vectors: int = 16,
        n_tensors: int = 8,
        r_max: float = 4.0,
        num_basis: int = 8,
        radial_hidden: tuple[int, ...] = (64,),
        irreps_out: o3.Irreps | str | None = None,
        film_dim: int = 0,
    ):
        super().__init__()
        self.r_max = r_max
        self.num_basis = num_basis
        self.n_scalars = n_scalars
        self.film_dim = film_dim
        self.irreps_sh = sh_irreps(l_max)
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        hidden = feature_irreps(n_scalars, n_vectors, n_tensors)
        self.irreps_out = o3.Irreps(irreps_out) if irreps_out is not None else hidden

        self.layers = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.films = nn.ModuleList()
        irreps = self.irreps_node_input
        for _ in range(n_layers):
            gate, conv_target = _gate_for(hidden)
            conv = Convolution(irreps, self.irreps_sh, conv_target, num_basis, list(radial_hidden))
            self.layers.append(conv)
            self.gates.append(gate)
            # FiLM modulates the scalar (0e) channels with scale+shift from a
            # global invariant conditioner -- equivariant because it only touches
            # invariant scalars (CLAUDE.md: FiLM on a scalar invariant per layer).
            self.films.append(nn.Linear(film_dim, 2 * n_scalars) if film_dim > 0 else None)
            irreps = hidden
        self.readout = o3.Linear(irreps, self.irreps_out)

    def forward(
        self,
        pos: torch.Tensor,  # (N, 3)
        node_input: torch.Tensor,  # (N, irreps_node_input)
        batch: torch.Tensor,  # (N,)
        film_cond: torch.Tensor | None = None,  # (N, film_dim) invariant conditioner
    ) -> torch.Tensor:
        edge_src, edge_dst = radius_graph(pos, batch, self.r_max)
        edge_vec = pos[edge_dst] - pos[edge_src]
        edge_len = edge_vec.norm(dim=1)
        edge_sh = o3.spherical_harmonics(
            self.irreps_sh, edge_vec, normalize=True, normalization="component"
        )
        edge_radial = soft_one_hot_linspace(
            edge_len, 0.0, self.r_max, self.num_basis, basis="smooth_finite", cutoff=True
        ) * (self.num_basis**0.5)

        x = node_input
        for conv, gate, film in zip(self.layers, self.gates, self.films, strict=False):
            x = conv(x, edge_src, edge_dst, edge_sh, edge_radial)
            x = gate(x)
            if film is not None and film_cond is not None:
                scale_shift = film(film_cond)  # (N, 2 * n_scalars)
                scale, shift = scale_shift[:, : self.n_scalars], scale_shift[:, self.n_scalars :]
                # Functional (no in-place) so double-backward through the FiLM'd
                # scalars works for the contractivity Jacobian estimate.
                scalars = x[:, : self.n_scalars] * (1.0 + scale) + shift
                x = torch.cat([scalars, x[:, self.n_scalars :]], dim=1)
        return self.readout(x)
