"""Network A: saddle predictor ``(R, P, M) -> displacement from M``.

R<->P symmetry is enforced **by construction** (CLAUDE.md "R<->P symmetry of
Network A"): R and P are embedded with a *shared* encoder and combined by a
symmetric sum, so swapping R and P leaves the prediction unchanged. The output
is a per-atom displacement; the predicted saddle is ``M + displacement``
(residual-style, easier to learn than absolute coordinates).

The same class serves the Phase-2 direct baseline (saddle loss only) and the
Phase-4 model (saddle loss + auxiliary descent loss) -- the only difference is
the training objective, so the architecture comparison is clean.
"""

from __future__ import annotations

import torch
from e3nn import o3
from torch import nn

from katabasis.data.batching import GraphBatch
from katabasis.models.backbone import EquivariantBackbone
from katabasis.models.embedding import AtomEmbedding
from katabasis.models.irreps import OUTPUT_VECTOR_IRREPS, feature_irreps


class NetworkA(nn.Module):
    """Predict the saddle as ``M + f(R, P, M)`` with ``f`` R<->P symmetric."""

    def __init__(
        self,
        *,
        embed_dim: int = 32,
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
        hidden = feature_irreps(n_scalars, n_vectors, n_tensors)

        bb_kwargs = dict(
            n_layers=n_layers,
            l_max=l_max,
            n_scalars=n_scalars,
            n_vectors=n_vectors,
            n_tensors=n_tensors,
            r_max=r_max,
            num_basis=num_basis,
            radial_hidden=radial_hidden,
        )
        # Shared encoder applied to R and to P (weights tied -> R<->P symmetry).
        self.encoder_rp = EquivariantBackbone(self.embed.irreps_out, irreps_out=hidden, **bb_kwargs)
        # Main network operates on the midpoint M, conditioned on the symmetric
        # combination of the R/P encodings.
        main_input = (self.embed.irreps_out + hidden).simplify()
        self.main = EquivariantBackbone(main_input, irreps_out=OUTPUT_VECTOR_IRREPS, **bb_kwargs)
        # Residual init: start with ~zero displacement so the initial prediction
        # is M itself (the trivial baseline), then learn the correction.
        with torch.no_grad():
            self.main.readout.weight.zero_()

    def forward(
        self,
        reactant: torch.Tensor,  # (N, 3)
        product: torch.Tensor,  # (N, 3)
        midpoint: torch.Tensor,  # (N, 3)
        graph: GraphBatch,
    ) -> torch.Tensor:
        z_emb = self.embed(graph.z)  # (N, embed_dim)
        f_r = self.encoder_rp(reactant, z_emb, graph.batch)
        f_p = self.encoder_rp(product, z_emb, graph.batch)
        f_sym = f_r + f_p  # symmetric combine -> invariant to R<->P swap
        node_input = torch.cat([z_emb, f_sym], dim=1)
        displacement = self.main(midpoint, node_input, graph.batch)  # (N, 3)
        return midpoint + displacement

    @property
    def irreps_out(self) -> o3.Irreps:
        return OUTPUT_VECTOR_IRREPS
