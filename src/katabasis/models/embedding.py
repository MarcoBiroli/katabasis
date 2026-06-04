"""Atomic-number embedding into invariant (0e) scalar node features."""

from __future__ import annotations

import torch
from e3nn import o3
from torch import nn

# Covered elements: H, C, N, O, F, Cl, Br (Halo8 chemistry). Index by Z.
SUPPORTED_Z = (1, 6, 7, 8, 9, 17, 35)


class AtomEmbedding(nn.Module):
    """Embed atomic numbers as ``dim`` scalar (0e) features."""

    def __init__(self, dim: int = 32, max_z: int = 36):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(max_z + 1, dim)
        self.irreps_out = o3.Irreps(f"{dim}x0e")

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # (N,) -> (N, dim)
        return self.embed(z)
