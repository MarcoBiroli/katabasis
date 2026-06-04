"""Graph-style batching for variable-size molecules.

Molecules in a batch have different atom counts, so we concatenate nodes along
a single axis and track a ``batch`` index (which graph each node belongs to),
mirroring the PyG convention. The equivariant networks build a radius graph
from ``positions`` + ``batch`` so messages never cross molecule boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GraphBatch:
    """A batch of molecules as one big disconnected graph.

    Attributes
    ----------
    z:          ``(sumN,)`` atomic numbers
    batch:      ``(sumN,)`` graph index per node
    ptr:        ``(B + 1,)`` cumulative node counts (CSR-style boundaries)
    n_graphs:   number of molecules
    """

    z: torch.Tensor
    batch: torch.Tensor
    ptr: torch.Tensor
    n_graphs: int

    def to(self, device: torch.device | str) -> GraphBatch:
        return GraphBatch(
            z=self.z.to(device),
            batch=self.batch.to(device),
            ptr=self.ptr.to(device),
            n_graphs=self.n_graphs,
        )


def build_batch_index(atom_counts: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(batch, ptr)`` for a list of per-graph atom counts."""
    batch = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(atom_counts)])
    ptr = torch.zeros(len(atom_counts) + 1, dtype=torch.long)
    ptr[1:] = torch.tensor(atom_counts, dtype=torch.long).cumsum(0)
    return batch, ptr


def collate_positions(coords_list: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate a list of ``(N_i, 3)`` tensors into ``(sumN, 3)``."""
    return torch.cat(coords_list, dim=0)
