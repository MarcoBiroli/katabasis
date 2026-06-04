"""Differentiable, connectivity-constrained, Kabsch-aligned RMSD.

The optimal *permutation* (within interchangeable groups) is found with the
non-differentiable numpy routine on detached coordinates; the alignment
*rotation* is then recomputed differentiably in torch so gradients flow to the
predictor. Never raw Cartesian RMSD (CLAUDE.md).
"""

from __future__ import annotations

import numpy as np
import torch

from katabasis.data.alignment import permutation_align


def kabsch_rmsd_torch(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Differentiable Kabsch-aligned RMSD between ``(N, 3)`` ``p`` and ``q``.

    Permutation is assumed already applied. Differentiable through ``p`` (and
    ``q``) via torch SVD.
    """
    pc = p - p.mean(0, keepdim=True)
    qc = q - q.mean(0, keepdim=True)
    h = pc.transpose(0, 1) @ qc  # (3, 3)
    u, _, vt = torch.linalg.svd(h)
    d = torch.sign(torch.det(vt.transpose(0, 1) @ u.transpose(0, 1)))
    diag = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
    rot = vt.transpose(0, 1) @ diag @ u.transpose(0, 1)
    p_aligned = pc @ rot.transpose(0, 1)
    return torch.sqrt(torch.mean(torch.sum((p_aligned - qc) ** 2, dim=1)) + 1e-12)


def batched_aligned_rmsd(
    pred: torch.Tensor,  # (sumN, 3)
    target: torch.Tensor,  # (sumN, 3)
    ptr: torch.Tensor,  # (B + 1,)
    z: torch.Tensor,  # (sumN,)
    groups_per_graph: list[list[list[int]]],
    *,
    reduce: str = "mean",
) -> torch.Tensor:
    """Mean connectivity-constrained aligned RMSD over a batch of molecules."""
    rmsds = []
    z_np = z.detach().cpu().numpy()
    for i in range(len(ptr) - 1):
        a, b = int(ptr[i]), int(ptr[i + 1])
        p_i, q_i = pred[a:b], target[a:b]
        groups = groups_per_graph[i]
        perm, _, _ = permutation_align(
            p_i.detach().cpu().numpy(), q_i.detach().cpu().numpy(), z_np[a:b], groups
        )
        perm_t = torch.as_tensor(perm, dtype=torch.long, device=pred.device)
        rmsds.append(kabsch_rmsd_torch(p_i[perm_t], q_i))
    out = torch.stack(rmsds)
    if reduce == "mean":
        return out.mean()
    if reduce == "sum":
        return out.sum()
    return out


def rmsd_to_target(pred: np.ndarray, target: np.ndarray, z: np.ndarray, groups) -> float:
    """Evaluation helper (numpy): connectivity-constrained aligned RMSD."""
    _, _, rmsd = permutation_align(pred, target, z, groups)
    return rmsd
