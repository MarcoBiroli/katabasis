"""Kabsch alignment + connectivity-constrained permutation-aware RMSD.

Used **everywhere** two structures are compared (CLAUDE.md code conventions):
never raw Cartesian RMSD. The permutation search is constrained by
:func:`katabasis.data.connectivity.interchangeable_groups` so only genuinely
interchangeable atoms can be swapped -- a global Hungarian match over all atoms
of one element would swap H across different heavy atoms and silently corrupt
every RMSD.

The numpy functions here are for preprocessing/evaluation. A differentiable
torch RMSD (fixed permutation, Kabsch with gradients) lives in
:mod:`katabasis.losses.rmsd` for use inside training loops.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from katabasis.data.connectivity import interchangeable_groups

# Above this many combined within-group permutations we fall back from the exact
# brute force to the iterative Hungarian alternation.
MAX_BRUTE_FORCE = 20000


def centroid(coords: np.ndarray) -> np.ndarray:
    """Mean over atoms; ``(..., N, 3) -> (..., 1, 3)``."""
    return coords.mean(axis=-2, keepdims=True)


def kabsch_rotation(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Optimal proper rotation aligning centered ``p`` onto centered ``q``.

    Both inputs are ``(N, 3)`` and assumed already centered. Returns ``(3, 3)``
    with determinant +1 (reflection corrected), acting as ``p @ R.T``.
    """
    h = p.T @ q  # (3, 3) covariance
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    diag = np.diag(np.array([1.0, 1.0, d]))
    return vt.T @ diag @ u.T


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))


def kabsch_align(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, float]:
    """Rigidly align ``p`` onto ``q`` (no permutation). Returns ``(p_aligned, rmsd)``."""
    pc = p - centroid(p)
    qc = q - centroid(q)
    rot = kabsch_rotation(pc, qc)
    p_aligned = pc @ rot.T + centroid(q)
    return p_aligned, _rmsd(p_aligned, q)


def _brute_force_align(
    p: np.ndarray, q: np.ndarray, nontrivial_groups: list[list[int]], n: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact connectivity-constrained alignment by enumerating group permutations.

    Group sizes are tiny (methyl H, CF3 F, ...) and groups are few, so the
    Cartesian product of within-group permutations is small. Coupling Kabsch
    with a greedy Hungarian step can get stuck in a local optimum (a
    near-symmetric group lets the global rotation "spin" instead of permuting);
    brute force over the (few) candidates avoids that.
    """
    best_perm = np.arange(n)
    best_aligned, best_rmsd = kabsch_align(p, q)
    # Per-group permutations: list of (group, [tuple-permutations]).
    per_group = [(g, list(itertools.permutations(g))) for g in nontrivial_groups]
    for choice in itertools.product(*[perms for _, perms in per_group]):
        perm = np.arange(n)
        for (g, _), assigned in zip(per_group, choice, strict=False):
            perm[np.array(g)] = np.array(assigned)
        aligned, rmsd = kabsch_align(p[perm], q)
        if rmsd < best_rmsd:
            best_rmsd, best_perm, best_aligned = rmsd, perm, aligned
    return best_perm, best_aligned, best_rmsd


def permutation_align(
    p: np.ndarray,  # (N, 3) the structure to be reordered/aligned
    q: np.ndarray,  # (N, 3) the reference
    atomic_numbers: np.ndarray,  # (N,)
    groups: list[list[int]] | None = None,
    *,
    reference_coords: np.ndarray | None = None,
    max_iter: int = 10,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Connectivity-constrained permutation + Kabsch alignment of ``p`` to ``q``.

    Iterates: Kabsch-align, then within each interchangeable group solve a local
    assignment problem on current pairwise distances, until the permutation is
    stable. Permutation/alignment is a chicken-and-egg problem; the alternation
    converges quickly because group sizes are tiny (methyl H, CF3 F, ...).

    Parameters
    ----------
    groups:
        Interchangeable-atom groups. If ``None`` they are inferred from
        ``reference_coords`` (defaults to ``q``) -- always from a *stable*
        geometry, never from a saddle whose bonds are partly broken.

    Returns
    -------
    (perm, p_aligned, rmsd) where ``p[perm]`` reordered then rigid-aligned gives
    ``p_aligned`` and the connectivity-constrained minimal RMSD to ``q``.
    """
    n = p.shape[0]
    if groups is None:
        ref = reference_coords if reference_coords is not None else q
        groups = interchangeable_groups(atomic_numbers, ref)

    nontrivial = [g for g in groups if len(g) > 1]
    n_combos = math.prod(math.factorial(len(g)) for g in nontrivial) if nontrivial else 1
    if n_combos <= MAX_BRUTE_FORCE:
        return _brute_force_align(p, q, nontrivial, n)

    # Fallback: iterative Hungarian alternation for pathologically large groups.
    perm = np.arange(n)
    prev_perm = None
    p_aligned = p
    rmsd = float("inf")
    for _ in range(max_iter):
        p_perm = p[perm]
        p_aligned, rmsd = kabsch_align(p_perm, q)
        # Re-solve assignment within each group based on aligned distances.
        new_perm = perm.copy()
        for g in groups:
            if len(g) <= 1:
                continue
            idx = np.array(g)
            cost = np.linalg.norm(
                p_aligned[idx][:, None, :] - q[idx][None, :, :], axis=-1
            )  # (m, m): rows are current atoms, cols are reference slots
            rows, cols = linear_sum_assignment(cost)
            # Map: reference slot idx[col] should be filled by current atom idx[row].
            assigned = np.empty(len(idx), dtype=int)
            for r, c in zip(rows, cols, strict=False):
                assigned[c] = perm[idx[r]]
            new_perm[idx] = assigned
        if prev_perm is not None and np.array_equal(new_perm, perm):
            break
        prev_perm = perm
        perm = new_perm

    # Final alignment with the converged permutation.
    p_aligned, rmsd = kabsch_align(p[perm], q)
    return perm, p_aligned, rmsd


def aligned_rmsd(
    p: np.ndarray,
    q: np.ndarray,
    atomic_numbers: np.ndarray,
    groups: list[list[int]] | None = None,
    *,
    reference_coords: np.ndarray | None = None,
) -> float:
    """Connectivity-constrained, Kabsch-aligned RMSD between ``p`` and ``q``."""
    _, _, rmsd = permutation_align(p, q, atomic_numbers, groups, reference_coords=reference_coords)
    return rmsd
