"""Connectivity inference and *interchangeable-atom* grouping.

This module is the guardrail behind Critical correctness requirement 3: when we
align two structures we may only permute atoms that are *genuinely*
interchangeable (the three H of one methyl, the three F of one CF3). A naive
"all H of one element are interchangeable" rule will swap an H on carbon A with
an H on carbon B and silently corrupt every RMSD in the project.

We define interchangeability by an element-aware Weisfeiler--Lehman style
colour refinement over the bond graph: two atoms are interchangeable iff they
have the same refined colour *and* are bonded to a common neighbour (so they
are terminal substituents on the same heavy atom, not merely two atoms that
happen to sit in symmetric environments on opposite ends of the molecule).
"""

from __future__ import annotations

import numpy as np
from ase.data import covalent_radii

# Bond is declared when distance < (r_cov_i + r_cov_j) * (1 + tolerance).
DEFAULT_BOND_TOLERANCE = 0.45


def infer_bonds(
    atomic_numbers: np.ndarray,  # (N,)
    coords: np.ndarray,  # (N, 3)
    tolerance: float = DEFAULT_BOND_TOLERANCE,
) -> np.ndarray:
    """Return an ``(N, N)`` boolean adjacency matrix from covalent radii.

    A geometry-based heuristic; for a reaction we infer bonds from a *stable*
    geometry (the reactant) and reuse that topology, because bonds are by
    definition broken/formed at the saddle.
    """
    radii = covalent_radii[atomic_numbers]  # (N,)
    diff = coords[:, None, :] - coords[None, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=-1)  # (N, N)
    cutoff = (radii[:, None] + radii[None, :]) * (1.0 + tolerance)
    adj = (dist < cutoff) & (dist > 1e-6)
    np.fill_diagonal(adj, False)
    return adj.astype(bool)


def refine_colors(
    atomic_numbers: np.ndarray,  # (N,)
    adjacency: np.ndarray,  # (N, N) bool
    n_iter: int = 4,
) -> np.ndarray:
    """Weisfeiler--Lehman colour refinement; returns ``(N,)`` integer colours.

    Atoms with equal colour are indistinguishable up to the local bonded
    environment captured within ``n_iter`` hops.
    """
    n = atomic_numbers.shape[0]
    colors = atomic_numbers.astype(np.int64).copy()
    for _ in range(n_iter):
        signatures = []
        for i in range(n):
            neigh = np.sort(colors[adjacency[i]]).tolist()
            signatures.append((int(colors[i]), tuple(neigh)))
        # Re-label signatures to dense integer colours, deterministically.
        unique = {sig: idx for idx, sig in enumerate(sorted(set(signatures)))}
        new_colors = np.array([unique[s] for s in signatures], dtype=np.int64)
        if np.array_equal(new_colors, colors):
            break
        colors = new_colors
    return colors


def interchangeable_groups(
    atomic_numbers: np.ndarray,  # (N,)
    coords: np.ndarray,  # (N, 3)
    tolerance: float = DEFAULT_BOND_TOLERANCE,
    n_iter: int = 4,
) -> list[list[int]]:
    """Partition atom indices into groups that may be freely permuted.

    Two atoms share a group iff (a) they have identical refined WL colours and
    (b) they share at least one common bonded neighbour (terminal substituents
    on the same heavy atom). Singletons are returned as length-1 groups so the
    union of all groups is exactly ``range(N)``.
    """
    n = atomic_numbers.shape[0]
    adj = infer_bonds(atomic_numbers, coords, tolerance)
    colors = refine_colors(atomic_numbers, adj, n_iter)

    # Union-find over atoms that are colour-equal and share a neighbour.
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for k in range(n):
        # Neighbours of heavy atom k that are colour-equal to each other.
        neigh = np.where(adj[k])[0]
        for ii in range(len(neigh)):
            for jj in range(ii + 1, len(neigh)):
                a, b = int(neigh[ii]), int(neigh[jj])
                if colors[a] == colors[b]:
                    union(a, b)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in groups.values()]
