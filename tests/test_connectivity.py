"""Connectivity-constrained interchangeable grouping (Critical req. 3).

The load-bearing guard: only genuinely interchangeable atoms group together,
and H atoms on *different* heavy atoms must never share a group.
"""

from __future__ import annotations

import numpy as np

from katabasis.data.connectivity import infer_bonds, interchangeable_groups


def test_methyl_hydrogens_group_together(reaction):
    groups = interchangeable_groups(reaction.atomic_numbers, reaction.reactant)
    # Indices 1,2,3 are the three methyl H of the synthetic methanol-like scaffold.
    methyl = next((g for g in groups if set(g) == {1, 2, 3}), None)
    assert methyl is not None, f"methyl H did not group: {groups}"


def test_hydrogens_on_different_heavy_atoms_do_not_group():
    # Two CH3 groups far apart: ethane-like, mirror methyls. H on C0 must never
    # group with H on C1.
    z = np.array([6, 1, 1, 1, 6, 1, 1, 1], dtype=np.int64)
    coords = np.array(
        [
            [0.0, 0.0, 0.0],  # C0
            [-0.36, 1.03, 0.0],
            [-0.36, -0.51, 0.89],
            [-0.36, -0.51, -0.89],
            [1.5, 0.0, 0.0],  # C1
            [1.86, 1.03, 0.0],
            [1.86, -0.51, 0.89],
            [1.86, -0.51, -0.89],
        ]
    )
    groups = interchangeable_groups(z, coords)
    for g in groups:
        h_set = set(g)
        assert not (
            h_set & {1, 2, 3} and h_set & {5, 6, 7}
        ), f"H across different carbons grouped: {groups}"
    # Each methyl's three H should still group within its own carbon.
    assert any(set(g) == {1, 2, 3} for g in groups)
    assert any(set(g) == {5, 6, 7} for g in groups)


def test_groups_partition_all_atoms(reaction):
    groups = interchangeable_groups(reaction.atomic_numbers, reaction.reactant)
    flat = sorted(i for g in groups for i in g)
    assert flat == list(range(reaction.n_atoms))


def test_infer_bonds_symmetric(reaction):
    adj = infer_bonds(reaction.atomic_numbers, reaction.reactant)
    assert np.array_equal(adj, adj.T)
    assert not adj.diagonal().any()
