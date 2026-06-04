"""Kabsch + connectivity-constrained permutation alignment."""

from __future__ import annotations

from katabasis.data.alignment import aligned_rmsd, kabsch_align, permutation_align
from katabasis.utils import random_rotation, random_translation


def test_kabsch_recovers_rigid_motion(reaction, rng):
    Q = random_rotation(rng)
    t = random_translation(rng=rng)
    moved = reaction.reactant @ Q.T + t
    _, rmsd = kabsch_align(moved, reaction.reactant)
    assert rmsd < 1e-8


def test_permutation_alignment_invariant_to_methyl_permutation(reaction, rng):
    # Permute the three methyl H of the product; aligned RMSD to reactant must
    # be unchanged because they are interchangeable.
    base = aligned_rmsd(reaction.product, reaction.reactant, reaction.atomic_numbers)
    permuted = reaction.product.copy()
    permuted[[1, 2, 3]] = permuted[[3, 1, 2]]
    after = aligned_rmsd(permuted, reaction.reactant, reaction.atomic_numbers)
    assert abs(base - after) < 1e-6


def test_alignment_is_rotation_invariant(reaction, rng):
    Q = random_rotation(rng)
    rotated = reaction.product @ Q.T
    a = aligned_rmsd(reaction.product, reaction.reactant, reaction.atomic_numbers)
    b = aligned_rmsd(rotated, reaction.reactant, reaction.atomic_numbers)
    assert abs(a - b) < 1e-6


def test_permutation_returns_valid_permutation(reaction):
    perm, _, _ = permutation_align(reaction.product, reaction.reactant, reaction.atomic_numbers)
    assert sorted(perm.tolist()) == list(range(reaction.n_atoms))
