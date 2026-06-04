"""Synthetic reaction generator.

Produces small :class:`Reaction` objects that satisfy every structural
invariant of Halo8 (shared frame, consistent atom indexing, single-barrier NEB
profile, a CH3-like group to exercise connectivity-constrained permutation) so
the entire pipeline and the training loops can be smoke-tested on CPU without
downloading the real dataset. Not chemically meaningful -- a *fixture*, not data.
"""

from __future__ import annotations

import numpy as np

from katabasis.data.arclength import cumulative_arclength
from katabasis.data.types import Reaction


def _methyl_scaffold(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A C with three H (interchangeable) plus an O and an H, all bonded sanely.

    Returns ``(atomic_numbers (N,), coords (N, 3))`` for a stable-ish geometry.
    Indices: 0=C, 1,2,3=H(methyl), 4=O, 5=H(hydroxyl-ish).
    """
    z = np.array([6, 1, 1, 1, 8, 1], dtype=np.int64)
    # Methanol-like: C at origin, O along +x; the three methyl H point into the
    # -x hemisphere (tetrahedral, ~1.09 A, 109.5 deg from the C-O axis) so they
    # are bonded only to C and genuinely interchangeable. The hydroxyl H sits
    # well away from the methyl Hs so connectivity inference keeps them distinct.
    coords = np.array(
        [
            [0.00, 0.00, 0.000],  # 0 C
            [-0.364, 1.028, 0.000],  # 1 H (methyl)
            [-0.364, -0.514, 0.890],  # 2 H (methyl)
            [-0.364, -0.514, -0.890],  # 3 H (methyl)
            [1.43, 0.00, 0.000],  # 4 O
            [2.23, 0.60, 0.000],  # 5 H (hydroxyl)
        ],
        dtype=np.float64,
    )
    coords = coords + 0.02 * rng.standard_normal(coords.shape)
    return z, coords


def make_synthetic_reaction(
    reaction_id: str,
    *,
    seed: int = 0,
    n_images: int = 11,
    barrier: float = 20.0,
    subset: str = "t1x_core",
    multistep: bool = False,
    with_forces: bool = True,
) -> Reaction:
    """Build one synthetic single-step (or, if requested, multi-step) reaction."""
    rng = np.random.default_rng(seed)
    z, r = _methyl_scaffold(rng)

    # Product: stretch the C-O bond (a toy "bond breaking") + small wiggle.
    p = r.copy()
    p[4] += np.array([0.9, 0.0, 0.0])  # move O out along x
    p[5] += np.array([1.1, -0.3, 0.0])  # the O-H follows
    p += 0.03 * rng.standard_normal(p.shape)

    # Saddle: near the midpoint but displaced perpendicular to the R->P motion,
    # so M = (R+P)/2 is NOT the saddle (the load-bearing subtlety).
    m = 0.5 * (r + p)
    perp = np.zeros_like(m)
    perp[4] = np.array([0.0, 0.5, 0.0])
    saddle = m + perp + 0.02 * rng.standard_normal(m.shape)

    # NEB band: R -> saddle -> P, images interpolated through the saddle.
    half = n_images // 2
    left = np.linspace(0.0, 1.0, half + 1)[:-1]  # excludes saddle
    right = np.linspace(0.0, 1.0, n_images - half)
    imgs = [(1 - a) * r + a * saddle for a in left]
    imgs += [(1 - a) * saddle + a * p for a in right]
    images = np.stack(imgs, axis=0)
    images += 0.005 * rng.standard_normal(images.shape)
    saddle_idx = half  # index of the saddle image in the band

    # Energy profile: single barrier (parabola peaking at the saddle) unless
    # multistep is requested, in which case inject an interior minimum.
    k = images.shape[0]
    x = np.linspace(-1.0, 1.0, k)
    energies = barrier * (1.0 - x**2)  # 0 at endpoints, `barrier` at center
    energies = energies - energies[0]  # reference to reactant
    if multistep:
        # Carve a deep dip on the R-side to mimic a hidden intermediate.
        energies[half // 2] -= 0.5 * barrier
    energies += 0.001 * rng.standard_normal(k)

    forces = None
    if with_forces:
        # Toy "forces" = negative finite-difference gradient of a quadratic well;
        # only used to exercise code paths, not physically meaningful.
        forces = -0.1 * (images - saddle)

    reaction = Reaction(
        reaction_id=reaction_id,
        atomic_numbers=z,
        reactant=r,
        product=p,
        saddle=saddle,
        neb_images=images,
        neb_energies=energies,
        forces=forces,
        saddle_image_index=saddle_idx,
        barrier=float(energies.max()),
        subset=subset,
        metadata={"synthetic": True, "energy_unit": "kcal/mol"},
    )
    reaction.validate()
    # arclength sanity: legs are well-formed
    _ = cumulative_arclength(reaction.neb_images)
    return reaction


def make_synthetic_dataset(
    n_reactions: int = 16, *, seed: int = 0, subset: str = "t1x_core"
) -> list[Reaction]:
    """A small list of synthetic reactions for end-to-end smoke training."""
    return [
        make_synthetic_reaction(
            f"{subset}_rxn_{i:04d}", seed=seed + i, barrier=15.0 + 5.0 * (i % 4), subset=subset
        )
        for i in range(n_reactions)
    ]
