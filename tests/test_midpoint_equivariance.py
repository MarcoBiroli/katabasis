"""Phase-0 canary: midpoint equivariance (Critical req. 1, 2)."""

from __future__ import annotations

import numpy as np

from katabasis.data.midpoint import midpoint
from katabasis.utils import random_reflection, random_rotation, random_translation


def test_midpoint_rotation_translation_equivariance(reaction, rng):
    Q = random_rotation(rng)
    t = random_translation(rng=rng)
    M = midpoint(reaction.reactant, reaction.product)
    M2 = midpoint(reaction.reactant @ Q.T + t, reaction.product @ Q.T + t)
    assert np.abs((M @ Q.T + t) - M2).max() < 1e-10


def test_midpoint_reflection_equivariance(reaction, rng):
    F = random_reflection(rng)
    M = midpoint(reaction.reactant, reaction.product)
    M2 = midpoint(reaction.reactant @ F.T, reaction.product @ F.T)
    assert np.abs((M @ F.T) - M2).max() < 1e-10


def test_midpoint_is_rp_symmetric(reaction):
    a = midpoint(reaction.reactant, reaction.product)
    b = midpoint(reaction.product, reaction.reactant)
    assert np.abs(a - b).max() < 1e-12
