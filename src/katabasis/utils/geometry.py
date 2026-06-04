"""Rigid-motion generators used across equivariance tests and augmentation.

All functions return plain numpy ``float64`` arrays so they are framework
agnostic; callers convert to torch as needed. A rotation/reflection is a
``(3, 3)`` matrix ``Q`` acting on row-vector coordinates as ``x @ Q.T``.
"""

from __future__ import annotations

import numpy as np


def random_rotation(rng: np.random.Generator | None = None) -> np.ndarray:
    """Uniform random proper rotation in SO(3) via QR of a Gaussian matrix.

    Returns
    -------
    (3, 3) float64 with ``det == +1``.
    """
    rng = np.random.default_rng() if rng is None else rng
    a = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(a)
    # Fix the sign ambiguity of QR so the result is Haar-uniform.
    d = np.sign(np.diag(r))
    d[d == 0] = 1.0
    q = q * d
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def random_reflection(rng: np.random.Generator | None = None) -> np.ndarray:
    """A random improper orthogonal transform (``det == -1``) in O(3) \\ SO(3)."""
    q = random_rotation(rng)
    q[:, 0] = -q[:, 0]  # flip one column -> determinant becomes -1
    return q


def random_translation(scale: float = 1.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """A random ``(3,)`` translation vector."""
    rng = np.random.default_rng() if rng is None else rng
    return rng.standard_normal(3) * scale


def apply_transform(
    coords: np.ndarray, rot: np.ndarray, translation: np.ndarray | None = None
) -> np.ndarray:
    """Apply ``x -> x @ rot.T + t`` to ``(..., 3)`` coordinates."""
    out = coords @ rot.T
    if translation is not None:
        out = out + translation
    return out
