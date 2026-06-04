"""Finite-difference Hessian and curvature analysis for saddle topology checks.

A first-order saddle has exactly one negative Hessian eigenvalue along the
bond-making/breaking mode. Halo8 may store Hessians; if not, this computes a
small-displacement finite-difference Hessian from a forces function and removes
the trivial translation/rotation modes before counting negative eigenvalues.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from ase.data import atomic_masses

ForcesFn = Callable[[np.ndarray], np.ndarray]  # (N,3) -> (N,3) forces


def finite_difference_hessian(
    forces_fn: ForcesFn, x: np.ndarray, delta: float = 1e-3
) -> np.ndarray:
    """Central-difference Hessian ``(3N, 3N)`` from forces ``F = -dE/dx``.

    ``H_ij = -dF_i/dx_j``; symmetrized to remove finite-difference asymmetry.
    """
    n = x.size
    flat = x.reshape(-1)
    h = np.zeros((n, n))
    for j in range(n):
        xp = flat.copy()
        xm = flat.copy()
        xp[j] += delta
        xm[j] -= delta
        fp = forces_fn(xp.reshape(x.shape)).reshape(-1)
        fm = forces_fn(xm.reshape(x.shape)).reshape(-1)
        h[:, j] = -(fp - fm) / (2 * delta)
    return 0.5 * (h + h.T)


def _trans_rot_basis(x: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Mass-weighted translation/rotation null-space basis ``(3N, k)`` (orthonormal)."""
    n = x.shape[0]
    sqrt_m = np.sqrt(np.repeat(masses, 3))
    vecs = []
    # Translations.
    for d in range(3):
        t = np.zeros((n, 3))
        t[:, d] = 1.0
        vecs.append(t.reshape(-1) * sqrt_m)
    # Infinitesimal rotations about the centroid.
    c = x - x.mean(0)
    for axis in np.eye(3):
        r = np.cross(np.tile(axis, (n, 1)), c)
        vecs.append(r.reshape(-1) * sqrt_m)
    basis = np.stack(vecs, axis=1)
    q, _ = np.linalg.qr(basis)
    return q


def curvature_eigenvalues(
    hessian: np.ndarray, x: np.ndarray, atomic_numbers: np.ndarray
) -> np.ndarray:
    """Eigenvalues of the mass-weighted Hessian with trans/rot modes projected out."""
    masses = atomic_masses[atomic_numbers]
    sqrt_m = np.sqrt(np.repeat(masses, 3))
    mw = hessian / np.outer(sqrt_m, sqrt_m)  # mass-weighted
    basis = _trans_rot_basis(x, masses)
    proj = np.eye(mw.shape[0]) - basis @ basis.T
    mw_proj = proj @ mw @ proj
    eigvals = np.linalg.eigvalsh(0.5 * (mw_proj + mw_proj.T))
    # Drop the (near-zero) eigenvalues corresponding to the removed modes.
    return np.sort(eigvals)[basis.shape[1] :]


def has_single_negative_mode(
    forces_fn: ForcesFn, x: np.ndarray, atomic_numbers: np.ndarray, *, tol: float = 1e-6
) -> bool:
    """True iff the projected Hessian has exactly one negative eigenvalue."""
    h = finite_difference_hessian(forces_fn, x)
    ev = curvature_eigenvalues(h, x, atomic_numbers)
    return int((ev < -tol).sum()) == 1
