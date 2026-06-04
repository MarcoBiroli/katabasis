"""Evaluation metrics: saddle RMSD, path fidelity, cycle consistency, topology.

All RMSDs are connectivity-constrained and Kabsch-aligned (never raw Cartesian).
Path fidelity guards against the target-conditioned-flow tautology: a flow that
teleports to the target along a non-physical path still scores well on endpoint
RMSD, which would invalidate pathway generation (CLAUDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from katabasis.data.alignment import permutation_align
from katabasis.data.arclength import DescentLeg, interpolate_bridge
from katabasis.data.connectivity import interchangeable_groups


def saddle_rmsd(pred: np.ndarray, true_saddle: np.ndarray, z: np.ndarray, groups=None) -> float:
    """Connectivity-constrained aligned RMSD of a predicted saddle."""
    if groups is None:
        groups = interchangeable_groups(z, true_saddle)
    _, _, rmsd = permutation_align(pred, true_saddle, z, groups)
    return rmsd


def path_fidelity(
    traj: np.ndarray,  # (S+1, N, 3) integrated trajectory, t in [0, 1] uniform in step
    leg: DescentLeg,
    z: np.ndarray,
    *,
    t_samples: tuple[float, ...] = (0.25, 0.5, 0.75),
    groups=None,
) -> dict[str, float]:
    """Intermediate-path RMSD: integrated ``x_t`` vs arc-length-matched NEB image.

    The trajectory is uniform in integration step (== uniform in ``t`` for a
    fixed-step solver), so sample it at the same ``t`` as the arc-length bridge.
    """
    if groups is None:
        groups = interchangeable_groups(z, leg.endpoint)
    n_steps = traj.shape[0] - 1
    out = {}
    for t in t_samples:
        idx = int(round(t * n_steps))
        x_integrated = traj[idx]
        x_reference = interpolate_bridge(leg, t)
        _, _, rmsd = permutation_align(x_integrated, x_reference, z, groups)
        out[f"path_rmsd_t{t}"] = rmsd
    out["path_rmsd_mean"] = float(np.mean(list(out.values())))
    return out


def cycle_consistency_error(
    descent_endpoint_R: np.ndarray,
    descent_endpoint_P: np.ndarray,
    true_R: np.ndarray,
    true_P: np.ndarray,
    z: np.ndarray,
    groups_R=None,
    groups_P=None,
) -> float:
    """Mean RMSD of the two descent endpoints vs ground-truth R and P.

    This is the confidence score: it correlates with saddle prediction error
    and is available at inference (no ground-truth saddle needed).
    """
    if groups_R is None:
        groups_R = interchangeable_groups(z, true_R)
    if groups_P is None:
        groups_P = interchangeable_groups(z, true_P)
    _, _, er = permutation_align(descent_endpoint_R, true_R, z, groups_R)
    _, _, ep = permutation_align(descent_endpoint_P, true_P, z, groups_P)
    return 0.5 * (er + ep)


@dataclass
class CycleConsistencyStats:
    """Correlation between cycle-consistency error and true saddle error."""

    pearson: float
    spearman: float
    n: int


def cycle_consistency_correlation(
    cycle_errors: np.ndarray, saddle_errors: np.ndarray
) -> CycleConsistencyStats:
    """Correlate the (inference-available) cycle error with the true saddle error.

    Positive correlation -> the cycle error is a usable confidence score.
    """
    from scipy.stats import pearsonr, spearmanr

    ce = np.asarray(cycle_errors)
    se = np.asarray(saddle_errors)
    pear = float(pearsonr(ce, se)[0]) if ce.size > 1 else float("nan")
    spear = float(spearmanr(ce, se)[0]) if ce.size > 1 else float("nan")
    return CycleConsistencyStats(pearson=pear, spearman=spear, n=int(ce.size))


def fraction_single_negative_mode(
    hessian_eigenvalues: list[np.ndarray], tol: float = 1e-6
) -> float:
    """Fraction of geometries with exactly one negative Hessian eigenvalue.

    A valid first-order saddle has exactly one negative curvature direction
    (after projecting out trivial translation/rotation modes upstream).
    """
    if not hessian_eigenvalues:
        return float("nan")
    ok = sum(1 for ev in hessian_eigenvalues if int((np.asarray(ev) < -tol).sum()) == 1)
    return ok / len(hessian_eigenvalues)
