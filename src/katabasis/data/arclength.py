"""Arc-length parameterized descent legs and flow-matching bridge construction.

Network B is supervised by flow matching along the NEB descent path
(CLAUDE.md "Training objective for Network B"). The band is split at the saddle
into two *descent legs* (saddle -> R, saddle -> P), each parameterized by
**arc length** (not NEB-image index, which is optimizer-noise). ``t = 0`` is the
saddle, ``t = 1`` the basin endpoint.

Velocity-scale normalization: the raw target ``dx/dt = L * unit_tangent`` scales
with the leg length ``L``, which spans orders of magnitude across reactions. We
therefore expose the *unit tangent* (direction) and the scalar *speed* ``L``
separately so the trainer can regress a scale-free direction plus a scalar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from katabasis.data.types import Reaction


@dataclass(slots=True)
class DescentLeg:
    """One descent leg from the saddle (t=0) into a basin endpoint (t=1)."""

    positions: np.ndarray  # (M, N, 3) ordered saddle -> endpoint
    arclength: np.ndarray  # (M,) cumulative arc length, arclength[0] == 0
    endpoint_label: str  # "R" or "P"

    @property
    def total_length(self) -> float:
        return float(self.arclength[-1])

    @property
    def t(self) -> np.ndarray:
        L = self.total_length
        return self.arclength / L if L > 0 else np.zeros_like(self.arclength)

    @property
    def endpoint(self) -> np.ndarray:
        return self.positions[-1]


def frame_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance in the shared 3N coordinate frame (no re-alignment).

    The legs come from a single NEB band that already shares a frame, so we must
    *not* re-align consecutive images -- doing so would remove genuine motion.
    """
    return float(np.sqrt(np.sum((a - b) ** 2)))


def cumulative_arclength(positions: np.ndarray) -> np.ndarray:
    """``(M, N, 3) -> (M,)`` cumulative arc length, starting at 0."""
    seg = np.sqrt(np.sum(np.diff(positions, axis=0) ** 2, axis=(1, 2)))  # (M-1,)
    return np.concatenate([[0.0], np.cumsum(seg)])


def build_legs(
    reaction: Reaction,
    *,
    include_endpoints: bool = True,
) -> list[DescentLeg]:
    """Split the NEB band at the saddle into the two descent legs.

    The leg toward R is ``[saddle, <images R-side, saddle->R order>, R]`` and
    likewise toward P. If ``include_endpoints`` is False the R/P anchor points
    are omitted (use when the NEB band already terminates at R and P).
    """
    s_idx = reaction.saddle_image_index
    if not (0 <= s_idx < reaction.n_images):
        # Fall back: use the energy-max image as the saddle split point.
        s_idx = int(np.argmax(reaction.neb_energies))

    images = reaction.neb_images
    saddle = reaction.saddle

    # R-side: images with index < s_idx, ordered from saddle outward (descending).
    r_side = [images[i] for i in range(s_idx - 1, -1, -1)]
    # P-side: images with index > s_idx, ordered from saddle outward (ascending).
    p_side = [images[i] for i in range(s_idx + 1, reaction.n_images)]

    legs = []
    for side, label, endpoint in [
        (r_side, "R", reaction.reactant),
        (p_side, "P", reaction.product),
    ]:
        pts = [saddle, *side]
        if include_endpoints:
            pts.append(endpoint)
        pos = np.stack(pts, axis=0)
        legs.append(
            DescentLeg(positions=pos, arclength=cumulative_arclength(pos), endpoint_label=label)
        )
    return legs


def interpolate_bridge(leg: DescentLeg, t: float) -> np.ndarray:
    """Piecewise-linear-in-arc-length interpolation ``x_t`` at ``t in [0, 1]``."""
    L = leg.total_length
    if L <= 0:
        return leg.positions[0].copy()
    s = float(np.clip(t, 0.0, 1.0)) * L
    arc = leg.arclength
    j = int(np.searchsorted(arc, s, side="right")) - 1
    j = max(0, min(j, len(arc) - 2))
    seg = arc[j + 1] - arc[j]
    frac = 0.0 if seg <= 0 else (s - arc[j]) / seg
    return (1.0 - frac) * leg.positions[j] + frac * leg.positions[j + 1]


def bridge_tangent(leg: DescentLeg, t: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(unit_tangent, dxdt, speed)`` for the bridge at ``t``.

    With piecewise-linear-in-arc-length geometry the path has unit speed in
    arc length, so ``dx/ds`` is the unit tangent of the bracketing segment and
    ``dx/dt = L * dx/ds``. ``speed = L`` is the scalar used for velocity-scale
    normalization. ``unit_tangent`` has Frobenius norm 1 (up to degenerate legs).
    """
    L = leg.total_length
    if L <= 0:
        z = np.zeros_like(leg.positions[0])
        return z, z, 0.0
    s = float(np.clip(t, 0.0, 1.0)) * L
    arc = leg.arclength
    j = int(np.searchsorted(arc, s, side="right")) - 1
    j = max(0, min(j, len(arc) - 2))
    delta = leg.positions[j + 1] - leg.positions[j]
    seg_len = arc[j + 1] - arc[j]
    unit = delta / seg_len if seg_len > 0 else np.zeros_like(delta)
    dxdt = L * unit
    return unit, dxdt, L


def measure_image_spacing(reactions: list[Reaction]) -> float:
    """Median NEB image spacing (Frobenius), for sizing B's noise injection."""
    spacings = []
    for r in reactions:
        d = np.sqrt(np.sum(np.diff(r.neb_images, axis=0) ** 2, axis=(1, 2)))
        spacings.extend(d.tolist())
    return float(np.median(spacings)) if spacings else 0.0
