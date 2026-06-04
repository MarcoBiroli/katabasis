"""Pathological-reaction filters (PLAN.md Phase 1).

Removes reactions that violate the single-saddle factorization or are not real
chemistry: low barriers, tiny R<->P change, and multi-step (multi-barrier /
kinked) NEB profiles. The single-dominant-negative-Hessian-mode check is
optional and only run when forces/Hessians are available.

All thresholds come from config -- no magic numbers in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from katabasis.data.alignment import aligned_rmsd
from katabasis.data.types import Reaction


@dataclass
class FilterConfig:
    """Thresholds for Phase-1 filtering. Defaults are PLAN.md starting points."""

    min_barrier: float = 5.0  # kcal/mol; conformer-only "reactions" below this
    min_rp_rmsd: float = 0.5  # Angstrom; no real chemistry below this
    # Fraction of the barrier height tolerated as non-monotonic noise before a
    # dip is treated as a genuine intermediate minimum (multi-step).
    monotonic_noise_frac: float = 0.05
    require_single_barrier: bool = True
    max_removed_frac: float = 0.30  # acceptance gate: filtering removes <30%


@dataclass
class FilterReport:
    """Counts of removed reactions by reason; for documenting the filter."""

    total: int = 0
    kept: int = 0
    removed_low_barrier: int = 0
    removed_small_rp: int = 0
    removed_multistep: int = 0
    removed_ids: dict[str, list[str]] = field(default_factory=dict)

    @property
    def removed(self) -> int:
        return self.total - self.kept

    @property
    def removed_frac(self) -> float:
        return self.removed / self.total if self.total else 0.0

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "kept": self.kept,
            "removed": self.removed,
            "removed_frac": round(self.removed_frac, 4),
            "removed_low_barrier": self.removed_low_barrier,
            "removed_small_rp": self.removed_small_rp,
            "removed_multistep": self.removed_multistep,
        }


def is_single_barrier(energies: np.ndarray, noise_frac: float) -> bool:
    """True if the NEB energy profile is single-barrier (one interior maximum).

    The profile must rise (modulo noise) to a single dominant maximum and then
    fall to the other endpoint. We detect a hidden intermediate minimum by
    counting interior local minima whose depth exceeds ``noise_frac`` of the
    total energy span; any such dip flags a multi-step reaction.
    """
    e = np.asarray(energies, dtype=float)
    if e.shape[0] < 3:
        return True  # too few images to judge; let other filters decide
    span = float(e.max() - e.min())
    if span <= 0:
        return False
    tol = noise_frac * span
    argmax = int(np.argmax(e))
    if argmax == 0 or argmax == e.shape[0] - 1:
        return False  # maximum at an endpoint -> not a barrier between R and P

    # Count interior local minima deeper than tolerance relative to neighbours.
    interior_min = 0
    for i in range(1, e.shape[0] - 1):
        if e[i] < e[i - 1] - tol and e[i] < e[i + 1] - tol:
            interior_min += 1
    return interior_min == 0


def passes_filters(reaction: Reaction, cfg: FilterConfig) -> tuple[bool, str | None]:
    """Return ``(keep, reason_if_removed)`` for one reaction."""
    if np.isfinite(reaction.barrier) and reaction.barrier < cfg.min_barrier:
        return False, "low_barrier"

    rp = aligned_rmsd(
        reaction.product,
        reaction.reactant,
        reaction.atomic_numbers,
        reference_coords=reaction.reactant,
    )
    if rp < cfg.min_rp_rmsd:
        return False, "small_rp"

    if cfg.require_single_barrier and not is_single_barrier(
        reaction.neb_energies, cfg.monotonic_noise_frac
    ):
        return False, "multistep"

    return True, None


def apply_filters(
    reactions: list[Reaction], cfg: FilterConfig
) -> tuple[list[Reaction], FilterReport]:
    """Filter a list of reactions, returning the kept set and a report."""
    report = FilterReport(total=len(reactions))
    reason_to_field = {
        "low_barrier": "removed_low_barrier",
        "small_rp": "removed_small_rp",
        "multistep": "removed_multistep",
    }
    report.removed_ids = {k: [] for k in reason_to_field}
    kept: list[Reaction] = []
    for r in reactions:
        keep, reason = passes_filters(r, cfg)
        if keep:
            kept.append(r)
        else:
            setattr(report, reason_to_field[reason], getattr(report, reason_to_field[reason]) + 1)
            report.removed_ids[reason].append(r.reaction_id)
    report.kept = len(kept)
    return kept, report
