"""Core data containers shared across the data pipeline.

A :class:`Reaction` holds one single-step reaction with a *consistent atom
indexing* across R, P, the saddle, and every NEB image (Critical correctness
requirement 1). All coordinates are ``(N, 3)`` numpy arrays in a shared frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Reaction:
    """A single single-step reaction with a shared coordinate frame.

    Attributes
    ----------
    reaction_id:
        Stable identifier; the unit of train/val/test splitting.
    atomic_numbers:
        ``(N,)`` int array, identical for R, P, saddle and all NEB images.
    reactant / product / saddle:
        ``(N, 3)`` geometries.
    neb_images:
        ``(K, N, 3)`` images along the minimum energy path, ordered R-side ->
        P-side (index 0 nearest R). The saddle index is recorded separately.
    neb_energies:
        ``(K,)`` per-image energies (eV or kcal/mol; unit tracked in metadata).
    forces:
        Optional ``(K, N, 3)`` per-image forces (ωB97X-3c), if stored.
    saddle_image_index:
        Index into ``neb_images`` of the image closest to the saddle (the
        energy maximum along the band), used to split the band into two legs.
    barrier:
        Forward barrier height (max energy minus reactant energy).
    subset:
        One of ``{"t1x_core", "halogen_only"}``; ``full`` is the union.
    metadata:
        Free-form provenance (units, source row index, element set, ...).
    """

    reaction_id: str
    atomic_numbers: np.ndarray  # (N,)
    reactant: np.ndarray  # (N, 3)
    product: np.ndarray  # (N, 3)
    saddle: np.ndarray  # (N, 3)
    neb_images: np.ndarray  # (K, N, 3)
    neb_energies: np.ndarray  # (K,)
    forces: np.ndarray | None = None  # (K, N, 3) or None
    saddle_image_index: int = -1
    barrier: float = float("nan")
    subset: str = "t1x_core"
    metadata: dict = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return int(self.atomic_numbers.shape[0])

    @property
    def n_images(self) -> int:
        return int(self.neb_images.shape[0])

    def midpoint(self) -> np.ndarray:
        """The R<->P symmetric midpoint ``M = (R + P) / 2``. See data.midpoint."""
        from katabasis.data.midpoint import midpoint

        return midpoint(self.reactant, self.product)

    def validate(self) -> None:
        """Cheap structural invariants; raises ``ValueError`` on violation."""
        n = self.n_atoms
        for name, arr, shape in [
            ("reactant", self.reactant, (n, 3)),
            ("product", self.product, (n, 3)),
            ("saddle", self.saddle, (n, 3)),
        ]:
            if arr.shape != shape:
                raise ValueError(f"{name} shape {arr.shape} != {shape} for {self.reaction_id}")
        if self.neb_images.ndim != 3 or self.neb_images.shape[1:] != (n, 3):
            raise ValueError(f"neb_images shape {self.neb_images.shape} incompatible with N={n}")
        if self.neb_energies.shape[0] != self.n_images:
            raise ValueError("neb_energies length must match number of NEB images")
        if self.forces is not None and self.forces.shape != self.neb_images.shape:
            raise ValueError("forces shape must match neb_images shape")
