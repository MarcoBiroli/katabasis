"""Halo8 ASE-database reader with per-reaction disk caching.

Halo8 (Zenodo 16737590) ships as an ASE database of ~20M structures from
~19k reaction pathways. Rows are grouped into reactions and each reaction is
assembled into a :class:`Reaction` (shared frame, consistent atom indexing).

The exact ASE ``key_value_pairs`` schema is dataset-specific. This reader is
deliberately defensive: the relevant key names are configurable
(:class:`Halo8Schema`) with sensible fallbacks, and the role of each row
(reactant / product / transition-state / NEB image) is resolved from a small
set of candidate keys. **Verify against the real database** before trusting the
loader on the full set (PLAN.md risk register: manually inspect ~50 reactions).

Parsed reactions are cached one ``.npz`` per reaction so repeated runs are fast.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from katabasis.data.types import Reaction

# kcal/mol per eV, for normalizing barrier units if the db stores eV.
EV_TO_KCAL = 23.060541945


@dataclass
class Halo8Schema:
    """Configurable mapping from ASE ``key_value_pairs`` to reaction structure."""

    reaction_id_keys: tuple[str, ...] = ("reaction_id", "rxn_id", "reaction", "pathway_id")
    role_keys: tuple[str, ...] = ("role", "type", "structure_type", "label")
    image_index_keys: tuple[str, ...] = ("image_index", "neb_image", "image", "frame")
    energy_keys: tuple[str, ...] = ("energy", "total_energy", "E")
    subset_keys: tuple[str, ...] = ("subset", "source", "origin")
    reactant_roles: tuple[str, ...] = ("reactant", "r", "initial", "reac")
    product_roles: tuple[str, ...] = ("product", "p", "final", "prod")
    saddle_roles: tuple[str, ...] = ("ts", "saddle", "transition_state", "transitionstate")
    neb_roles: tuple[str, ...] = ("neb", "image", "band", "mep")
    halogen_elements: tuple[int, ...] = (9, 17, 35)  # F, Cl, Br
    energy_unit: str = "eV"  # converted to kcal/mol for the barrier field


def _first_present(kvp: dict, keys: tuple[str, ...]) -> object | None:
    for k in keys:
        if k in kvp and kvp[k] is not None:
            return kvp[k]
    return None


def _classify_role(value: object, schema: Halo8Schema) -> str:
    v = str(value).strip().lower()
    for role, names in [
        ("reactant", schema.reactant_roles),
        ("product", schema.product_roles),
        ("saddle", schema.saddle_roles),
        ("neb", schema.neb_roles),
    ]:
        if v in names or any(v.startswith(n) for n in names):
            return role
    return "neb"  # default: treat unknown rows as band images


@dataclass
class Halo8Reader:
    """Reads reactions from an ASE database file and caches them to disk."""

    db_path: str | Path
    cache_dir: str | Path
    schema: Halo8Schema = field(default_factory=Halo8Schema)

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- grouping ---------------------------------------------------------
    def _group_rows(self) -> dict[str, list]:
        """Group ASE rows by reaction id (kept in memory only as references)."""
        from ase.db import connect

        groups: dict[str, list] = {}
        with connect(str(self.db_path)) as db:
            for row in db.select():
                kvp = dict(row.key_value_pairs)
                rid = _first_present(kvp, self.schema.reaction_id_keys)
                if rid is None:
                    continue
                groups.setdefault(str(rid), []).append(row)
        return groups

    # -- assembly ---------------------------------------------------------
    def _assemble(self, reaction_id: str, rows: list) -> Reaction | None:
        reactant = product = saddle = None
        atomic_numbers = None
        neb: list[tuple[int, np.ndarray, float, np.ndarray | None]] = []
        subset = "t1x_core"

        for row in rows:
            kvp = dict(row.key_value_pairs)
            coords = np.asarray(row.positions, dtype=np.float64)
            z = np.asarray(row.numbers, dtype=np.int64)
            if atomic_numbers is None:
                atomic_numbers = z
            # Energy/forces normally live in the ASE calculator results
            # (row.energy / row.forces); fall back to key_value_pairs. Note ASE
            # reserves "energy" as a kvp key, so real data stores it on the calc.
            energy = _first_present(kvp, self.schema.energy_keys)
            if energy is None:
                energy = row.get("energy")
            energy = float(energy) if energy is not None else float("nan")
            forces = np.asarray(row.forces, dtype=np.float64) if "forces" in row else None
            role = _classify_role(_first_present(kvp, self.schema.role_keys), self.schema)
            subset_val = _first_present(kvp, self.schema.subset_keys)
            if subset_val is not None:
                subset = str(subset_val)

            if role == "reactant":
                reactant = coords
            elif role == "product":
                product = coords
            elif role == "saddle":
                saddle = coords
            else:
                idx = _first_present(kvp, self.schema.image_index_keys)
                neb.append((int(idx) if idx is not None else len(neb), coords, energy, forces))

        if reactant is None or product is None or saddle is None or not neb:
            return None  # incomplete reaction; caller counts these

        neb.sort(key=lambda x: x[0])
        neb_images = np.stack([c for _, c, _, _ in neb], axis=0)
        neb_energies = np.array([e for _, _, e, _ in neb], dtype=np.float64)
        have_forces = all(f is not None for _, _, _, f in neb)
        forces = np.stack([f for _, _, _, f in neb], axis=0) if have_forces else None

        saddle_idx = (
            int(np.argmax(neb_energies)) if np.isfinite(neb_energies).all() else len(neb) // 2
        )
        e_ref = neb_energies[0]
        barrier = float(np.nanmax(neb_energies) - e_ref)
        if self.schema.energy_unit == "eV":
            barrier *= EV_TO_KCAL

        # Infer subset from elements if not annotated.
        if subset == "t1x_core" and np.isin(atomic_numbers, self.schema.halogen_elements).any():
            subset = "halogen_only"

        reaction = Reaction(
            reaction_id=reaction_id,
            atomic_numbers=atomic_numbers,
            reactant=reactant,
            product=product,
            saddle=saddle,
            neb_images=neb_images,
            neb_energies=neb_energies,
            forces=forces,
            saddle_image_index=saddle_idx,
            barrier=barrier,
            subset=subset,
            metadata={"source": "halo8", "energy_unit": self.schema.energy_unit},
        )
        reaction.validate()
        return reaction

    # -- caching ----------------------------------------------------------
    def _cache_path(self, reaction_id: str) -> Path:
        safe = reaction_id.replace("/", "_")
        return self.cache_dir / f"{safe}.npz"

    def _save(self, reaction: Reaction) -> None:
        path = self._cache_path(reaction.reaction_id)
        np.savez_compressed(
            path,
            reaction_id=reaction.reaction_id,
            atomic_numbers=reaction.atomic_numbers,
            reactant=reaction.reactant,
            product=reaction.product,
            saddle=reaction.saddle,
            neb_images=reaction.neb_images,
            neb_energies=reaction.neb_energies,
            forces=reaction.forces if reaction.forces is not None else np.array([]),
            saddle_image_index=reaction.saddle_image_index,
            barrier=reaction.barrier,
            subset=reaction.subset,
            metadata=json.dumps(reaction.metadata),
        )

    @staticmethod
    def load_cached(path: str | Path) -> Reaction:
        d = np.load(path, allow_pickle=False)
        forces = d["forces"]
        return Reaction(
            reaction_id=str(d["reaction_id"]),
            atomic_numbers=d["atomic_numbers"],
            reactant=d["reactant"],
            product=d["product"],
            saddle=d["saddle"],
            neb_images=d["neb_images"],
            neb_energies=d["neb_energies"],
            forces=forces if forces.size else None,
            saddle_image_index=int(d["saddle_image_index"]),
            barrier=float(d["barrier"]),
            subset=str(d["subset"]),
            metadata=json.loads(str(d["metadata"])),
        )

    def build_cache(self) -> list[str]:
        """Parse the whole database into the cache; returns cached reaction ids."""
        cached: list[str] = []
        for rid, rows in self._group_rows().items():
            reaction = self._assemble(rid, rows)
            if reaction is not None:
                self._save(reaction)
                cached.append(rid)
        return cached

    def iter_reactions(self, reaction_ids: list[str] | None = None) -> Iterator[Reaction]:
        """Yield reactions from cache, building the cache first if it is empty."""
        existing = sorted(p.stem for p in self.cache_dir.glob("*.npz"))
        if not existing:
            self.build_cache()
            existing = sorted(p.stem for p in self.cache_dir.glob("*.npz"))
        wanted = set(reaction_ids) if reaction_ids is not None else None
        for stem in existing:
            if wanted is not None and stem not in wanted:
                continue
            yield self.load_cached(self.cache_dir / f"{stem}.npz")
