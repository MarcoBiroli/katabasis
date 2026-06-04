"""Halo8 reader: assemble reactions from an ASE database and cache round-trip.

Builds a tiny synthetic ASE db with the documented schema, parses it, and
checks the reaction is assembled with consistent indexing and a single-barrier
saddle. (No real download required.)"""

from __future__ import annotations

import numpy as np
import pytest

ase_db = pytest.importorskip("ase.db")
from ase import Atoms  # noqa: E402
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402

from katabasis.data.halo8 import Halo8Reader  # noqa: E402
from katabasis.data.synthetic import make_synthetic_reaction  # noqa: E402


def _write_db(path):
    r = make_synthetic_reaction("rxnA", seed=0, n_images=7)
    db = ase_db.connect(str(path))

    def add(coords, role, idx=None, energy=0.0):
        atoms = Atoms(numbers=r.atomic_numbers, positions=coords)
        # Energy on the calculator (as Halo8 does); "energy" is a reserved kvp key.
        atoms.calc = SinglePointCalculator(atoms, energy=float(energy))
        kvp = {"reaction_id": "rxnA", "role": role}
        if idx is not None:
            kvp["image_index"] = idx
        db.write(atoms, key_value_pairs=kvp)

    add(r.reactant, "reactant", energy=0.0)
    add(r.product, "product", energy=0.0)
    add(r.saddle, "ts", energy=float(r.neb_energies.max()))
    for i in range(r.n_images):
        add(r.neb_images[i], "neb", idx=i, energy=float(r.neb_energies[i]))
    return r


def test_halo8_assemble_and_cache(tmp_path):
    db_path = tmp_path / "halo8.db"
    cache = tmp_path / "cache"
    truth = _write_db(db_path)

    reader = Halo8Reader(db_path, cache)
    ids = reader.build_cache()
    assert ids == ["rxnA"]

    reactions = list(reader.iter_reactions())
    assert len(reactions) == 1
    rx = reactions[0]
    assert rx.n_atoms == truth.n_atoms
    assert np.array_equal(rx.atomic_numbers, truth.atomic_numbers)
    assert rx.n_images == truth.n_images
    # Saddle image index lands on the energy maximum.
    assert rx.saddle_image_index == int(np.argmax(rx.neb_energies))


def test_halo8_cache_roundtrip(tmp_path):
    db_path = tmp_path / "halo8.db"
    cache = tmp_path / "cache"
    _write_db(db_path)
    reader = Halo8Reader(db_path, cache)
    reader.build_cache()
    loaded = Halo8Reader.load_cached(cache / "rxnA.npz")
    loaded.validate()
    assert loaded.reaction_id == "rxnA"
