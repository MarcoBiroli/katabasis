"""Reaction-level train/val/test splits with persisted, hashed assignments.

Splits are **by reaction**, never by image: every NEB image of a held-out
reaction must land in the same split or validation leaks (CLAUDE.md). The split
assignment is persisted to JSON together with a content hash so experiments are
reproducible (cross-cutting: reproducibility).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DEFAULT_FRACTIONS = (0.8, 0.1, 0.1)  # train / val / test


@dataclass(frozen=True)
class Split:
    """Reaction-id assignment for one subset."""

    subset: str
    seed: int
    fractions: tuple[float, float, float]
    train: list[str]
    val: list[str]
    test: list[str]

    def hash(self) -> str:
        payload = json.dumps(
            {"train": sorted(self.train), "val": sorted(self.val), "test": sorted(self.test)},
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    @property
    def all_ids(self) -> list[str]:
        return [*self.train, *self.val, *self.test]


def make_split(
    reaction_ids: list[str],
    subset: str,
    seed: int = 0,
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
) -> Split:
    """Deterministically partition reaction ids into train/val/test by reaction."""
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    ids = sorted(set(reaction_ids))  # dedupe + determinism before shuffle
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    ids = [ids[i] for i in perm]
    n = len(ids)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    train = ids[:n_train]
    val = ids[n_train : n_train + n_val]
    test = ids[n_train + n_val :]
    return Split(
        subset=subset, seed=seed, fractions=tuple(fractions), train=train, val=val, test=test
    )


def assert_no_leakage(split: Split) -> None:
    """Raise if any reaction id appears in more than one split (Crit. req. 5)."""
    s_train, s_val, s_test = set(split.train), set(split.val), set(split.test)
    overlaps = {
        "train/val": s_train & s_val,
        "train/test": s_train & s_test,
        "val/test": s_val & s_test,
    }
    leaked = {k: sorted(v) for k, v in overlaps.items() if v}
    if leaked:
        raise ValueError(f"Split leakage detected: {leaked}")


def save_split(split: Split, path: str | Path) -> str:
    """Persist split to JSON; returns the content hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(split)
    data["hash"] = split.hash()
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return data["hash"]


def load_split(path: str | Path) -> Split:
    data = json.loads(Path(path).read_text())
    stored_hash = data.pop("hash", None)
    data["fractions"] = tuple(data["fractions"])
    split = Split(**data)
    if stored_hash is not None and stored_hash != split.hash():
        raise ValueError(f"Split hash mismatch for {path}: data was modified after saving")
    return split
