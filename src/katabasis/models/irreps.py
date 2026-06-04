"""Symmetry-group configuration and irreps construction.

CLAUDE.md "Equivariance scope": we pin the intended group in config. A point
cloud carrying only positions yields, through equivariant tensor products,
*natural-parity* features (l even -> e, l odd -> o), so an e3nn backbone built
this way is O(3)-equivariant and therefore also SE(3)-equivariant. The group
choice here controls **what the equivariance test asserts**:

- ``"O3"``  (parity-aware): reflections are a symmetry; the equivariance test
  includes reflections.
- ``"SE3"`` (chirality-preserving, default): we only *require* rotation +
  translation equivariance; the reflection test is deliberately excluded,
  because forcing reflection-equivariance would map a molecule to the TS of its
  mirror image. (Genuinely discriminating enantiomers further requires
  injecting a pseudoscalar feature; flagged as a documented v1 limitation.)

Either way the network we build is mathematically the same; the contract is in
the tests, exactly as the plan prescribes.
"""

from __future__ import annotations

from enum import StrEnum

from e3nn import o3


class Group(StrEnum):
    SE3 = "SE3"  # rotations + translations (chirality-preserving)
    O3 = "O3"  # + reflections (parity-aware)

    @property
    def test_includes_reflections(self) -> bool:
        return self is Group.O3


def feature_irreps(
    n_scalars: int,
    n_vectors: int,
    n_tensors: int,
) -> o3.Irreps:
    """Hidden node-feature irreps: scalars (0e), vectors (1o), rank-2 (2e)."""
    parts = []
    if n_scalars:
        parts.append(f"{n_scalars}x0e")
    if n_vectors:
        parts.append(f"{n_vectors}x1o")
    if n_tensors:
        parts.append(f"{n_tensors}x2e")
    return o3.Irreps("+".join(parts))


def sh_irreps(l_max: int) -> o3.Irreps:
    """Spherical-harmonic irreps used to embed edge directions."""
    return o3.Irreps.spherical_harmonics(l_max)


# A true displacement (difference of positions) is a parity-odd vector.
OUTPUT_VECTOR_IRREPS = o3.Irreps("1x1o")
