"""The R<->P symmetric midpoint ``M = (R + P) / 2``.

This is intentionally trivial, but it is the *canary* for end-to-end
equivariance (Phase 0 smoke test) and it is load-bearing: M is an R<->P
symmetric initialization that Network A corrects, **not** an approximation of
the saddle (CLAUDE.md "things to avoid").

Equivariance note: ``(R + P) / 2`` is SE(3)/O(3)-equivariant *only* if R and P
already share a frame. Halo8 should satisfy this within a reaction, but the
data pipeline Kabsch-aligns P onto R defensively before constructing M.
"""

from __future__ import annotations

import numpy as np


def midpoint(reactant: np.ndarray, product: np.ndarray) -> np.ndarray:
    """``M = (R + P) / 2``; inputs and output are ``(N, 3)`` in a shared frame."""
    if reactant.shape != product.shape:
        raise ValueError(f"R/P shape mismatch: {reactant.shape} vs {product.shape}")
    return 0.5 * (reactant + product)
