"""Katabasis: physics-informed equivariant transition-state and pathway prediction.

The package is organized into:

- :mod:`katabasis.data`   data loading, alignment, midpoint, splits, filters, arc length
- :mod:`katabasis.models` equivariant Network A (saddle) and Network B (descent flow)
- :mod:`katabasis.losses` RMSD, flow-matching, contractivity surrogate
- :mod:`katabasis.train`  the phase training loops
- :mod:`katabasis.eval`   flow integration and metrics
- :mod:`katabasis.utils`  geometry helpers and seeding
"""

from __future__ import annotations

__version__ = "0.1.0"
