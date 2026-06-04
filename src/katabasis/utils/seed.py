"""Global seeding for reproducible experiments (cross-cutting concern: reproducibility)."""

from __future__ import annotations

import contextlib
import os
import random

import numpy as np


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed python, numpy and (if available) torch.

    Parameters
    ----------
    seed:
        The integer seed.
    deterministic_torch:
        If True and torch is installed, request deterministic cuDNN/algorithms.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # torch optional for pure-numpy code paths / tests
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        # use_deterministic_algorithms can raise for ops without a deterministic
        # implementation; we keep it best-effort so CPU smoke tests never break.
        with contextlib.suppress(RuntimeError, AttributeError):
            torch.use_deterministic_algorithms(True, warn_only=True)
