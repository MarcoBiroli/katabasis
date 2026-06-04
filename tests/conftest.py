from __future__ import annotations

import numpy as np
import pytest
import torch

from katabasis.data.synthetic import make_synthetic_reaction


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    """Isolate torch's global default dtype so float64 tests don't leak."""
    old = torch.get_default_dtype()
    yield
    torch.set_default_dtype(old)


@pytest.fixture
def use_float64():
    torch.set_default_dtype(torch.float64)
    return torch.float64


@pytest.fixture
def reaction():
    return make_synthetic_reaction("test_rxn", seed=1)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
