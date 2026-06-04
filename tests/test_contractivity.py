"""Contractivity surrogate: must estimate the numerical abscissa, not the
spectral radius or spectral abscissa (project risk: wrong surrogate)."""

from __future__ import annotations

import pytest
import torch

from katabasis.losses.contractivity import hutchinson_trace, numerical_abscissa


@pytest.fixture(autouse=True)
def _f64():
    torch.set_default_dtype(torch.float64)


def _linear_field(matrices, batch):
    def f(x):
        out = torch.empty_like(x)
        for g, A in enumerate(matrices):
            m = batch == g
            out[m] = x[m] @ A.T
        return out

    return f


def test_recovers_numerical_abscissa_diagonal():
    A = torch.diag(torch.tensor([0.5, -1.0, -2.0]))  # sym lambda_max = 0.5
    batch = torch.tensor([0, 0])
    f = _linear_field([A], batch)
    x = torch.randn(2, 3, requires_grad=True)
    mu = numerical_abscissa(f, x, batch, 1, n_iter=40)
    assert abs(float(mu[0]) - 0.5) < 1e-3


def test_distinguishes_numerical_abscissa_from_spectral_radius():
    # Non-normal A: eigenvalues all negative (spectral abscissa < 0) but the
    # symmetric part has a positive eigenvalue, so it is NOT contractive.
    A = torch.tensor([[-0.1, 5.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, -0.1]])
    sym = 0.5 * (A + A.T)
    true_mu = float(torch.linalg.eigvalsh(sym).max())
    assert true_mu > 0  # genuinely non-contractive despite negative eigenvalues
    spectral_abscissa = float(torch.linalg.eigvals(A).real.max())
    assert spectral_abscissa < 0  # the trap: would falsely pass

    batch = torch.tensor([0, 0])
    f = _linear_field([A], batch)
    x = torch.randn(2, 3, requires_grad=True)
    mu = numerical_abscissa(f, x, batch, 1, n_iter=60)
    assert abs(float(mu[0]) - true_mu) < 1e-2
    assert float(mu[0]) > 0  # surrogate correctly flags the expansive mode


def test_per_graph_independence():
    A0 = torch.diag(torch.tensor([-1.0, -2.0, -3.0]))  # mu = -1
    A1 = torch.diag(torch.tensor([1.0, -1.0, -1.0]))  # mu = +1
    batch = torch.tensor([0, 0, 1, 1])
    f = _linear_field([A0, A1], batch)
    x = torch.randn(4, 3, requires_grad=True)
    mu = numerical_abscissa(f, x, batch, 2, n_iter=40)
    assert abs(float(mu[0]) - (-1.0)) < 1e-3
    assert abs(float(mu[1]) - 1.0) < 1e-3


def test_hutchinson_trace_matches():
    A = torch.tensor([[-2.0, 0.3, 0.0], [0.1, -1.0, 0.0], [0.0, 0.0, -0.5]])
    batch = torch.tensor([0, 0])
    f = _linear_field([A], batch)
    x = torch.randn(2, 3, requires_grad=True)
    # Two atoms per graph -> trace of the block-diagonal Jacobian = 2 * tr(A).
    est = hutchinson_trace(f, x, batch, 1, n_samples=200)
    assert abs(float(est[0]) - 2 * float(torch.trace(A))) < 0.5
