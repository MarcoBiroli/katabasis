"""Contractivity surrogate: penalize the **numerical abscissa**, not the
spectral radius.

Contraction (monotone convergence of nearby trajectories) is governed by
``mu(J_v) = lambda_max(sym(J_v))`` with ``sym(M) = (M + M^T)/2`` -- the numerical
abscissa -- because ``||exp(J_v tau)|| <= exp(mu(J_v) tau)``. Two traps
(CLAUDE.md):

1. plain power iteration on ``J_v`` returns the largest-*magnitude* eigenvalue
   (spectral radius), which controls nothing relevant;
2. even the largest-*real-part* eigenvalue (spectral abscissa) being negative
   does not imply contraction -- equivariant Jacobians are non-normal and can
   grow transiently.

So we estimate ``mu`` by power iteration on the **symmetrized** Jacobian-vector
product ``u -> 1/2 (J_v u + J_v^T u)`` (both directions via autograd), using a
two-stage shift to recover the most-positive eigenvalue (not the largest in
magnitude). A Hutchinson estimator on ``tr(J_v)`` is provided as a cheap
averaged-contractivity signal.

The estimate adds extra backward passes; the trainer applies it on a batch
subsample and/or every k steps (PLAN.md cost control).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

VelocityFn = Callable[[torch.Tensor], torch.Tensor]


def _graph_dot(a: torch.Tensor, b: torch.Tensor, batch: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros(n, device=a.device, dtype=a.dtype)
    out.index_add_(0, batch, (a * b).sum(dim=1))
    return out


def _graph_norm(a: torch.Tensor, batch: torch.Tensor, n: int, eps: float = 1e-12) -> torch.Tensor:
    return _graph_dot(a, a, batch, n).clamp_min(eps).sqrt()


def jacobian_products(
    f: VelocityFn, x: torch.Tensor, u: torch.Tensor, *, create_graph: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(J u, J^T u)`` for ``v = f(x)`` via autograd.

    ``J^T u`` is a standard VJP. ``J u`` uses the double-backward trick: define
    ``g(w) = J^T w``; then ``d/dw [g . u]`` evaluated by a second grad gives
    ``J u``.
    """
    x = x.requires_grad_(True)
    v = f(x)
    # J^T u  (vector-Jacobian product)
    jt_u = torch.autograd.grad(v, x, grad_outputs=u, create_graph=True, retain_graph=True)[0]
    # J u via double backward through a dummy variable.
    w = torch.zeros_like(v, requires_grad=True)
    jt_w = torch.autograd.grad(v, x, grad_outputs=w, create_graph=True, retain_graph=True)[0]
    j_u = torch.autograd.grad(
        jt_w, w, grad_outputs=u, create_graph=create_graph, retain_graph=True
    )[0]
    if not create_graph:
        jt_u = jt_u.detach()
        j_u = j_u.detach()
    return j_u, jt_u


def symmetrized_jvp(
    f: VelocityFn, x: torch.Tensor, u: torch.Tensor, *, create_graph: bool
) -> torch.Tensor:
    """``S u = 1/2 (J u + J^T u)`` -- the action of ``sym(J_v)`` on ``u``."""
    j_u, jt_u = jacobian_products(f, x, u, create_graph=create_graph)
    return 0.5 * (j_u + jt_u)


def numerical_abscissa(
    f: VelocityFn,
    x: torch.Tensor,  # (N, 3) evaluation point
    batch: torch.Tensor,  # (N,) graph index
    n_graphs: int,
    *,
    n_iter: int = 8,
    create_graph: bool = False,
    seed: int | None = None,
) -> torch.Tensor:
    """Per-graph estimate of ``lambda_max(sym(J_v))`` (N_graphs,).

    Two-stage shifted power iteration on the symmetrized operator:
    (1) plain iteration estimates the per-graph spectral radius ``rho`` of
    ``sym(J_v)``; (2) iteration on ``sym(J_v) + rho I`` makes the spectrum
    non-negative, so its dominant eigenvalue is ``lambda_max + rho``; subtract
    ``rho`` to recover the most-positive eigenvalue.
    """
    gen = None
    if seed is not None:
        gen = torch.Generator(device=x.device).manual_seed(seed)
    u = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
    u = u / _graph_norm(u, batch, n_graphs)[batch][:, None]

    # Stage 1: spectral radius of sym(J) (largest magnitude eigenvalue).
    rho = torch.zeros(n_graphs, device=x.device, dtype=x.dtype)
    for _ in range(n_iter):
        su = symmetrized_jvp(f, x, u, create_graph=False)
        norm = _graph_norm(su, batch, n_graphs)
        rho = norm  # Rayleigh on a unit vector of a symmetric op -> |lambda|
        u = su / norm[batch][:, None]

    # Stage 2: shifted iteration to extract the most-positive eigenvalue.
    u2 = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
    u2 = u2 / _graph_norm(u2, batch, n_graphs)[batch][:, None]
    lam_shift = torch.zeros(n_graphs, device=x.device, dtype=x.dtype)
    for it in range(n_iter):
        last = it == n_iter - 1
        su = symmetrized_jvp(f, x, u2, create_graph=create_graph and last)
        shifted = su + rho[batch][:, None] * u2  # (sym(J) + rho I) u
        lam_shift = _graph_dot(u2, shifted, batch, n_graphs)  # Rayleigh quotient
        u2 = shifted / _graph_norm(shifted, batch, n_graphs)[batch][:, None]
    return lam_shift - rho  # lambda_max(sym(J))


def hutchinson_trace(
    f: VelocityFn,
    x: torch.Tensor,
    batch: torch.Tensor,
    n_graphs: int,
    *,
    n_samples: int = 4,
    create_graph: bool = False,
) -> torch.Tensor:
    """Per-graph Hutchinson estimate of ``tr(J_v)`` (averaged contractivity)."""
    est = torch.zeros(n_graphs, device=x.device, dtype=x.dtype)
    for _ in range(n_samples):
        z = torch.randint(0, 2, x.shape, device=x.device, dtype=x.dtype) * 2 - 1  # Rademacher
        _, jt_z = jacobian_products(f, x, z, create_graph=create_graph)
        est = est + _graph_dot(z, jt_z, batch, n_graphs)
    return est / n_samples


def contractivity_penalty(
    f: VelocityFn,
    x: torch.Tensor,
    batch: torch.Tensor,
    n_graphs: int,
    *,
    n_iter: int = 8,
    create_graph: bool = True,
) -> torch.Tensor:
    """Scalar penalty ``E[ ReLU(lambda_max(sym(J_v))) ]`` over graphs."""
    mu = numerical_abscissa(f, x, batch, n_graphs, n_iter=n_iter, create_graph=create_graph)
    return torch.relu(mu).mean()
