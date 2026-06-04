"""Fixed-step integration of Network B's velocity field.

RK4 with ~32 steps is the default (CLAUDE.md). The integrator is written in
pure torch so it is differentiable end-to-end: Network A's auxiliary loss
backpropagates through descents run with a frozen B, and evaluation reuses the
same code path.
"""

from __future__ import annotations

import torch

from katabasis.data.batching import GraphBatch
from katabasis.models.network_b import NetworkB


def integrate_flow(
    net_b: NetworkB,
    x0: torch.Tensor,  # (N, 3) start state
    target: torch.Tensor,  # (N, 3) basin endpoint
    graph: GraphBatch,
    *,
    n_steps: int = 32,
    method: str = "rk4",
    return_trajectory: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Integrate ``dx/dt = v(x, target, t)`` from ``t=0`` to ``t=1``.

    Returns the final state ``(N, 3)`` (and the ``(n_steps+1, N, 3)`` trajectory
    if requested). Time is broadcast to nodes; the same scalar ``t`` is used for
    every atom of a molecule (per-graph time).
    """
    dt = 1.0 / n_steps
    x = x0
    traj = [x0]

    def vel(state: torch.Tensor, t_scalar: float) -> torch.Tensor:
        t_node = torch.full((state.shape[0],), t_scalar, device=state.device, dtype=state.dtype)
        return net_b(state, target, t_node, graph)

    for i in range(n_steps):
        t0 = i * dt
        if method == "euler":
            x = x + dt * vel(x, t0)
        elif method == "rk4":
            k1 = vel(x, t0)
            k2 = vel(x + 0.5 * dt * k1, t0 + 0.5 * dt)
            k3 = vel(x + 0.5 * dt * k2, t0 + 0.5 * dt)
            k4 = vel(x + dt * k3, t0 + dt)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"unknown integrator method: {method}")
        if return_trajectory:
            traj.append(x)

    if return_trajectory:
        return x, torch.stack(traj, dim=0)
    return x


def steepest_descent_on_forces(
    forces_fn,
    x0: torch.Tensor,
    *,
    n_steps: int = 200,
    step_size: float = 0.01,
) -> torch.Tensor:
    """Non-learned physics baseline: integrate steepest descent on real forces.

    ``forces_fn(x) -> (N, 3)`` returns ωB97X-3c forces. Used to ground the
    descent-flow premise against physics (PLAN.md Phase 3).
    """
    x = x0.clone()
    for _ in range(n_steps):
        x = x + step_size * forces_fn(x)
    return x
