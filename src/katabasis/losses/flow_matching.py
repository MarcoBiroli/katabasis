"""Flow-matching loss for Network B with velocity-scale normalization.

The raw target ``dx/dt = L * unit_tangent`` scales with leg length ``L``, which
spans orders of magnitude; left alone the loss is dominated by long-path
reactions. We therefore regress the **direction** (unit tangent, scale-free)
and the **scalar speed** ``L`` separately, then combine (CLAUDE.md "velocity
scale normalization").
"""

from __future__ import annotations

import torch


def flow_matching_loss(
    pred_direction: torch.Tensor,  # (N, 3) network direction (not yet unit)
    pred_speed: torch.Tensor,  # (N, 1) network per-atom speed
    unit_tangent: torch.Tensor,  # (N, 3) target direction (Frobenius unit per graph)
    speed: torch.Tensor,  # (B,) per-graph leg length L
    batch: torch.Tensor,  # (N,) graph index
    *,
    direction_weight: float = 1.0,
    speed_weight: float = 1.0,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Return total + component losses for direction and (log) speed.

    The direction term is scale-free: predicted directions are normalized
    per-graph (Frobenius) before comparison to the unit tangent. The speed term
    matches per-atom speed to the per-graph leg length in log space so short and
    long legs contribute comparably.
    """
    # Normalize predicted direction per graph (Frobenius over the graph's atoms).
    sq = (pred_direction**2).sum(dim=1)  # (N,)
    graph_norm = torch.zeros(int(batch.max()) + 1, device=pred_direction.device)
    graph_norm.index_add_(0, batch, sq)
    graph_norm = graph_norm.sqrt().clamp_min(eps)  # (B,)
    pred_dir_unit = pred_direction / graph_norm[batch][:, None]

    direction_loss = ((pred_dir_unit - unit_tangent) ** 2).sum(dim=1).mean()

    # Speed: per-atom predicted speed vs per-graph L, matched in log space.
    target_speed = speed[batch][:, None].clamp_min(eps)  # (N, 1)
    speed_loss = (torch.log(pred_speed.clamp_min(eps)) - torch.log(target_speed)) ** 2
    speed_loss = speed_loss.mean()

    total = direction_weight * direction_loss + speed_weight * speed_loss
    return {
        "loss": total,
        "direction_loss": direction_loss.detach(),
        "speed_loss": speed_loss.detach(),
    }
