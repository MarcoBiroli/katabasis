"""Flow-matching loss: velocity-scale normalization keeps short and long legs
balanced (project risk: velocity-scale imbalance)."""

from __future__ import annotations

import pytest
import torch

from katabasis.losses.flow_matching import flow_matching_loss


@pytest.fixture(autouse=True)
def _f64():
    torch.set_default_dtype(torch.float64)


def test_direction_loss_is_scale_free():
    # Two graphs whose target directions are identical but leg lengths differ by
    # 100x. With scale-free direction matching the perfectly-aligned prediction
    # has zero direction loss regardless of L.
    batch = torch.tensor([0, 0, 1, 1])
    unit = torch.tensor([[1.0, 0, 0], [0, 0, 0], [1.0, 0, 0], [0, 0, 0]])
    speed = torch.tensor([0.01, 1.0])  # tiny vs large leg
    pred_dir = unit.clone() * 7.0  # arbitrary positive scale
    pred_speed = speed[batch][:, None]
    out = flow_matching_loss(pred_dir, pred_speed, unit, speed, batch)
    assert float(out["direction_loss"]) < 1e-10
    assert float(out["speed_loss"]) < 1e-10


def test_speed_loss_penalizes_wrong_scale():
    batch = torch.tensor([0, 0])
    unit = torch.tensor([[1.0, 0, 0], [0, 0, 0]])
    speed = torch.tensor([2.0])
    pred_dir = unit.clone()
    wrong_speed = torch.full((2, 1), 20.0)  # 10x too large
    out = flow_matching_loss(pred_dir, wrong_speed, unit, speed, batch)
    assert float(out["speed_loss"]) > 0.1


def test_loss_is_finite_and_differentiable():
    batch = torch.tensor([0, 0, 0])
    unit = torch.randn(3, 3)
    unit = unit / unit.norm()
    speed = torch.tensor([1.5])
    pred_dir = torch.randn(3, 3, requires_grad=True)
    pred_speed = torch.rand(3, 1, requires_grad=True) + 0.1
    out = flow_matching_loss(pred_dir, pred_speed, unit, speed, batch)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert pred_dir.grad is not None
