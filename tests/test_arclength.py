"""Arc-length bridge construction, tangents, and velocity-scale normalization."""

from __future__ import annotations

import numpy as np

from katabasis.data.arclength import (
    bridge_tangent,
    build_legs,
    interpolate_bridge,
)
from katabasis.utils import random_rotation


def test_cumulative_arclength_monotone(reaction):
    legs = build_legs(reaction)
    for leg in legs:
        arc = leg.arclength
        assert arc[0] == 0.0
        assert np.all(np.diff(arc) >= -1e-12)


def test_legs_start_at_saddle_end_at_endpoint(reaction):
    legs = build_legs(reaction)
    labels = {leg.endpoint_label for leg in legs}
    assert labels == {"R", "P"}
    for leg in legs:
        np.testing.assert_allclose(leg.positions[0], reaction.saddle)


def test_bridge_endpoints(reaction):
    legs = build_legs(reaction)
    for leg in legs:
        np.testing.assert_allclose(interpolate_bridge(leg, 0.0), leg.positions[0], atol=1e-10)
        np.testing.assert_allclose(interpolate_bridge(leg, 1.0), leg.positions[-1], atol=1e-10)


def test_unit_tangent_is_normalized(reaction):
    legs = build_legs(reaction)
    for leg in legs:
        for t in (0.1, 0.5, 0.9):
            unit, dxdt, speed = bridge_tangent(leg, t)
            assert abs(np.sqrt((unit**2).sum()) - 1.0) < 1e-8
            # dx/dt = speed * unit_tangent; speed == leg length L.
            np.testing.assert_allclose(dxdt, speed * unit, atol=1e-10)
            assert abs(speed - leg.total_length) < 1e-8


def test_bridge_equivariance_under_rotation(reaction, rng):
    """Linear interpolation in a shared frame is rotation-equivariant."""
    Q = random_rotation(rng)
    legs = build_legs(reaction)
    leg = legs[0]
    x_t = interpolate_bridge(leg, 0.4)

    rot = type(leg)(
        positions=leg.positions @ Q.T,
        arclength=leg.arclength.copy(),
        endpoint_label=leg.endpoint_label,
    )
    x_t_rot = interpolate_bridge(rot, 0.4)
    np.testing.assert_allclose(x_t @ Q.T, x_t_rot, atol=1e-10)
