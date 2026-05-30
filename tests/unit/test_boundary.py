from __future__ import annotations

import numpy as np

from kevlargrid.solver.boundary import apply_clamped_boundary, compute_min_radius


class TestBoundary:
    """Unit tests for the boundary conditions module."""

    def test_apply_clamped_boundary(self) -> None:
        """Verify that clamped boundary nodes have their velocities zeroed."""
        velocities = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ]
        )
        # Clamp nodes 0 and 3, leave 1 and 2 free
        boundary_mask = np.array([True, False, False, True])

        updated = apply_clamped_boundary(velocities, boundary_mask)

        np.testing.assert_allclose(updated[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(updated[1], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(updated[2], [7.0, 8.0, 9.0])
        np.testing.assert_allclose(updated[3], [0.0, 0.0, 0.0])

    def test_compute_min_radius(self) -> None:
        """Verify that the minimum radius calculation matches expectations."""
        wave_speed = 2000.0  # m/s
        sim_duration = 0.005  # 5 ms
        safety_factor = 1.5

        r_min = compute_min_radius(wave_speed, sim_duration, safety_factor)
        # R_min = 2000 * 0.005 * 1.5 = 10 * 1.5 = 15.0 metres
        assert r_min == 15.0
