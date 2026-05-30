from __future__ import annotations

import numpy as np

from kevlargrid.solver.damping import viscous_damping


class TestDamping:
    """Tests for viscous and Rayleigh damping models."""

    def test_viscous_damping_opposes_velocity(self) -> None:
        """Verify that the damping force vector directly opposes the velocity vector."""
        velocities = np.array([[1.0, 2.0, -3.0]])
        coefficient = 0.5

        forces = viscous_damping(velocities, coefficient)
        # Should be -0.5 * velocities
        np.testing.assert_allclose(forces[0], np.array([-0.5, -1.0, 1.5]))

    def test_viscous_damping_magnitude(self) -> None:
        """Verify the magnitude of the damping force for known coefficient and velocity."""
        velocities = np.array([[3.0, 4.0, 0.0]])
        coefficient = 2.0

        forces = viscous_damping(velocities, coefficient)
        # ||v|| = 5.0 -> ||F|| = 2.0 * 5.0 = 10.0
        assert np.abs(np.linalg.norm(forces[0]) - 10.0) < 1e-7

    def test_zero_velocity_zero_damping(self) -> None:
        """Verify that zero velocity results in zero damping force."""
        velocities = np.array([[0.0, 0.0, 0.0]])
        coefficient = 10.0

        forces = viscous_damping(velocities, coefficient)
        np.testing.assert_allclose(forces, np.zeros_like(velocities))
